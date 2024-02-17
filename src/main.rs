use ndarray::{s, Array1, Array2, Axis};
use polars::prelude::*;
use rand::{
    seq::{IteratorRandom, SliceRandom},
    Rng,
};
use rayon::prelude::*;
use std::{env, fs, path::PathBuf};

use std::{fs::File, sync::Arc};

use serde::Deserialize;

#[derive(Deserialize)]
struct Config {
    genetic_algorithm: GeneticAlgorithmConfig,
}

#[derive(Deserialize)]
struct GeneticAlgorithmConfig {
    pop_size: usize,
    k: usize,
    mutation_rate: f64,
    generations: usize,
    num_parents: usize,
    num_top_parents: usize,
}

type Population = Vec<Array2<f64>>;

type FitnessFunction = Arc<dyn Fn(&Array2<f64>, &Array1<f64>, &Array1<f64>) -> f64 + Send + Sync>;
type SelectionMethodFunction = Arc<dyn Fn(&[Array2<f64>], &[f64]) -> Array2<f64> + Send + Sync>;

type WriterFunction = Arc<dyn Fn(&Array2<f64>, f64, &DataFrame) + Send + Sync>;

fn initialize_population_from_solution(
    df: &DataFrame,
    pop_size: usize,
    n: usize,
    buckets: usize,
) -> Population {
    let mut initial_population = Vec::with_capacity(pop_size);

    let bucket_series = df.column("bucket").unwrap().i64().unwrap();

    for _ in 0..pop_size {
        let mut x = Array2::<f64>::zeros((n, buckets));

        for (i, bucket_value) in bucket_series.into_iter().enumerate() {
            if let Some(bucket) = bucket_value {
                let bucket_index = (bucket - 1) as usize; // assuming 1-indexed buckets
                if bucket_index < buckets {
                    x[(i, bucket_index)] = 1.0;
                }
            }
        }

        initial_population.push(x);
    }

    initial_population
}

fn calculate_objective(x: &Array2<f64>, costs: &Array1<f64>, discounts: &Array1<f64>) -> f64 {
    let total_costs_per_bucket = x.t().dot(costs);
    let total_discounts_per_bucket = x.t().dot(discounts);

    let items_per_bucket: Array1<f64> = x.t().sum_axis(Axis(1));

    let avg_discounts_per_bucket = (total_discounts_per_bucket / items_per_bucket).mapv(
        // round to 2 decimal places
        |x| (x * 100.0).round() / 100.0,
    );

    let discount_costs_per_bucket = total_costs_per_bucket * avg_discounts_per_bucket;

    discount_costs_per_bucket.sum()
}

fn k_point_crossover(parents: &[Array2<f64>], k: usize) -> Array2<f64> {
    let num_parents = parents.len();
    assert!(
        num_parents >= 2,
        "There must be at least two parents for crossover."
    );

    let (n_rows, n_cols) = parents[0].dim();
    let mut rng = rand::thread_rng();

    // Generate sorted crossover points
    let mut crossover_points: Vec<usize> = (1..n_rows - 1).collect();
    crossover_points.shuffle(&mut rng);
    crossover_points.truncate(k);
    crossover_points.sort_unstable();

    // Initialize the child matrix with zeros
    let mut child = Array2::<f64>::zeros((n_rows, n_cols));

    // Perform crossover operations
    let mut start_idx = 0;
    for (i, &end_idx) in crossover_points.iter().chain(&[n_rows]).enumerate() {
        let parent_idx = i % num_parents;
        for row in start_idx..end_idx {
            child
                .slice_mut(s![row, ..])
                .assign(&parents[parent_idx].slice(s![row, ..]));
        }
        start_idx = end_idx;
    }

    child
}

fn mutation(x: &mut Array2<f64>, mutation_rate: f64) {
    let (n_rows, n_cols) = x.dim();
    let mut rng = rand::thread_rng();

    for i in 0..n_rows {
        for j in 0..n_cols {
            if rng.gen::<f64>() < mutation_rate {
                x[(i, j)] = 0.0;
                let new_col = rng.gen_range(0..n_cols);
                x[(i, new_col)] = 1.0;
            }
        }
    }
}

fn repair_constraint(x: &mut Array2<f64>) {
    let mut rng = rand::thread_rng();

    for i in 0..x.nrows() {
        if x.row(i).sum() > 1.0 {
            let mut indices: Vec<_> = x
                .row(i)
                .indexed_iter()
                .filter(|&(_, &v)| v > 0.0)
                .map(|(idx, _)| idx)
                .collect();

            indices.shuffle(&mut rng);

            for &idx in indices.iter().skip(1) {
                x[(i, idx)] = 0.0;
            }
        }
    }
}

fn tournament_selection(
    population: &[Array2<f64>],
    fitnesses: &[f64],
    tournament_size: usize,
) -> Array2<f64> {
    let mut rng = rand::thread_rng();
    let selected_indices: Vec<_> = (0..population.len()).choose_multiple(&mut rng, tournament_size);
    let &best_index = selected_indices
        .iter()
        .max_by(|&&x, &&y| fitnesses[x].partial_cmp(&fitnesses[y]).unwrap())
        .unwrap();

    population[best_index].clone()
}

fn roulette_wheel_selection(population: &[Array2<f64>], fitnesses: &[f64]) -> Array2<f64> {
    let total_fitness: f64 = fitnesses.iter().sum();
    let probabilities: Vec<f64> = fitnesses.iter().map(|&f| f / total_fitness).collect();
    let mut rng = rand::thread_rng();
    let mut cumulative_probabilities = vec![0.0; probabilities.len()];
    probabilities
        .iter()
        .enumerate()
        .fold(0.0, |acc, (i, &prob)| {
            cumulative_probabilities[i] = acc + prob;
            acc + prob
        });

    let choice = rng.gen::<f64>();
    let selected_index = cumulative_probabilities
        .iter()
        .position(|&p| p >= choice)
        .unwrap_or(0);

    population[selected_index].clone()
}

fn rank_selection(population: &[Array2<f64>], fitnesses: &[f64]) -> Array2<f64> {
    let mut rng = rand::thread_rng();

    // Pair each fitness with its index, sort by fitness, then calculate rank-based probabilities.
    let mut indexed_fitnesses: Vec<(usize, &f64)> = fitnesses.iter().enumerate().collect();
    // Sort by descending order of fitness
    indexed_fitnesses.sort_unstable_by(|a, b| b.1.partial_cmp(a.1).unwrap());

    let total_ranks: usize = (1..=indexed_fitnesses.len()).sum();
    let probabilities: Vec<f64> = indexed_fitnesses
        .iter()
        .enumerate()
        .map(|(rank, _)| (rank + 1) as f64 / total_ranks as f64)
        .collect();

    // Calculate cumulative probabilities for roulette wheel selection
    let mut cumulative_probabilities = vec![0.0; probabilities.len()];
    probabilities.iter().fold(0.0, |acc, &prob| {
        let cumulative = acc + prob;
        cumulative_probabilities.push(cumulative);
        cumulative
    });

    // Draw a random number and find the corresponding individual
    let choice = rng.gen::<f64>();
    let mut selected_index = 0;
    for (i, &cumulative_prob) in cumulative_probabilities.iter().enumerate() {
        if choice <= cumulative_prob {
            selected_index = i;
            break;
        }
    }

    // Correct for possible off-by-one due to cumulative probabilities
    selected_index = selected_index.min(population.len() - 1);

    // Return the selected individual based on calculated rank
    population[indexed_fitnesses[selected_index].0].clone()
}

fn write_solution_to_csv(file_path: PathBuf, solution: &Array2<f64>, fitness: f64, df: &DataFrame) {
    // Calculate bucket assignments from the solution
    let bucket_assignments = solution.map_axis(Axis(1), |row| {
        row.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx as i64 + 1)
            .unwrap() // +1 to make it 1-indexed
    });

    // Create a new DataFrame with bucket assignments
    let mut output_df = df.clone();
    output_df
        .with_column(Series::new("bucket", bucket_assignments.to_vec()))
        .unwrap();

    CsvWriter::new(File::create(file_path).unwrap())
        .has_headers(true)
        .finish(&output_df)
        .expect("Failed to write DataFrame to CSV");
}

fn run_genetic_algorithm(
    df: &DataFrame,
    buckets: usize,
    pop_size: usize,
    k: usize,
    mutation_rate: f64,
    generations: usize,
    fitness_func: FitnessFunction,
    selection_method_func: SelectionMethodFunction,
    num_parents: usize,
    num_top_parents: usize,
    writer: WriterFunction,
) -> Option<Array2<f64>> {
    let max_exponent = (usize::BITS - 1) / 2; // Ensures the result of 2^exp won't overflow

    let n = df.height();
    let mut population = initialize_population_from_solution(df, pop_size, n, buckets);

    let costs_array = df
        .column("cost")
        .unwrap()
        .i64()
        .unwrap()
        .into_no_null_iter()
        .map(|x| x as f64)
        .collect::<Vec<f64>>();
    let costs = Array1::from_vec(costs_array);

    let discounts_array = df
        .column("discount")
        .unwrap()
        .i64()
        .unwrap()
        .into_no_null_iter()
        .map(|x| x as f64 / 100.0)
        .collect::<Vec<f64>>();
    let discounts = Array1::from_vec(discounts_array);

    let mut best_solution = population[0].clone();
    let mut best_fitness = fitness_func(&best_solution, &costs, &discounts);

    println!("Initial best fitness: {}", best_fitness);

    let mut no_improvement_counter = 0;
    for gen in 0..generations {
        let fitnesses: Vec<f64> = population
            .par_iter()
            .map(|x| fitness_func(x, &costs, &discounts))
            .collect();

        let mut fitness_population_pairs: Vec<_> =
            population.iter().zip(fitnesses.iter()).collect();
        fitness_population_pairs.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

        let top_parents: Vec<Array2<f64>> = fitness_population_pairs
            .iter()
            .take(num_top_parents)
            .map(|(x, _)| (*x).clone())
            .collect();

        let mut new_population = top_parents.clone();
        while new_population.len() < pop_size {
            let parents: Vec<Array2<f64>> = (0..num_parents)
                .map(|_| selection_method_func(&population, &fitnesses))
                .collect();

            let mut child = k_point_crossover(&parents, k);
            mutation(&mut child, mutation_rate);
            repair_constraint(&mut child);

            new_population.push(child);
        }

        let new_fitnesses: Vec<f64> = new_population
            .par_iter()
            .map(|x| fitness_func(x, &costs, &discounts))
            .collect();
        let best_fitness_idx = new_fitnesses
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i);

        if let Some(idx) = best_fitness_idx {
            let t_best_solution = &new_population[idx];
            let t_best_fitness = new_fitnesses[idx];

            println!("Generation {} best fitness: {}", gen, t_best_fitness);

            if t_best_fitness > best_fitness {
                best_solution = t_best_solution.clone();
                best_fitness = t_best_fitness;
                no_improvement_counter = 0;

                writer(&best_solution, best_fitness, df);
            } else {
                no_improvement_counter += 1;
            }
        }

        let exponent = (gen as u32 / 7).min(max_exponent);

        if no_improvement_counter >= 2_usize.pow(exponent) {
            println!("Resetting population to previous best solution due to stagnation.");
            new_population = vec![best_solution.clone(); pop_size];
            no_improvement_counter = 0;
        }

        population = new_population;
    }

    Some(best_solution)
}

fn main() {
    let input_file_path = PathBuf::from("./data/input.csv");
    let output_file_path = PathBuf::from("./data/output.csv");

    let config_str = fs::read_to_string("./config.toml").expect("Failed to read config file");
    let config: Config = toml::from_str(&config_str).expect("Failed to parse config");

    let ga_config = config.genetic_algorithm;

    let df = CsvReader::from_path(input_file_path)
        .unwrap()
        .finish()
        .unwrap();

    let buckets = df.column("bucket").unwrap().n_unique().unwrap();

    let fitness_func: FitnessFunction = Arc::new(calculate_objective);
    let selection_method_func: SelectionMethodFunction =
        Arc::new(|population, fitnesses| tournament_selection(population, fitnesses, 3));

    let writer: WriterFunction = Arc::new(move |solution, fitness, df| {
        write_solution_to_csv(output_file_path.clone(), solution, fitness, df)
    });

    run_genetic_algorithm(
        &df,
        buckets, // This might override the config if it's determined dynamically
        ga_config.pop_size,
        ga_config.k,
        ga_config.mutation_rate,
        ga_config.generations,
        fitness_func,
        selection_method_func,
        ga_config.num_parents,
        ga_config.num_top_parents,
        writer,
    );
}
