use std::sync::Arc;

use crate::utils::*;
use ndarray::{s, Array2, Axis};
use ndarray_rand::rand_distr::Normal;
use rand::{
    distributions::Distribution,
    seq::{IteratorRandom, SliceRandom},
    Rng,
};
use rayon::prelude::*;

pub fn k_point_crossover(parents: &[Array2<f64>], k: usize) -> Array2<f64> {
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

pub fn uniform_crossover(parents: &[Array2<f64>]) -> Array2<f64> {
    assert!(
        parents.len() >= 2,
        "There must be at least two parents for crossover."
    );

    let (n_rows, n_cols) = parents[0].dim();
    let mut child = Array2::<f64>::zeros((n_rows, n_cols));
    let mask: Vec<Vec<bool>> = (0..n_rows)
        .map(|_| (0..n_cols).map(|_| rand::random()).collect())
        .collect();

    child
        .axis_iter_mut(Axis(0))
        .enumerate()
        .for_each(|(i, mut row)| {
            for j in 0..n_cols {
                row[j] = if mask[i][j] {
                    parents[0][(i, j)]
                } else {
                    parents[1][(i, j)]
                };
            }
        });

    child
}

pub fn mutation(x: &mut Array2<f64>, mutation_rate: f64) {
    let mut rng = rand::thread_rng();
    let n_cols = x.ncols();

    x.axis_iter_mut(Axis(0)).for_each(|mut row| {
        for j in 0..n_cols {
            if rng.gen::<f64>() >= mutation_rate {
                continue;
            }

            row[j] = 0.0;
            let new_col = rng.gen_range(0..n_cols);
            row[new_col] = 1.0;
        }
    });
}

pub fn bit_flip_mutation(x: &mut Array2<f64>, mutation_rate: f64) {
    let mut rng = rand::thread_rng();
    let (n_rows, n_cols) = x.dim();

    for i in 0..n_rows {
        for j in 0..n_cols {
            if rng.gen::<f64>() < mutation_rate {
                x[[i, j]] = if x[[i, j]] == 1.0 { 0.0 } else { 1.0 };
            }
        }
    }
}

pub fn uniform_mutation(
    x: &mut Array2<f64>,
    mutation_rate: f64,
    lower_bound: f64,
    upper_bound: f64,
) {
    let mut rng = rand::thread_rng();
    let n_cols = x.ncols();

    x.axis_iter_mut(Axis(0)).for_each(|mut row| {
        for j in 0..n_cols {
            if rng.gen::<f64>() < mutation_rate {
                row[j] = rng.gen_range(lower_bound..upper_bound);
            }
        }
    });
}

pub fn gaussian_mutation(x: &mut Array2<f64>, mutation_rate: f64, mean: f64, std_dev: f64) {
    let n_cols = x.ncols();
    let normal_dist = Normal::new(mean, std_dev).unwrap();
    let mut rng = rand::thread_rng();

    x.axis_iter_mut(Axis(0)).for_each(|mut row| {
        for j in 0..n_cols {
            if rng.gen::<f64>() < mutation_rate {
                let noise = normal_dist.sample(&mut rng);
                row[j] += noise;
            }
        }
    });
}

pub fn tournament_selection(
    population: &[Array2<f64>],
    fitnesses: &[f64],
    tournament_size: usize,
) -> Array2<f64> {
    let mut rng = rand::thread_rng();

    let selected_indices: Vec<_> = (0..population.len()).choose_multiple(&mut rng, tournament_size);

    let best_index = *selected_indices
        .iter()
        .max_by(|&&x, &&y| fitnesses[x].partial_cmp(&fitnesses[y]).unwrap())
        .unwrap();

    population[best_index].clone()
}

pub fn roulette_wheel_selection(population: &[Array2<f64>], fitnesses: &[f64]) -> Array2<f64> {
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

pub fn rank_selection(population: &[Array2<f64>], fitnesses: &[f64]) -> Array2<f64> {
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

pub fn run_genetic_algorithm(
    mut population: Population,
    config: &Config,
    fitness_func: FitnessFunction,
    selection_method_func: SelectionMethodFunction,
    mating_func: MatingFunction,
    mutation_func: MutationFunction,
    writer: WriterFunction,
) -> Option<Array2<f64>> {
    let ga_config = &config.genetic_algorithm;

    // Calculate the effective population that needs processing after accounting for top parents
    let effective_pop_size = (ga_config.pop_size - ga_config.num_elites).max(1);

    let num_cpus = num_cpus::get().max(1); // Number of logical CPUs

    // Calculate default chunk size, aiming for at least 4 chunks per CPU for balanced workload
    // and ensure at least 1 individual per chunk to avoid creating empty chunks
    let default_chunk_size = (ga_config.pop_size / num_cpus).max(1);

    let num_chunks = ((effective_pop_size + default_chunk_size - 1) / default_chunk_size).max(1);

    let mut best_solution = population[0].clone();
    let mut best_fitness = fitness_func(&best_solution, ga_config);

    let mut no_improvement_counter = 0;
    let mut reset_counter = 0;

    let mate = |chunk_ix: usize, population: &Population, fitnesses: &[f64]| {
        let start_ix = chunk_ix * default_chunk_size;
        let end_ix = std::cmp::min(start_ix + default_chunk_size, effective_pop_size);

        let mut local_population = Vec::with_capacity(end_ix - start_ix);

        for _ in start_ix..end_ix {
            let parents: Vec<_> = (0..ga_config.num_parents)
                .map(|_| selection_method_func(population, fitnesses, ga_config))
                .collect();

            let mut child = mating_func(&parents, ga_config);
            mutation_func(&mut child, ga_config);

            local_population.push(child);
        }

        local_population
    };

    for gen in 0..ga_config.generations {
        let fitnesses = Arc::new(
            population
                .par_iter()
                .map(|x| fitness_func(x, ga_config))
                .collect::<Vec<_>>(),
        );

        let mut fitness_ixs = fitnesses.iter().enumerate().collect::<Vec<_>>();
        fitness_ixs.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

        let best_ix = fitness_ixs[0].0;

        let t_best_fitness = fitnesses[best_ix];
        println!("Generation {} best fitness: {}", gen, t_best_fitness);

        if t_best_fitness > best_fitness {
            best_solution = population[best_ix].clone();
            best_fitness = t_best_fitness;

            println!("**New best fitness: {}", best_fitness);

            no_improvement_counter = 0;
            reset_counter = 0;

            writer(&best_solution, best_fitness, config);
        } else if gen > 0 {
            no_improvement_counter += 1;
        }

        if no_improvement_counter >= 2_usize.pow(reset_counter.min(MAX_EXPONENT)) {
            println!("Resetting population due to stagnation");

            population = Arc::new(vec![best_solution.clone(); ga_config.pop_size]);

            no_improvement_counter = 0;
            reset_counter += 1;
        }

        population = Arc::new(
            fitness_ixs
                .iter()
                .take(ga_config.num_elites)
                .map(|(i, _)| population[*i].clone())
                .chain(
                    (0..num_chunks)
                        .into_par_iter()
                        .map(|chunk_index| mate(chunk_index, &population, &fitnesses))
                        .flatten()
                        .collect::<Vec<_>>(),
                )
                .collect(),
        );
    }

    Some(best_solution)
}
