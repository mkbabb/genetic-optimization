use crate::utils::*;

use ndarray::{s, Array2, Axis};
use ndarray_rand::rand_distr::Normal;
use rand::{
    distributions::Distribution,
    seq::{IteratorRandom, SliceRandom},
    Rng,
};
use rayon::prelude::*;
use std::sync::Arc;

pub fn k_point_crossover(parents: &[Chromosome], k: usize) -> Chromosome {
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

pub fn uniform_crossover(parents: &[Chromosome]) -> Chromosome {
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

pub fn standard_mutation(x: &mut Chromosome, mutation_rate: f64) {
    let mut rng = rand::thread_rng();
    let n_cols = x.ncols();

    x.axis_iter_mut(Axis(0)).for_each(|mut row| {
        if rng.gen::<f64>() >= mutation_rate {
            return;
        }

        let new_col = rng.gen_range(0..n_cols);
        row.fill(0.0);
        row[new_col] = 1.0;
    });
}

pub fn gaussian_mutation(x: &mut Chromosome, mutation_rate: f64, mean: f64, std_dev: f64) {
    let n_cols = x.ncols();
    let normal_dist = Normal::new(mean, std_dev).unwrap();
    let mut rng = rand::thread_rng();

    x.axis_iter_mut(Axis(0)).for_each(|mut row| {
        if rng.gen::<f64>() >= mutation_rate {
            return;
        }

        for i in 0..n_cols {
            if normal_dist.sample(&mut rng).round() >= 1.0 {
                row.fill(0.0);
                row[i] = 1.0;
                break;
            }
        }
    });
}

pub fn tournament_selection(
    population: &[Chromosome],
    fitnesses: &[f64],
    tournament_size: usize,
) -> Chromosome {
    let mut rng = rand::thread_rng();

    let selected_indices: Vec<_> = (0..population.len()).choose_multiple(&mut rng, tournament_size);

    let best_index = *selected_indices
        .iter()
        .max_by(|&&x, &&y| fitnesses[x].partial_cmp(&fitnesses[y]).unwrap())
        .unwrap();

    population[best_index].clone()
}

pub fn roulette_wheel_selection(population: &[Chromosome], fitnesses: &[f64]) -> Chromosome {
    let mut rng = rand::thread_rng();

    let total_fitness: f64 = fitnesses.iter().sum();
    let probabilities: Vec<f64> = fitnesses.iter().map(|&f| f / total_fitness).collect();
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

pub fn rank_selection(population: &[Chromosome], fitnesses: &[f64]) -> Chromosome {
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

pub fn cull_best_mutants(
    population: &[Chromosome],
    best_solution: &Chromosome,
    num_to_cull: usize,
) -> Vec<Chromosome> {
    let mut new_population = Vec::with_capacity(population.len());

    new_population.extend((0..num_to_cull).map(|_| {
        let mut clone = best_solution.clone();
        standard_mutation(&mut clone, 1.0);
        clone
    }));

    new_population.extend(
        (0..(population.len() - num_to_cull))
            .map(|_| population.choose(&mut rand::thread_rng()).unwrap().clone()),
    );

    new_population
}

pub fn cull_best_mutants_randomized(
    population: &[Chromosome],
    best_solution: &Chromosome,
    num_to_cull: usize,
) -> Vec<Chromosome> {
    let mut new_population = Vec::with_capacity(population.len());

    // Half should be best mutants, half should be random
    new_population.extend((0..num_to_cull).map(|i| {
        let mutation_rate = if i < num_to_cull / 2 { 1.0 } else { 100.0 };

        let mut clone = best_solution.clone();
        standard_mutation(&mut clone, mutation_rate);
        clone
    }));

    new_population.extend(
        (0..(population.len() - num_to_cull))
            .map(|_| population.choose(&mut rand::thread_rng()).unwrap().clone()),
    );

    new_population
}

pub fn cull_randomized(
    population: &[Chromosome],
    _: &Chromosome,
    num_to_cull: usize,
) -> Vec<Chromosome> {
    let mut new_population = Vec::with_capacity(population.len());

    new_population.extend((0..num_to_cull).map(|i| {
        let mut clone = population[i].clone();
        standard_mutation(&mut clone, 100.0);
        clone
    }));

    new_population.extend(
        (0..(population.len() - num_to_cull))
            .map(|_| population.choose(&mut rand::thread_rng()).unwrap().clone()),
    );

    new_population
}

pub fn cull_best(
    population: &[Chromosome],
    best_solution: &Chromosome,
    num_to_cull: usize,
) -> Vec<Chromosome> {
    let mut new_population = Vec::with_capacity(population.len());

    new_population.extend((0..num_to_cull).map(|_| best_solution.clone()));

    new_population.extend(
        (0..(population.len() - num_to_cull))
            .map(|_| population.choose(&mut rand::thread_rng()).unwrap().clone()),
    );

    new_population
}

pub fn init_ga_funcs(
    fitness_func: FitnessFunction,
    writer_func: WriterFunction,
    _: &Config,
) -> GeneticAlgorithmFunctions {
    let selection_method_func: SelectionMethodFunction = Arc::new(
        move |population, fitnesses, config| match config.selection_method {
            SelectionMethod::Rank => rank_selection(population, fitnesses),
            SelectionMethod::Roulette => roulette_wheel_selection(population, fitnesses),
            SelectionMethod::Tournament => {
                tournament_selection(population, fitnesses, config.tournament_size)
            }
            SelectionMethod::StochasticUniversalSampling => {
                unimplemented!()
            }
        },
    );

    let mating_func: MatingFunction = Arc::new(|parents, config| match config.mating_method {
        MatingMethod::KPointCrossover => k_point_crossover(parents, config.k),
        MatingMethod::UniformCrossover => uniform_crossover(parents),
    });

    let mutation_func: MutationFunction = Arc::new(|x, config| match config.mutation_method {
        MutationMethod::Gaussian => gaussian_mutation(
            x,
            config.mutation_rate,
            config.mutation_mean,
            config.mutation_std_dev,
        ),
        MutationMethod::Standard => standard_mutation(x, config.mutation_rate),
        _ => unimplemented!(),
    });

    let culling_func = Arc::new(
        |population: &_, best_solution: &_, num_to_cull: _, config: &GeneticAlgorithmConfig| {
            match config.culling_method {
                CullingMethod::BestMutants => {
                    cull_best_mutants(population, best_solution, num_to_cull)
                }
                CullingMethod::BestMutantsRandom => {
                    cull_best_mutants_randomized(population, best_solution, num_to_cull)
                }
                CullingMethod::Best => cull_best(population, best_solution, num_to_cull),
                CullingMethod::Random => cull_randomized(population, best_solution, num_to_cull),
            }
        },
    );

    GeneticAlgorithmFunctions {
        fitness: fitness_func,
        selection_method: selection_method_func,
        mating: mating_func,
        mutation: mutation_func,
        culling: culling_func,
        writer: writer_func,
    }
}

pub fn run(
    mut population: Arc<Population>,
    config: &Config,
    funcs: &GeneticAlgorithmFunctions,
) -> Option<Chromosome> {
    let ga_config = &config.genetic_algorithm;

    let GeneticAlgorithmFunctions {
        fitness: fitness_func,
        selection_method: selection_method_func,
        mating: mating_func,
        mutation: mutation_func,
        culling: culling_func,
        writer: writer_func,
    } = funcs;

    let effective_pop_size = (ga_config.pop_size - ga_config.num_elites).max(1);
    let num_cpus = ga_config.num_cpus.unwrap_or_else(num_cpus::get);
    let default_chunk_size = (ga_config.pop_size / num_cpus).max(1);
    let num_chunks = ((effective_pop_size + default_chunk_size - 1) / default_chunk_size).max(1);

    // Initialize the best solution and its fitness
    let mut best_solution = population[0].clone();
    let mut best_fitness = fitness_func(&best_solution, ga_config);

    // Initialize the culling percentage and counters
    let mut culling_percent = ga_config.min_culling_percent;
    let mut no_improvement_counter = 0_usize;
    let mut reset_counter = 0_usize;

    let was_no_improvement = |no_improvement_counter: usize, reset_counter: usize| {
        let backoff = 2_usize.pow((reset_counter.min(MAX_EXPONENT) + 1) as u32);

        backoff.min(ga_config.max_no_improvement_generations) <= no_improvement_counter
    };

    let par_mate = |chunk_ix: usize, population: &[Chromosome], fitnesses: &[f64]| {
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

    let get_elites = |population: &[Chromosome], fitnesses: &[f64]| {
        let mut fitness_ixs = fitnesses.iter().enumerate().collect::<Vec<_>>();
        fitness_ixs.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

        fitness_ixs
            .iter()
            .take(ga_config.num_elites)
            .map(|(i, _)| population[*i].clone())
            .collect::<Vec<_>>()
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

        log::debug!("Generation {} best fitness: {:.4}", gen, t_best_fitness);

        if t_best_fitness > best_fitness {
            best_solution = population[best_ix].clone();
            best_fitness = t_best_fitness;

            log::info!("**New best fitness: {:.4}", best_fitness);

            culling_percent = ga_config.min_culling_percent;
            no_improvement_counter = 0;
            reset_counter = 0;

            writer_func(&best_solution, best_fitness, config)
        } else if gen > 0 {
            no_improvement_counter += 1;
        }

        if was_no_improvement(no_improvement_counter, reset_counter) {
            culling_percent = match ga_config.culling_direction {
                CullingDirection::Forward => (culling_percent
                    * ga_config.culling_percent_increment)
                    .min(ga_config.max_culling_percent),
                CullingDirection::Reverse => (culling_percent
                    / ga_config.culling_percent_increment)
                    .max(ga_config.min_culling_percent),
            };

            let num_to_cull = (population.len() as f64 * culling_percent).ceil() as usize;

            log::warn!(
                "Resetting {}% of the population ({} of {}) due to stagnation; reset counter: {}",
                (culling_percent * 100.0).round(),
                num_to_cull,
                population.len(),
                reset_counter,
            );

            no_improvement_counter = 0;
            reset_counter += 1;

            let culled_population =
                culling_func(&population, &best_solution, num_to_cull, ga_config)
                    .into_iter()
                    .take(effective_pop_size);

            population = Arc::new(
                get_elites(&population, &fitnesses)
                    .into_iter()
                    .chain(culled_population)
                    .collect(),
            );

            assert_eq!(population.len(), ga_config.pop_size);
        }

        population = Arc::new(
            get_elites(&population, &fitnesses)
                .into_iter()
                .chain(
                    (0..num_chunks)
                        .into_par_iter()
                        .map(|chunk_index| par_mate(chunk_index, &population, &fitnesses))
                        .flatten()
                        .collect::<Vec<_>>(),
                )
                .collect(),
        );
    }

    Some(best_solution)
}
