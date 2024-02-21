pub mod genetic;
pub mod utils;

use crate::genetic::{
    gaussian_mutation, k_point_crossover, mutation, rank_selection, roulette_wheel_selection,
    run_genetic_algorithm, tournament_selection, uniform_crossover,
};
use crate::utils::{round, Config, MatingMethod, MutationMethod, SelectionMethod};

use env_logger::Builder;
use ndarray::{Array1, Array2, Axis};
use polars::frame::DataFrame;
use polars::io::csv::{CsvReader, CsvWriter};
use polars::io::{SerReader, SerWriter};
use polars::prelude::*;
use polars::series::Series;
use std::cmp;
use std::fs;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tempfile::NamedTempFile;
use utils::{
    download_sheet_to_csv, upload_csv_to_sheet, FitnessFunction, GeneticAlgorithmConfig,
    MatingFunction, MutationFunction, SelectionMethodFunction, WriterFunction,
};

use clap::Parser;

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Name of the person to greet
    #[arg(short, long)]
    name: String,

    /// Number of times to greet
    #[arg(short, long, default_value_t = 1)]
    count: u8,
}

pub fn initialize_population_from_solution(
    df: &DataFrame,
    buckets: usize,
    pop_size: usize,
) -> Vec<Array2<f64>> {
    // Determine the maximum bucket value from the DataFrame
    let bucket_series = df.column("bucket").unwrap().i64().unwrap();

    let max_bucket_value = bucket_series.into_iter().flatten().max().unwrap_or(0) as usize;

    // Adjust the bucket count to be the upper bound of the largest bucket
    let buckets = cmp::max(buckets, max_bucket_value);

    // Initialize the population using the adjusted bucket count
    (0..pop_size)
        .map(|_| {
            let mut x = Array2::<f64>::zeros((df.height(), buckets));

            for (i, bucket) in bucket_series.into_no_null_iter().enumerate() {
                let bucket = (bucket - 1) as usize; // assuming 1-indexed buckets
                x[(i, bucket)] = 1.0;
            }
            x
        })
        .collect()
}

fn initialize_population_random(
    df: &DataFrame,
    buckets: usize,
    pop_size: usize,
    mutation_fn: MutationFunction,
    config: &GeneticAlgorithmConfig,
) -> Vec<Array2<f64>> {
    (0..pop_size)
        .map(|_| {
            let mut x = Array2::<f64>::zeros((df.height(), buckets));
            x.column_mut(0).fill(1.0);
            mutation_fn(&mut x, config);
            x
        })
        .collect()
}

fn calculate_fitness(x: &Array2<f64>, costs: &Array1<f64>, discounts: &Array1<f64>) -> f64 {
    let xt = x.t();

    let total_costs_per_bucket = xt.dot(costs);
    // log::debug!("total_costs_per_bucket: {:?}", total_costs_per_bucket);

    let total_discounts_per_bucket = xt.dot(discounts);
    // log::debug!(
    //     "total_discounts_per_bucket: {:?}",
    //     total_discounts_per_bucket
    // );

    let items_per_bucket: Array1<f64> = xt
        .sum_axis(Axis(1))
        .mapv(|x| if x == 0.0 { 1.0 } else { x });
    // log::debug!("items_per_bucket: {:?}", items_per_bucket);

    let avg_discounts_per_bucket =
        (total_discounts_per_bucket / items_per_bucket).mapv(|x| round(x, 2, 3));
    // log::debug!("avg_discounts_per_bucket: {:?}", avg_discounts_per_bucket);

    let discount_costs_per_bucket = total_costs_per_bucket * avg_discounts_per_bucket;
    // log::debug!("discount_costs_per_bucket: {:?}", discount_costs_per_bucket);

    let discount_cost_sum = discount_costs_per_bucket.sum();
    // log::debug!("discount_cost_sum: {:?}", discount_cost_sum);
    discount_cost_sum
}

fn write_solution_to_csv(file_path: &Path, solution: &Array2<f64>, _: f64, df: &DataFrame) {
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
        .finish(&mut output_df)
        .expect("Failed to write DataFrame to CSV");
}

fn main() {
    let input_file_path = NamedTempFile::new().unwrap().into_temp_path().to_path_buf();
    let output_file_path = NamedTempFile::new().unwrap().into_temp_path().to_path_buf();

    Builder::new()
        .format(|buf, record| writeln!(buf, "{}: {}", record.level(), record.args()))
        .filter(None, log::LevelFilter::Debug) // Set the global log level filter
        .target(env_logger::Target::Stdout) // Set target to stdout
        .init();

    let config_file_path: PathBuf =
        arg!(-c --config <CONFIG> "Provides an input config file to the program").into();

    let config_str = fs::read_to_string(config_file_path).expect("Failed to read config file");
    let config: Config = toml::from_str(&config_str).expect("Failed to parse config");

    log::info!("{:#?}", config);

    download_sheet_to_csv(&input_file_path, &config);

    let df = CsvReader::from_path(input_file_path)
        .expect("Failed to read CSV file. Make sure the file exists and the path is correct.")
        .finish()
        .unwrap();

    let costs_array = df
        .column("cost")
        .expect("Cost column not found")
        .i64()
        .expect("Cost values must be i64")
        .into_no_null_iter()
        .map(|x| x as f64)
        .collect::<Vec<f64>>();
    let costs = Array1::from_vec(costs_array);

    let discounts_array = df
        .column("discount")
        .expect("Discount column not found")
        .i64()
        .expect("Discount values must be i64")
        .into_no_null_iter()
        .map(|x| x as f64 / 100.0)
        .collect::<Vec<f64>>();
    let discounts = Array1::from_vec(discounts_array);

    let buckets = df
        .column("bucket")
        .expect("Bucket column not found")
        .n_unique()
        .unwrap();

    let population = Arc::new(initialize_population_from_solution(
        &df,
        buckets,
        config.genetic_algorithm.pop_size,
    ));

    let fitness_func: FitnessFunction =
        Arc::new(move |solution, _| calculate_fitness(solution, &costs, &discounts));

    let selection_method = config.genetic_algorithm.selection_method.clone();
    let selection_method_func: SelectionMethodFunction =
        Arc::new(move |population, fitnesses, _| match selection_method {
            SelectionMethod::Rank => rank_selection(population, fitnesses),
            SelectionMethod::Roulette => roulette_wheel_selection(population, fitnesses),
            SelectionMethod::Tournament => tournament_selection(
                population,
                fitnesses,
                config.genetic_algorithm.tournament_size,
            ),
            SelectionMethod::StochasticUniversalSampling => {
                unimplemented!()
            }
        });

    let mating_method = config.genetic_algorithm.mating_method.clone();
    let mating_func: MatingFunction = Arc::new(move |parents, config| match mating_method {
        MatingMethod::KPointCrossover => k_point_crossover(parents, config.k),
        MatingMethod::UniformCrossover => uniform_crossover(parents),
    });

    let mutation_method = config.genetic_algorithm.mutation_method.clone();
    let mutation_func: MutationFunction = Arc::new(move |x, config| {
        match mutation_method {
            MutationMethod::Gaussian => gaussian_mutation(
                x,
                config.mutation_rate,
                config.mutation_mean,
                config.mutation_std_dev,
            ),
            // MutationMethod::Uniform => uniform_mutation(
            //     x,
            //     config.mutation_rate,
            //     config.mutation_lower_bound,
            //     config.mutation_upper_bound,
            // ),
            MutationMethod::Standard => mutation(x, config.mutation_rate),
            // MutationMethod::BitFlip => bit_flip_mutation(x, config.mutation_rate),
            _ => unimplemented!(),
        }
    });

    let config_clone = config.clone();

    let writer_func: WriterFunction = Arc::new(move |solution, fitness, _| {
        write_solution_to_csv(&output_file_path, solution, fitness, &df);
        upload_csv_to_sheet(&output_file_path, &config);
    });

    run_genetic_algorithm(
        population,
        &config_clone,
        fitness_func,
        selection_method_func,
        mating_func,
        mutation_func,
        writer_func,
    );
}
