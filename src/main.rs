pub mod genetic;
pub mod utils;

use crate::genetic::{init_ga_funcs, run};
use crate::utils::{Config, init_logger, round};
use clap::{Parser, arg, command};
use genetic::{gaussian_mutation, standard_mutation};
use ndarray::{Array1, Array2, Axis};
use polars::frame::DataFrame;
use polars::io::csv::{CsvReader, CsvWriter};
use polars::io::{SerReader, SerWriter};
use polars::prelude::*;
use polars::series::Series;
use std::cmp;
use std::collections::HashSet;
use std::fs;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tempfile::NamedTempFile;
use utils::{
    FitnessFunction, GeneticAlgorithmConfig, MutationFunction, MutationMethod, WriterFunction,
    download_sheet_to_csv, upload_csv_to_sheet,
};

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = "config.toml")]
    config: PathBuf,
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
    mutation_func: MutationFunction,
    config: &GeneticAlgorithmConfig,
) -> Vec<Array2<f64>> {
    (0..pop_size)
        .map(|_| {
            let mut x = Array2::<f64>::zeros((df.height(), buckets));
            x.column_mut(0).fill(1.0);
            mutation_func(&mut x, config);
            x
        })
        .collect()
}

fn calculate_fitness(x: &Array2<f64>, costs: &Array1<f64>, discounts: &Array1<f64>) -> f64 {
    let xt = x.t();

    let total_costs_per_bucket = xt.dot(costs);

    let total_discounts_per_bucket = xt.dot(discounts);

    let items_per_bucket: Array1<f64> = xt
        .sum_axis(Axis(1))
        .mapv(|x| if x == 0.0 { 1.0 } else { x });

    let avg_discounts_per_bucket =
        (total_discounts_per_bucket / items_per_bucket).mapv(|x| round(x, 2, 3));

    let discount_costs_per_bucket = total_costs_per_bucket * avg_discounts_per_bucket;

    let discount_cost_sum = discount_costs_per_bucket.sum();

    discount_cost_sum
}

fn calculate_fitness_frn_diversity(
    x: &Array2<f64>,
    bws: &Array1<f64>,
    costs: &Array1<f64>,
    discounts: &Array1<f64>,
) -> f64 {
    let calculate_frn_diversity_weights = || -> Array1<f64> {
        let bucket_assignments: Vec<usize> = x
            .map_axis(Axis(1), |row| {
                row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(index, _)| index)
                    .unwrap()
            })
            .to_vec();

        let mut bucket_frn_counts = vec![HashSet::new(); x.ncols()];

        for (index, &bucket) in bucket_assignments.iter().enumerate() {
            let frn = format!("{}-{}", bws[index], costs[index]);
            bucket_frn_counts[bucket].insert(frn);
        }

        let diversity_scores = bucket_frn_counts
            .iter()
            .map(|set| set.len() as f64)
            .collect::<Vec<_>>();
        let max_diversity = diversity_scores
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(&1.0);

        // Normalize diversity scores to a range that subtly adjusts the discount costs
        diversity_scores
            .iter()
            .map(|&score| 1.0 - (score / max_diversity))
            .collect::<Array1<f64>>()
            * 0.5
            + 1.0
    };

    let frn_diversity_weights = calculate_frn_diversity_weights();

    let calculate_discount_costs = || -> Array1<f64> {
        let xt = x.t();

        let total_costs_per_bucket = xt.dot(costs);

        let total_discounts_per_bucket = xt.dot(discounts);

        let items_per_bucket: Array1<f64> = xt
            .sum_axis(Axis(1))
            .mapv(|x| if x == 0.0 { 1.0 } else { x });

        let avg_discounts_per_bucket =
            total_discounts_per_bucket / (items_per_bucket * frn_diversity_weights);

        total_costs_per_bucket * avg_discounts_per_bucket.mapv(|x| round(x, 2, 3))
    };

    calculate_discount_costs().sum()
}

fn write_solution_to_csv(file_path: &Path, x: &Array2<f64>, fitness: f64, df: &DataFrame) {
    // Calculate bucket assignments from the x
    let bucket_assignments = x.map_axis(Axis(1), |row| {
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

    let args: Args = Args::parse();

    init_logger(None, None);

    let config_str = fs::read_to_string(args.config).expect("Failed to read config file");
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

    let bws_array = df
        .column("bw")
        .expect("BW column not found")
        .i64()
        .expect("BW values must be i64")
        .into_no_null_iter()
        .map(|x| x as f64)
        .collect::<Vec<f64>>();
    let bws = Array1::from_vec(bws_array);

    let discounts_array = df
        .column("discount")
        .expect("Discount column not found")
        .i64()
        .expect("Discount values must be i64")
        .into_no_null_iter()
        .map(|x| x as f64 / 100.0)
        .collect::<Vec<f64>>();
    let discounts = Array1::from_vec(discounts_array);

    // let buckets = df
    //     .column("bucket")
    //     .expect("Bucket column not found")
    //     .n_unique()
    //     .unwrap();

    // let population = Arc::new(initialize_population_from_solution(
    //     &df,
    //     buckets,
    //     config.genetic_algorithm.pop_size,
    // ));

    let buckets = 4;

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

    let population = Arc::new(initialize_population_random(
        &df,
        buckets,
        config.genetic_algorithm.pop_size,
        mutation_func,
        &config.genetic_algorithm,
    ));

    let fitness_func: FitnessFunction = Arc::new(move |x, _| {
        // let max_bucket_size = config
        //     .genetic_algorithm
        //     .max_bucket_size
        //     .unwrap_or(usize::MAX) as f64;
        // let balance_weight = config.genetic_algorithm.balance_weight.unwrap_or(1.0);
        // let size_penalty_weight = config.genetic_algorithm.size_penalty_weight.unwrap_or(2.0);

        // calculate_fitness_balanced_buckets(
        //     x,
        //     &costs,
        //     &discounts,
        //     max_bucket_size,
        //     balance_weight,
        //     size_penalty_weight,
        // )

        // calculate_fitness_frn_diversity(x, &bws, &costs, &discounts)

        calculate_fitness(x, &costs, &discounts)
    });

    let writer_func: WriterFunction = Arc::new(move |x, fitness, config| {
        write_solution_to_csv(&output_file_path, x, fitness, &df);
        upload_csv_to_sheet(&output_file_path, config);
    });

    let funcs = init_ga_funcs(fitness_func, writer_func, &config);

    run(population, &config, &funcs);
}
