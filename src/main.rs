pub mod genetic;
pub mod utils;

use crate::genetic::{
    gaussian_mutation, k_point_crossover, mutation, rank_selection, roulette_wheel_selection,
    run_genetic_algorithm, tournament_selection,
};
use crate::utils::Config;
use ndarray::{Array1, Array2, Axis};
use polars::frame::DataFrame;
use polars::io::csv::{CsvReader, CsvWriter};
use polars::io::{SerReader, SerWriter};
use polars::prelude::*;
use polars::series::Series;
use rand::seq::SliceRandom;
use std::cmp;
use std::fs;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use utils::{
    download_sheet_to_csv, upload_csv_to_sheet, FitnessFunction, MatingFunction, MutationFunction,
    SelectionMethodFunction, WriterFunction,
};

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

            for (i, bucket_value) in bucket_series.into_iter().enumerate() {
                if let Some(bucket) = bucket_value {
                    let bucket_index = (bucket - 1) as usize; // assuming 1-indexed buckets
                    if bucket_index < buckets {
                        x[(i, bucket_index)] = 1.0;
                    }
                }
            }
            x
        })
        .collect()
}

fn calculate_fitness(x: &Array2<f64>, costs: &Array1<f64>, discounts: &Array1<f64>) -> f64 {
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

pub fn repair_constraint(x: &mut Array2<f64>) {
    let mut rng = rand::thread_rng();

    x.axis_iter_mut(Axis(0))
        .filter(|row| row.sum() > 1.0)
        .for_each(|mut row| {
            let mut ixs = row
                .indexed_iter()
                .filter(|&(_, &v)| v > 0.0)
                .map(|(idx, _)| idx)
                .skip(1)
                .collect::<Vec<_>>();

            ixs.shuffle(&mut rng);

            for &ix in ixs.iter() {
                row[ix] = 0.0;
            }
        });
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
    let input_file_path = Path::new("./data/input.csv");
    let output_file_path = Path::new("./data/output.csv");

    let config_str = fs::read_to_string("./config.toml").expect("Failed to read config file");
    let config: Config = toml::from_str(&config_str).expect("Failed to parse config");

    dbg!(config.genetic_algorithm.clone());

    download_sheet_to_csv(input_file_path, &config);

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

    let population =
        initialize_population_from_solution(&df, buckets, config.genetic_algorithm.pop_size);

    let fitness_func: FitnessFunction =
        Arc::new(move |solution, _| calculate_fitness(solution, &costs, &discounts));

    let selection_method_func: SelectionMethodFunction = Arc::new(|population, fitnesses, _| {
        let n = 3;
        tournament_selection(population, fitnesses, n)
    });

    let mating_func: MatingFunction =
        Arc::new(|parents, config| k_point_crossover(parents, config.k));

    let mutation_func: MutationFunction = Arc::new(|x, config| {
        mutation(x, config.mutation_rate);
        repair_constraint(x);
    });

    let config_clone = config.clone();

    let writer_func: WriterFunction = Arc::new(move |solution, fitness, _| {
        write_solution_to_csv(output_file_path, solution, fitness, &df);
        upload_csv_to_sheet(output_file_path, &config);
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
