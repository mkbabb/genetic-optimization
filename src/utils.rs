use ndarray::Array2;
use serde::Deserialize;
use std::path::Path;
use std::process::Command;
use std::sync::Arc;

#[derive(Deserialize, Debug, Clone)]
pub struct Config {
    pub google: GoogleConfig,
    pub genetic_algorithm: GeneticAlgorithmConfig,
}

#[derive(Deserialize, Debug, Clone)]
pub struct GoogleConfig {
    pub optimization_sheet_id: String,
    pub input_range_name: String,
    pub output_range_name: String,
}

#[derive(Deserialize, Debug, Clone)]
pub struct GeneticAlgorithmConfig {
    pub pop_size: usize,
    pub k: usize,
    pub mutation_rate: f64,
    pub generations: usize,
    pub num_parents: usize,
    pub num_top_parents: usize,
}

pub type Population = Vec<Array2<f64>>;

pub type FitnessFunction = Arc<dyn Fn(&Array2<f64>, &GeneticAlgorithmConfig) -> f64 + Send + Sync>;
pub type SelectionMethodFunction =
    Arc<dyn Fn(&[Array2<f64>], &[f64], &GeneticAlgorithmConfig) -> Array2<f64> + Send + Sync>;
pub type MatingFunction =
    Arc<dyn Fn(&[Array2<f64>], &GeneticAlgorithmConfig) -> Array2<f64> + Send + Sync>;
pub type MutationFunction = Arc<dyn Fn(&mut Array2<f64>, &GeneticAlgorithmConfig) + Send + Sync>;

pub type WriterFunction = Arc<dyn Fn(&Array2<f64>, f64, &Config) + Send + Sync>;

pub const MAX_EXPONENT: u32 = (usize::BITS - 1) / 2;

// Helper function to run a Python script with Poetry
fn run_poetry_command(script_path: &Path, args: &[&str]) {
    let output = Command::new("poetry")
        .arg("run")
        .arg("python3")
        .arg("-m")
        .arg(script_path)
        .args(args)
        .output()
        .expect("Failed to execute command");

    println!("{}", String::from_utf8_lossy(&output.stdout));

    if !output.stderr.is_empty() {
        println!("{}", String::from_utf8_lossy(&output.stderr));
    }
}

pub fn download_sheet_to_csv(input_file_path: &Path, config: &Config) {
    let script_path = Path::new("scripts.download_sheet_to_csv");
    let args = [
        input_file_path.to_str().unwrap(),
        &config.google.optimization_sheet_id,
        &config.google.input_range_name,
    ];
    run_poetry_command(script_path, &args);
}

pub fn upload_csv_to_sheet(output_file_path: &Path, config: &Config) {
    let script_path = Path::new("scripts.upload_csv_to_sheet");
    let args = [
        output_file_path.to_str().unwrap(),
        &config.google.optimization_sheet_id,
        &config.google.output_range_name,
    ];

    run_poetry_command(script_path, &args);
}
