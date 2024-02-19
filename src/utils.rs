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

#[derive(Deserialize, Debug, Clone, Copy)]
pub enum CullingDirection {
    #[serde(rename = "forward")]
    Forward,
    #[serde(rename = "reverse")]
    Reverse,
}

#[derive(Deserialize, Debug, Clone)]
pub enum SelectionMethod {
    #[serde(rename = "rank")]
    Rank,
    #[serde(rename = "tournament")]
    Tournament,
    #[serde(rename = "roulette")]
    Roulette,
    #[serde(rename = "stochastic_universal_sampling")]
    StochasticUniversalSampling,
}

#[derive(Deserialize, Debug, Clone)]
pub enum MatingMethod {
    #[serde(rename = "k_point_crossover")]
    KPointCrossover,
    #[serde(rename = "uniform_crossover")]
    UniformCrossover,
}

#[derive(Deserialize, Debug, Clone)]
pub enum MutationMethod {
    #[serde(rename = "gaussian")]
    Gaussian,
    #[serde(rename = "uniform")]
    Uniform,
    #[serde(rename = "standard")]
    Standard,
    #[serde(rename = "bit_flip")]
    BitFlip,
}

#[derive(Deserialize, Debug, Clone)]
pub struct GeneticAlgorithmConfig {
    pub generations: usize,
    pub max_no_improvement_generations: usize,
    pub pop_size: usize,

    pub mutation_rate: f64,
    
    pub mutation_mean: f64,
    pub mutation_std_dev: f64,
    
    pub mutation_lower_bound: f64,
    pub mutation_upper_bound: f64,
    
    pub mutation_method: MutationMethod,
    
    pub num_parents: usize,

    pub num_elites: usize,

    pub min_culling_percent: f64,
    pub max_culling_percent: f64,
    pub culling_percent_increment: f64,
    pub culling_direction: CullingDirection,

    pub selection_method: SelectionMethod,
    pub tournament_size: usize,

    pub mating_method: MatingMethod,
    pub k: usize,

}

pub type Population = Arc<Vec<Array2<f64>>>;

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

pub fn round(x: f64, num_digits: i32, num_9s: usize) -> f64 {
    let multiplier = 10_f64.powi(num_digits);
    let threshold = 0.5 - 0.1_f64.powi(num_9s as i32);
    let scaled = x * multiplier;
    let fraction = scaled.fract();
    let integer = scaled.trunc();

    let rounded = if fraction >= threshold && fraction < 0.5 {
        integer + 0.5
    } else if fraction > 0.5 && fraction <= (0.5 + 0.1_f64.powi(num_9s as i32)) {
        integer + 1.0
    } else {
        scaled
    };

    rounded.round() / multiplier
}
