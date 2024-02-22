use chrono::Local;
use colored::*;
use env_logger::Builder;
use log::{Level, LevelFilter, Record};
use ndarray::Array2;
use serde::Deserialize;
use std::io::Write;
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
    pub sheet_id: String,
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
pub enum CullingMethod {
    #[serde(rename = "best_mutants")]
    BestMutants,
    #[serde(rename = "best")]
    Best,
    #[serde(rename = "random")]
    Random,
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

    pub culling_method: CullingMethod,

    pub num_cpus: Option<usize>,
}

pub type Chromosome = Array2<f64>;

pub type Population = Vec<Chromosome>;

pub type FitnessFunction = Arc<dyn Fn(&Chromosome, &GeneticAlgorithmConfig) -> f64 + Send + Sync>;

pub type SelectionMethodFunction =
    Arc<dyn Fn(&[Chromosome], &[f64], &GeneticAlgorithmConfig) -> Chromosome + Send + Sync>;

pub type MatingFunction =
    Arc<dyn Fn(&[Chromosome], &GeneticAlgorithmConfig) -> Chromosome + Send + Sync>;

pub type MutationFunction = Arc<dyn Fn(&mut Chromosome, &GeneticAlgorithmConfig) + Send + Sync>;

pub type CullingFunction = Arc<
    dyn Fn(&[Chromosome], &Chromosome, f64, &GeneticAlgorithmConfig) -> Population + Send + Sync,
>;

pub type WriterFunction = Arc<dyn Fn(&Array2<f64>, f64, &Config) + Send + Sync>;

pub struct GeneticAlgorithmFunctions {
    pub fitness: FitnessFunction,
    pub selection_method: SelectionMethodFunction,
    pub mating: MatingFunction,
    pub mutation: MutationFunction,
    pub culling: CullingFunction,
    pub writer: WriterFunction,
}

pub const MAX_EXPONENT: usize = (usize::BITS as usize - 1) / 2 - 1;

pub fn init_logger(log_level: Option<LevelFilter>, target: Option<env_logger::Target>) {
    Builder::new()
        .format(|buf, record: &Record| {
            let target = record.target();
            let level = record.level();
            let message = record.args();
            let thread = std::thread::current();
            let thread_name = thread.name().unwrap_or("unknown");
            let time = Local::now().format("%Y-%m-%d %H:%M:%S");

            let level_colored = match level {
                Level::Error => level.to_string().red(),
                Level::Warn => level.to_string().yellow(),
                Level::Info => level.to_string().cyan(),
                Level::Debug => level.to_string().purple(),
                Level::Trace => level.to_string().white(),
            };

            let log_line = if let Some(file) = record.file() {
                if let Some(line) = record.line() {
                    format!(
                        "{} [{}] {} - {}:{} - {} - {}",
                        time, level_colored, target, file, line, thread_name, message
                    )
                } else {
                    format!(
                        "{} [{}] {} - {} - {}",
                        time, level_colored, target, thread_name, message
                    )
                }
            } else {
                format!(
                    "{} [{}] {} - {} - {}",
                    time, level_colored, target, thread_name, message
                )
            };

            writeln!(buf, "{}", log_line)
        })
        .filter(None, log_level.unwrap_or(LevelFilter::Debug))
        .target(target.unwrap_or(env_logger::Target::Stdout))
        .init();
}

// Helper function to run a Python script with Poetry
pub fn run_poetry_command(script_path: &Path, args: &[&str]) {
    let output = Command::new("poetry")
        .arg("run")
        .arg("python3")
        .arg("-m")
        .arg(script_path)
        .args(args)
        .output()
        .expect("Failed to execute command");

    log::debug!("{}", String::from_utf8_lossy(&output.stdout));

    if !output.stderr.is_empty() {
        log::debug!("{}", String::from_utf8_lossy(&output.stderr));
    }
}

pub fn download_sheet_to_csv(input_file_path: &Path, config: &Config) {
    let script_path = Path::new("scripts.download_sheet_to_csv");
    let args = [
        input_file_path.to_str().unwrap(),
        &config.google.sheet_id,
        &config.google.input_range_name,
    ];
    run_poetry_command(script_path, &args);
}

pub fn upload_csv_to_sheet(output_file_path: &Path, config: &Config) {
    let script_path = Path::new("scripts.upload_csv_to_sheet");
    let args = [
        output_file_path.to_str().unwrap(),
        &config.google.sheet_id,
        &config.google.output_range_name,
    ];

    run_poetry_command(script_path, &args);
}

/// Rounds a floating-point number to a specified number of digits, with an adjustable rounding threshold.
///
/// This function allows for more nuanced control over the rounding process compared to the standard `.round()`
/// method, by permitting the specification of a threshold for rounding up based on a sequence of nines.
///
/// # Arguments
///
/// * `x` - The floating-point number to round.
/// * `num_digits` - The number of digits to round to.
/// * `num_9s` - The number of consecutive nines (`9`) that define the upper rounding threshold. This effectively
///   determines how "aggressive" the rounding should be. A higher value results in a narrower range where numbers
///   are rounded up.
///
/// # Examples
///
/// Basic rounding without adjusting the threshold:
///
/// ```
/// let result = round(3.14159, 2, 0); // Rounds to 3.14
/// ```
///
/// Rounding with an adjusted threshold:
///
/// ```
/// let result = round(3.14159, 2, 1); // Might round to 3.15 if within threshold
/// ```
///
/// # Returns
///
/// The rounded floating-point number.
///
/// # Note
///
/// The function's behavior is unique for values where the fractional part is close to `.5`. The `num_9s` argument
/// adjusts the threshold for when values are rounded up or down, allowing for finer control over rounding decisions
/// near half-integer values.
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
