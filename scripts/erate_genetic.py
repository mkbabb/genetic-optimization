from __future__ import annotations

import json
import locale
import pathlib
import pprint
import random
import re
import subprocess
import sys
import tempfile
import tomllib  # type: ignore
from typing import *

import ipinfo  # type: ignore
import numpy as np
import openai
import pandas as pd
from loguru import logger
from openai.types.chat import ChatCompletion

from googleapiutils2 import Drive, GoogleMimeTypes, Sheets, SheetSlice, get_oauth2_creds

FitnessFunction = Callable[[np.ndarray, np.ndarray, np.ndarray], float]
SelectionMethodFunction = Callable[[list[np.ndarray], list[float]], np.ndarray]

logger_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{module}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)
logger.configure(handlers=[{"sink": sys.stdout, "format": logger_format}])


locale.setlocale(locale.LC_ALL, "en_US.UTF-8")

DEBUG = False


def generate_population(pop_size: int, N: int, buckets: int) -> list[np.ndarray]:
    population = []
    for _ in range(pop_size):
        X = np.zeros((N, buckets))
        for i in range(N):
            X[i, random.randint(0, buckets - 1)] = 1

        population.append(X)

    return population


def initialize_population_from_solution(
    df: pd.DataFrame, pop_size: int, N: int, buckets: int
) -> list[np.ndarray]:
    initial_population = []

    for _ in range(pop_size):
        X = np.zeros((N, buckets))

        for i, bucket in enumerate(df["bucket"].values):
            bucket = int(bucket) - 1  # buckets are 1-indexed
            X[i, bucket] = 1

        initial_population.append(X)

    return initial_population


def calculate_objective(
    X: np.ndarray, costs: np.ndarray, discounts: np.ndarray
) -> float:
    # Calculate the total cost per bucket
    total_costs_per_bucket = X.T @ costs
    total_discounts_per_bucket = X.T @ discounts
    items_per_bucket = X.T.sum(axis=1)
    avg_discounts_per_bucket = np.round(
        total_discounts_per_bucket / items_per_bucket, 2
    )
    discount_costs_per_bucket = total_costs_per_bucket * avg_discounts_per_bucket
    discount_costs = discount_costs_per_bucket.sum()

    if DEBUG:
        total_costs_per_bucket_str = [
            locale.currency(cost, grouping=True) for cost in total_costs_per_bucket
        ]
        avg_discounts_per_bucket_str = [
            f"{discount * 100}%" for discount in avg_discounts_per_bucket
        ]
        discount_costs_per_bucket_str = [
            locale.currency(cost, grouping=True) for cost in discount_costs_per_bucket
        ]
        discount_costs_str = locale.currency(discount_costs, grouping=True)

        logger.info(f"Total costs per bucket: {total_costs_per_bucket_str}")
        logger.info(f"Average discounts per bucket: {avg_discounts_per_bucket_str}")
        logger.info(f"Discount costs per bucket: {discount_costs_per_bucket_str}")
        logger.info(f"Total discount costs: {discount_costs_str}")

    return discount_costs


def k_point_crossover(parents: list[np.ndarray], k: int = 2) -> np.ndarray:
    num_parents = len(parents)

    if num_parents < 2:
        raise ValueError("There must be at least two parents for crossover.")

    # Determine the shape of the parent matrices
    n_rows, n_cols = parents[0].shape

    # Generate sorted crossover points
    crossover_points = sorted(random.sample(range(1, n_rows - 1), k))

    # Initialize the child matrix with zeros or copy from the first parent
    child = np.zeros_like(parents[0])

    # Starting index for the first segment
    start_idx = 0

    # Perform crossover operations
    for i, end_idx in enumerate(crossover_points + [n_rows]):
        # Select the parent for the current segment
        parent_idx = i % num_parents
        # Copy the segment from the selected parent to the child
        child[start_idx:end_idx, :] = parents[parent_idx][start_idx:end_idx, :]
        # Update the starting index for the next segment
        start_idx = end_idx

    return child


def mutation(X: np.ndarray, mutation_rate: float) -> np.ndarray:
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if random.random() < mutation_rate:
                X[i, j] = 0
                X[i, random.randint(0, X.shape[1] - 1)] = 1
    return X


def repair_constraint(X: np.ndarray) -> np.ndarray:
    for i in range(X.shape[0]):
        if X[i].sum() > 1:
            nonzero_indices = np.nonzero(X[i])[0]
            X[i, nonzero_indices[1:]] = 0
    return X


def tournament_selection(
    population: list[np.ndarray], fitnesses: list[float], tournament_size: int = 3
) -> np.ndarray:
    selected_idx = np.random.choice(len(population), tournament_size)
    best_fitness_idx = selected_idx[np.argmax(fitnesses[selected_idx])]

    return population[best_fitness_idx]


def roulette_wheel_selection(
    population: list[np.ndarray], fitnesses: list[float]
) -> np.ndarray:
    total_fitness = sum(fitnesses)
    probabilities = [f / total_fitness for f in fitnesses]
    selected_idx = np.random.choice(len(population), p=probabilities)

    return population[selected_idx]


def rank_selection(population: list[np.ndarray], fitnesses: list[float]) -> np.ndarray:
    ranks = np.argsort(fitnesses)[::-1]  # Descending order of fitness
    probabilities = (ranks + 1) / ranks.sum()  # Linear ranking
    selected_idx = np.random.choice(len(population), p=probabilities)

    return population[selected_idx]


def run_genetic_algorithm(
    df: pd.DataFrame,
    buckets: int,
    pop_size: int = 100,
    k: int = 2,
    mutation_rate: float = 0.1,
    generations: int = 100,
    fitness_func: Callable[
        [np.ndarray, np.ndarray, np.ndarray], float
    ] = calculate_objective,
    selection_method_func: Callable[
        [list, list], np.ndarray
    ] = roulette_wheel_selection,
    num_parents: int = 2,
    num_top_parents: int = 5,
    writer: Optional[Callable[[np.ndarray, float], None]] = None,
) -> Optional[np.ndarray]:

    N = len(df)
    population = initialize_population_from_solution(
        df=df, pop_size=pop_size, N=N, buckets=buckets
    )

    costs: np.ndarray = np.array(df["cost"].values)
    discounts: np.ndarray = np.array(df["discount"].values) / 100.0

    best_solution = population[0]
    best_fitness = fitness_func(best_solution, costs, discounts)

    logger.info(f"Initial best fitness: {locale.currency(best_fitness, grouping=True)}")

    no_improvement_counter = 0
    for n in range(generations):
        fitnesses = [fitness_func(X, costs, discounts) for X in population]

        # Sort population based on fitness and keep the top num_top_parents
        indices = np.argsort(fitnesses)[::-1]  # Sort in descending order
        top_parents = [population[ix] for ix in indices[:num_top_parents]]

        new_population = top_parents.copy()
        for _ in range(
            (pop_size - len(top_parents)) // num_parents
        ):  # only spawn enough children to fill the population
            parents = [
                selection_method_func(population, fitnesses) for _ in range(num_parents)
            ]

            child = k_point_crossover(parents=parents, k=k)
            child = mutation(X=child, mutation_rate=mutation_rate)
            child = repair_constraint(X=child)

            new_population.append(child)

        population = new_population

        best_fitness_ix = np.argmax(fitnesses)
        t_best_solution = population[best_fitness_ix]
        t_best_fitness = fitnesses[best_fitness_ix]

        logger.info(
            f"Generation {n} best fitness: {locale.currency(t_best_fitness, grouping=True)}"
        )

        if t_best_fitness >= best_fitness:
            best_solution = t_best_solution
            best_fitness = t_best_fitness
            no_improvement_counter = 0

            if writer is not None:
                writer(best_solution, best_fitness)
        else:
            no_improvement_counter += 1

        # Exponential backoff check
        if no_improvement_counter >= 2 ** (
            n // 2
        ):  # Adjust the base of the exponent as needed
            logger.info(
                "Resetting population to previous best solution due to stagnation."
            )
            population = [best_solution] * pop_size
            no_improvement_counter = 0

    return best_solution


drive = Drive()
sheets = Sheets()

config_path = pathlib.Path("./config.toml")
config = tomllib.loads(
    config_path.read_text(),
)

optimization_sheet_id = config["google"]["optimization_sheet_id"]
input_range_name = config["google"]["input_range_name"]
output_range_name = config["google"]["output_range_name"]


df = sheets.to_frame(
    sheets.values(
        spreadsheet_id=optimization_sheet_id,
        range_name=input_range_name,
    )
)

buckets = df["bucket"].nunique()


def write_to_google_sheet(X: np.ndarray, fitness: float) -> None:
    bucket_assignments = np.argmax(X, axis=1) + 1  # +1 to make it 1-indexed

    output_df = df.copy()
    output_df["bucket"] = bucket_assignments

    sheets.batch_update(
        spreadsheet_id=optimization_sheet_id,
        data={
            output_range_name: sheets.from_frame(output_df),
        },
    )


run_genetic_algorithm(
    df=df,
    pop_size=2000,
    buckets=buckets,
    k=10,
    mutation_rate=0.01,
    generations=int(1e5),
    fitness_func=calculate_objective,
    selection_method_func=roulette_wheel_selection,
    num_parents=2,
    writer=write_to_google_sheet,
)
