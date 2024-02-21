# `genetic-optimization`

Herein, we define the implementation of a generalized genetic algorithm (GA), a class of evolutionary algorithms that simulate the process of natural selection. This approach iteratively improves a population of candidate solutions based on the principles of genetic inheritance and Darwinian strife for survival. Each solution is represented as a chromosome (a 2-d array, or matrix); the algorithm uses selection, crossover, and mutation operations to evolve the population towards an optimal solution.

## Representation

Each candidate solution, or chromosome, represents a potential partitioning of the dataset into buckets. This is encoded as a matrix where each row corresponds to an item, and columns represent buckets, with the matrix entries indicating the allocation of items to buckets. Items are assigned to exactly one bucket, and the chromosome's structure ensures that this constraint is satisfied.

## Function Components

The genetic algorithm is composed of several modular functions, each of which performs a specific operation. These functions can be customized and combined to create a genetic algorithm tailored to a specific problem.

### `fitness_func`; Fitness Function

The fitness function evaluates how well a given solution performs. As input, it takes only the chromosome and must return some measure of its fitness.

This function is user-defined and specific to a given problem. For our case, it calculates the aggregate `discount-cost` for a partitioning.

### `selection_func`; Selection

Various selection methods are implemented:

-   **Tournament Selection**: Randomly selects a subset of chromosomes from the population and chooses the best one based on their fitness values.
-   **Roulette Wheel Selection**: Assigns a probability to each chromosome based on its fitness, and then selects parents based on these probabilities.
-   **Rank Selection**: Assigns a probability to each chromosome based on its rank in the population, and then selects parents based on these probabilities.
-   **Stochastic Universal Sampling**: Similar to roulette wheel selection, but selects multiple parents at once, ensuring that the selected parents are spread out across the population.

This process is repeated to select multiple parents for the crossover operation.

### `mating_func`; Crossover; Mating

Various crossover, or mating, methods are implemented:

-   **k-point Crossover**: `k` points are randomly selected on the chromosome, and the genetic material between these points is exchanged between parents to produce offspring. This interleaves, at each `k`-crossover boundary, the genetic material.
-   **Uniform Crossover**: Each gene is inherited from one of the parents with equal probability, resulting in a more diverse offspring.

### `mutation_func`; Mutation

-   **Standard Mutation**: With a certain probability, the mutation operation introduces random changes to offspring, promoting genetic diversity within the population. This helps prevent premature convergence to suboptimal solutions.
-   **Gaussian Mutation**: Adds a small amount of Gaussian noise to the offspring, allowing for more fine-grained exploration of the solution space.
-   **Bit-Flip Mutation**: Randomly flips bits in the chromosome, introducing small changes to the offspring.
-   **Uniform Mutation**: Randomly changes the value of genes in the chromosome with a certain probability.

### `writer_func`, Writing the Result

When a new best solution is found, a function is called to write the result. This function is user-defined: for our case, it prints the best solution found hitherto and then writes it to a Google Sheet via a Python Google Sheets shim.

## Algorithmic Implementation

### Elitism

A certain number of the best-performing chromosomes from the current generation are preserved and directly carried over to the next generation. This ensures that the best solutions are not lost and continue to be part of the evolving population.

The `num_elites` hyperparameter specifies this number.

### Stagnant Mating Pool Culling

We implement an exponential-backoff culling mechanism to ensure that the mating pool does not become stagnant. If a new best solution is not found within `no_improvement_counter` iterations, the mating pool culled and replaced with the current best solution.

Each time the pool is culled, the number of generations before the next culling is multiplicatively increase by `culling_percent_increment`.

The `culling_percent` parameter is modified by the `max_culling_percent` , `min_culling_percent`, `culling_direction`, and `culling_percent_increment` hyperparameters:

-   `max_culling_percent` and `min_culling_percent`: Specifies the maximum and minimum culling percentages, respectively.
-   `culling_direction`: Specifies the strategy at which to either increase or decrease the `culling_percent`.
-   `culling_percent_increment`: Specifies the amount by which to, multiplicativley, increase or decrease the `culling_percent` at each culling.

#### Culling Mechanism

A new culled population is constructed as follows:

-   A selection of clones of the `best_solution` are added to the culled population, but mutated.
-   The remaining slots are filled with a random subset of the current population.
-   Finally, `num_elites` of the pure `best_solution` are set to the front of the culled population array.

### Hyperparameters and Configuration

The above-mentioned components of the genetic algorithm are configurable, allowing users to customize the algorithm to suit their specific problem and requirements. This is facilitated through the [configuration file](./config.toml), which allows users to specify the chosen configurable hyperparameters. See that file for a full list of hyperparameters and their detailed-descriptions.

### Implementation Details

The genetic algorithm is implemented in Rust, a systems programming language known for its performance, safety, and concurrency features. The algorithm is designed to be highly parallelizable, enabling efficient execution on multi-core CPUs and GPUs. Herein, we leverage several Rust packages and features to achieve this:

-   **Rayon**: A data-parallelism library that provides easy-to-use, efficient parallel iterators and parallel data structures.
-   **Polars**: A fast, efficient, and easy-to-use data manipulation library that provides a DataFrame abstraction and supports parallel execution.
-   **NDArray**: A library for n-dimensional arrays that provides efficient, parallelized operations on multi-dimensional data.

## Usage Scenario

A typical usage scenario for the genetic algorithm involves the following steps:

-   First, the user defines the fitness function, which evaluates the quality of a given partitioning solution.
-   Next, the user configures the genetic algorithm by specifying the population size, the number of generations, the selection, crossover, and mutation methods, and other parameters.
-   The user then provides the initial population of candidate solutions, which can be randomly generated or based on some heuristic or prior knowledge.
-   The genetic algorithm iteratively evolves the population, applying selection, crossover, mutation, and constraint handling operations to produce new generations of candidate solutions.

## E-Rate School Partitioning

The impetus behind this library, and thus the core objective of the project was, to optimize the partitioning of a dataset to minimize the overall aggregate `discount-cost`.

This dataset comprises multiple entries, each with attributes including an identifier, a discount value, and a cost:

-   `ben`: The unique identifier for the entry (Billed Entity Number)
-   `psu-id`: The identifier for the PSU (Public School Unit)
-   `psu-name`: The name of the PSU
-   `bw`: The bandwidth
-   `discount`: The discount percent applicable to the entry; integral value
-   `cost`: The cost of the entry; integral value
-   `bucket`: The bucket to which the entry is assigned; integral value

These entries are to be distributed into distinct buckets, with the aim to minimize the total cost across all buckets, factoring in the discounts applicable to each.

### Objective (Fitness) Function and Constraints

Let \(X\) be a binary matrix where each element \(x\_{ij}\) indicates whether item \(i\) is placed in bucket \(j\). Each item \(i\) has an associated cost, denoted by \(cost_i\), and a discount, denoted by \(discount_i\).

The `discount-cost` for a single bucket \(j\) is calculated as:

\[ \text{`discount-cost`}_j = \left( \sum_{i=1}^{N} x*{ij} \cdot cost_i \right) \cdot \left( \frac{\sum*{i=1}^{N} x*{ij} \cdot discount_i}{\sum*{i=1}^{N} x\_{ij}} \right) \]

where \(N\) is the total number of items.

The overall aggregate `discount-cost` across all buckets is the sum of the `discount-cost`s for each bucket:

\[ \text{Objective} = \sum\_{j=1}^{K} \text{`discount-cost`}\_j \]

where \(K\) is the number of buckets.
