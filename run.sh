#!/bin/bash
bucket_counts="4 5"
parent_counts="2 5"
population_counts="100 200"
crossover_counts="5"
n=0

mating_pool_count=100
max_bucket=200
mutation_rate=1

nuke_threshold=1000
nuke_threshold_max=100000
nuke_mutation_percent=10
nuke_mutation_percent_max=90
nuke_growth_rate=2
nuke_burnout=1

iterations=10000000
current_best=10764931.82

mkdir -p ./data/out

for bucket_count in $bucket_counts; do
    for parent_count in $parent_counts; do
        for population_count in $population_counts; do
            for crossover_count in $crossover_counts; do
                console_out_file=./data/out/console-$n.txt
                touch $console_out_file

                echo "**"bucket_count: $bucket_count, parent_count: $parent_count, population_count: $population_count, crossover_count: $crossover_count >$console_out_file

                ./erate.so -i ./data/discount_rates_invoices_10_2019.csv -o ./data/out/run-$n.csv --bucket_count $bucket_count --population_count $population_count --max_bucket $max_bucket --parent_count $parent_count --mating_pool_count $mating_pool_count --mutation_rate $mutation_rate --nuke_threshold $nuke_threshold --nuke_threshold_max $nuke_threshold_max --nuke_mutation_percent $nuke_mutation_percent --nuke_mutation_percent_max $nuke_mutation_percent_max --nuke_growth_rate $nuke_growth_rate --nuke_burnout $nuke_burnout --iterations $iterations --rng_state $n --current_best $current_best >$console_out_file &
                ((n++))
            done
        done
    done
done
