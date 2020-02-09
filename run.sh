#!/bin/bash
in_file="./data/csv-for-optimization.csv"

bucket_counts="4 5"
parent_counts="50 100"
population_counts="200 500"
crossover_counts="5"
n=0

mating_pool_count=100
max_bucket=200
mutation_rate=1

nuke_threshold=100
nuke_threshold_max=100000
nuke_mutation_percent=10
nuke_mutation_percent_max=90
nuke_growth_rate=2
nuke_burnout=1

iterations=10000000
current_best=11271171.92

out_dir=./data/$(date +%s)
mkdir -p $out_dir

for bucket_count in $bucket_counts; do
    for parent_count in $parent_counts; do
        for population_count in $population_counts; do
            for crossover_count in $crossover_counts; do

                console_out_file=$out_dir/console-$n.txt
                description_file=$out_dir/description-$n.txt

                touch $console_out_file
                touch $description_file

                chmod 666 $console_out_file
                chmod 666 $description_file

                echo "**"bucket_count: $bucket_count, parent_count: $parent_count, population_count: $population_count, crossover_count: $crossover_count >$description_file

                ./erate.so -i $in_file -o ./data/out/run-$n.csv --bucket_count $bucket_count --population_count $population_count --max_bucket $max_bucket --parent_count $parent_count --mating_pool_count $mating_pool_count --mutation_rate $mutation_rate --nuke_threshold $nuke_threshold --nuke_threshold_max $nuke_threshold_max --nuke_mutation_percent $nuke_mutation_percent --nuke_mutation_percent_max $nuke_mutation_percent_max --nuke_growth_rate $nuke_growth_rate --nuke_burnout $nuke_burnout --iterations $iterations --rng_state $n --current_best $current_best >$console_out_file &
                ((n++))
            done
        done
    done
done
