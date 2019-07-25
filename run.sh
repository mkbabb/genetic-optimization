#!/bin/bash
parents="2 5 10"
n=0
for i in {3..5}
do
    for j in $parents
    do
        for k in {1..2}
        do
            ./erate.so -i ./erate-data/erate-2019-start.csv -o ./erate-data/out-$n.csv --bucket_count $i --population_count 200 --max_bucket 200 --parent_count $j --mating_pool_count 20 --mutation_rate $k --mutation_threshold_low 100000 --mutation_threshold_high 500000 --iterations 100000000 > ./erate-data/console-$n.txt &
            ((n++))
        done
    done
done