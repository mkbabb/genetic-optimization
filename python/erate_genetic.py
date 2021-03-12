import math
import random
from typing import *
from datetime import datetime

import numba
import numpy as np
import pandas as pd
import pathlib
import argparse


from numba_optimizer_v1 import life as life1


def repeat_expand(x: np.ndarray) -> np.ndarray:
    return np.repeat(x.reshape((-1, 1)), 1, axis=1)


def make_ixs(init_buckets: np.ndarray, buckets: int):
    ixs = np.full((init_buckets.size, buckets), 0)
    for i in range(buckets):
        mask = init_buckets == i
        ixs[mask, i] = 1
    return ixs


def set_buckets(ixs: np.ndarray, df: pd.DataFrame):
    for i in range(ixs.shape[1]):
        mask = ixs[..., i] == 1
        df["bucket"].values[mask] = i
    return df


@numba.njit(fastmath=True, parallel=False)
def calc_cost(ixs: np.ndarray) -> float:
    global costs, discounts
    total = 0

    for n, ix in enumerate(ixs.T):
        ix = np.expand_dims(ix, 1)
        z_count = np.count_nonzero(ix)

        if z_count > 0:
            avg_discounts = np.round(np.sum(discounts * ix) / z_count) / 100.0
            bucket_costs = np.sum(costs * ix)
            discount_cost = avg_discounts * bucket_costs

            total += discount_cost

    return total


def main():
    global costs, discounts

    t = int(datetime.now().timestamp())
    now = datetime.now().isoformat()

    parser = argparse.ArgumentParser()

    parser.add_argument("-n", default=10 ** 6)
    parser.add_argument("--pop_size", default=100)
    parser.add_argument("--buckets", default=4)
    parser.add_argument("--seed", default=t)

    parser.add_argument("--dirpath", default="data/2021-optimization")
    parser.add_argument("-i", "--in_filepath", default=None)

    args = parser.parse_args()

    dirpath = pathlib.Path(args.dirpath)

    tmp_filepath = dirpath.joinpath("tmp.csv")
    out_filepath = dirpath.joinpath(f"out-{now}-{t}.csv")

    def get_in_filepath_seed():
        if args.in_filepath is None:
            return args.seed, tmp_filepath
        else:
            path = pathlib.Path(args.in_filepath)

            if len((comps := path.name.split("-"))) == 3:
                _, _, f_seed = comps
                return int(f_seed), path
            else:
                return args.seed, path

    seed, in_filepath = get_in_filepath_seed()

    random.seed(seed)
    np.random.seed(seed)

    df = pd.read_csv(in_filepath)

    init_buckets = df["bucket"].values
    costs = repeat_expand(df["cost"].values)
    discounts = repeat_expand(df["discount"].values)

    fitness_func = calc_cost

    critters = np.asarray(
        [make_ixs(init_buckets, args.buckets) for _ in range(args.pop_size)]
    )

    life = life1

    max_critter = life(critters, args.n, args.pop_size, fitness_func)

    df = set_buckets(max_critter, df)
    df.to_csv(out_filepath, index=False)
    df.to_csv(tmp_filepath, index=False)


if __name__ == "__main__":
    main()
