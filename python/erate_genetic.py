import random
from typing import *
from datetime import datetime

import numba
import numpy as np
import pandas as pd
import pathlib
import argparse


from numba_optimizer_v0 import life as life_v0
from numba_optimizer_v1 import life as life_v1


FitnessFunc = Callable[[np.ndarray], float]


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


def erate_genetic(
    buckets: int,
    n: int,
    pop_size: int,
    life_loop: Callable,
    fitness_func: FitnessFunc,
    in_filepath: pathlib.Path,
    out_filepath: pathlib.Path,
    tmp_filepath: pathlib.Path,
    **kwargs,
) -> None:
    global costs, discounts

    df = pd.read_csv(in_filepath)

    init_buckets = df["bucket"].values
    costs = repeat_expand(df["cost"].values)
    discounts = repeat_expand(df["discount"].values)

    critters = np.asarray([make_ixs(init_buckets, buckets) for _ in range(pop_size)])

    max_critter = life_loop(
        critters=critters,
        n=n,
        pop_size=pop_size,
        fitness_func=fitness_func,
    )

    df = set_buckets(max_critter, df)
    df.to_csv(out_filepath, index=False)
    df.to_csv(tmp_filepath, index=False)


def meta_erate_genetic():
    pass


def setup(**kwargs) -> dict:
    t = int(datetime.now().timestamp())
    now = datetime.now()

    defaults = dict(
        n=10 ** 6,
        pop_size=100,
        seed=t,
        thread_number="",
        dirpath=f"data/{now.year}-optimization",
    )
    defaults.update(kwargs)
    kwargs = defaults

    dirpath = pathlib.Path(kwargs.get("dirpath")).joinpath(kwargs.get("thread_number"))
    dirpath.mkdir(parents=True, exist_ok=True)

    tmp_filepath = dirpath.joinpath("tmp.csv")
    out_filepath = dirpath.joinpath(f"out-{now.isoformat()}-{t}.csv")

    def get_in_filepath_seed():
        if "in_filepath" not in kwargs:
            return kwargs.get("seed"), tmp_filepath
        else:
            path = pathlib.Path(kwargs.get("in_filepath"))

            if len((comps := path.name.split("-"))) == 3:
                _, _, f_seed = comps
                return int(f_seed), path
            else:
                return kwargs.get("seed"), path

    seed, in_filepath = get_in_filepath_seed()

    random.seed(seed)
    np.random.seed(seed)

    kwargs.update(
        dict(
            in_filepath=in_filepath,
            tmp_filepath=tmp_filepath,
            out_filepath=out_filepath,
            dirpath=dirpath,
        )
    )

    return kwargs


def run(**kwargs) -> None:
    kwargs = setup(**kwargs)
    erate_genetic(life_loop=life_v1, fitness_func=calc_cost, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("-n", "--n", type=int)
    parser.add_argument("--pop_size", type=int)
    parser.add_argument("--buckets", default=4, type=int)
    parser.add_argument("--seed", type=int)

    parser.add_argument("--thread_number", type=str)

    parser.add_argument("--dirpath")
    parser.add_argument("-i", "--in_filepath")

    args = parser.parse_args()
    kwargs = {k: v for k, v in args.__dict__.items() if v is not None}

    run(**kwargs)


if __name__ == "__main__":
    main()
