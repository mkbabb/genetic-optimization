import subprocess


make_screen = lambda x: f"""screen -dm bash -c '{x}'"""

make_command = (
    lambda args: f"""poetry shell;
python3 python/erate_genetic.py {args};
exec sh"""
)


def make_args(d: dict):
    return " ".join((f"--{k} {v}" for k, v in d.items()))


buckets = [4]
pop_sizes = [50, 50, 100, 200]
n = 1 * 10 ** 7

thread_number = 0

for bucket in buckets:
    for pop_size in pop_sizes:
        args = dict(
            bucket=bucket,
            pop_size=pop_size,
            n=n,
            thread_number=thread_number,
            in_filepath="data/2021-optimization/tmp.csv"
        )
        screen_command = make_screen(make_command(make_args(args)))

        process = subprocess.Popen(
            screen_command, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE
        )
        process.communicate()

        thread_number += 1
