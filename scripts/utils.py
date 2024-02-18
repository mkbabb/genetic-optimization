import argparse
import pathlib


def init_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download data from a Google Sheet to a CSV file."
    )
    parser.add_argument(
        "file_path", type=pathlib.Path, help="Path to the output CSV file."
    )
    parser.add_argument("file_id", type=str, help="Google Sheet ID.")
    parser.add_argument("range_name", type=str, help="Range name in A1 notation.")

    return parser
