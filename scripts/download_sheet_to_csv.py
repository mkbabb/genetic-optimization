# download_sheet_to_csv.py
import argparse
import pathlib

import pandas as pd
from googleapiutils2 import Sheets
from loguru import logger

from .utils import init_parser


def download_sheet_to_csv(
    file_path: pathlib.Path, file_id: str, range_name: str
) -> None:
    sheets = Sheets()

    try:
        df = sheets.to_frame(
            sheets.values(spreadsheet_id=file_id, range_name=range_name)
        )
        df.to_csv(file_path, index=False)
        logger.info(
            f"Downloaded data to {file_path} from sheet ID: {file_id}, range: {range_name}"
        )
    except Exception as e:
        logger.error(f"Failed to download sheet to CSV: {e}")


if __name__ == "__main__":

    args = init_parser().parse_args()
    download_sheet_to_csv(args.file_path, args.file_id, args.range_name)
