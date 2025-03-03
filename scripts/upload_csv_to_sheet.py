import pathlib

import pandas as pd
from googleapiutils2 import Sheets, ValueInputOption
from loguru import logger

from .utils import init_parser


def upload_csv_to_sheet(file_path: pathlib.Path, file_id: str, range_name: str) -> None:
    sheets = Sheets()

    try:
        df = pd.read_csv(file_path)
        sheets.update(
            spreadsheet_id=file_id,
            range_name=range_name,
            values=sheets.from_frame(df),
            value_input_option=ValueInputOption.user_entered,
        )
        logger.info(f"Uploaded {file_path} to sheet ID: {file_id}, range: {range_name}")
    except Exception as e:
        logger.error(f"Failed to upload CSV to sheet: {e}")


if __name__ == "__main__":
    args = init_parser().parse_args()
    upload_csv_to_sheet(args.file_path, args.file_id, args.range_name)
