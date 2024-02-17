from __future__ import annotations

import os
import pathlib
import time
import tomllib  # type: ignore
from typing import *

import pandas as pd
from googleapiutils2 import Drive, Sheets, get_oauth2_creds
from loguru import logger


def watch_file(
    file_path: pathlib.Path,
    fn: Callable[[pathlib.Path], None],
    check_interval: float = 1.0,
) -> None:
    last_modified_time = os.path.getmtime(file_path)

    while True:
        try:
            current_modified_time = os.path.getmtime(file_path)
            if current_modified_time != last_modified_time:

                fn(file_path)

                last_modified_time = current_modified_time
        except Exception as e:
            logger.error(e)
            pass

        time.sleep(check_interval)


drive = Drive()
sheets = Sheets()

config_path = pathlib.Path("./config.toml")
config = tomllib.loads(
    config_path.read_text(),
)

optimization_sheet_id = config["google"]["optimization_sheet_id"]
output_range_name = config["google"]["output_range_name"]


file_path = pathlib.Path("./data/output.csv")


def fn(file_path: pathlib.Path) -> None:
    logger.info(f"Updating {file_path}")

    output_df = pd.read_csv(file_path)

    sheets.batch_update(
        spreadsheet_id=optimization_sheet_id,
        data={
            output_range_name: sheets.from_frame(output_df),
        },
    )


fn(file_path)

watch_file(
    file_path=file_path,
    fn=fn,
)
