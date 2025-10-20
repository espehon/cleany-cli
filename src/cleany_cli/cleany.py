# Copyright (c) 2025 espehon
# MIT License

# region: Imports
# Standard library imports
import sys
import os
from typing import Union, Iterator, Optional, Callable
from collections import defaultdict
from dataclasses import dataclass, field

# Third-party imports
import pandas as pd
import numpy as np

import questionary as q
from halo import Halo


# endregion
# region: Startup

SUPPORTED_FILE_TYPES = ('.csv', '.xlsx', '.xls', '.json', '.parquet', '.feather', '.xml')

FILE_SIZE_LIMIT = 1024 * 1024 * 1024    # 1 GB
CHUNK_SIZE = 100000                     # 100k rows per chunk for large files
STREAMABLE = {'.csv', '.tsv',}

header_widths = {
    "Column": 16,
    "dtype": 8,
    "Missing": 10,
    "Minimum": 14,
    "Average": 14,
    "Maximum": 14,
    "Sample": 14
}

features = [
    "Preview data (datatypes, sample rows, summary statistics)",
    # "Rename columns",
    # "Remove columns",
    # "Remove duplicate rows",
    # "Handle missing values (drop or fill)",
    # "Filter rows based on column values",
    # "Remove outliers (IQR or Z-score method)",
    # "Save cleaned data to new file or overwrite existing (overwrite is not recommended)",
    # "Can save in different formats (see supported file types above)"
]


# endregion
# region: Classes


@dataclass
class ColumnStats:
    dtype: Optional[str] = None
    missing: int = 0
    total: int = 0
    min: Optional[str] = None
    max: Optional[str] = None
    mean_sum: float = 0.0
    mean_count: int = 0
    samples: set[str] = field(default_factory=set)





# endregion
# region: Functions


def help() -> None:
    help_text = f"""
Cleany - Interactive CLI tool to clean and prune tabular data files.

Usage:
    cleany [file_path]
If no file_path is provided, you will be prompted to select a supported file in the current directory.

Supported file types: {', '.join(SUPPORTED_FILE_TYPES)}

Features:
    - Preview data (datatypes, sample rows, summary statistics)
    - Rename columns
    - Remove columns
    - Remove duplicate rows
    - Handle missing values (drop or fill)
    - Filter rows based on column values
    - Remove outliers (IQR or Z-score method)
    - Save cleaned data to new file or overwrite existing (overwrite is not recommended)
        - Can save in different formats (see supported file types above)
"""
    print(help_text)

def inspect_argument(arg: str) -> None:
    if arg in ('-?', '--help'):
        help()
        sys.exit(0)
    elif arg[0] in ('-', '<', '>', ':', '|', '?', '*', '$', '@', '!'):
        print("Unrecognized argument. Use -? or --help for usage information.")
        sys.exit(1)
    else:
        return

def file_size_print(filepath: str) -> None:
    size = os.path.getsize(filepath)
    if size < 1024:
        print(f"File size: {size} bytes")
    elif size < 1024 * 1024:
        print(f"File size: {size / 1024:.2f} KB")
    elif size < 1024 * 1024 * 1024:
        print(f"File size: {size / (1024 * 1024):.2f} MB")
    else:
        print(f"File size: {size / (1024 * 1024 * 1024):.2f} GB")


def get_file_list() -> list:
    file_list = [f for f in os.listdir('.') if os.path.isfile(f) and f.lower().endswith(SUPPORTED_FILE_TYPES)]
    return file_list


# def count_rows(reader: Iterator[pd.DataFrame]) -> int:
#     return sum(len(chunk) for chunk in reader)


def find_file(filepath: str="") -> str:
    try:
        if filepath != "":
            if os.path.isfile(filepath):
                return filepath
            else:
                print(f"File not found: {filepath}")
                user = q.confirm("Do you want to search for a file in the current directory?", default=False).ask()
                if user is False:
                    print("Exiting...")
                    sys.exit(0)
        files = get_file_list()
        if len(files) == 0:
            print("No supported files found in the current directory.")
            sys.exit(0)
        file = q.select("Select a file to clean:", choices=files).ask()
        return file
    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Exiting...")
        sys.exit(0)


def load_file(filepath: str, chunksize: int = 10000) -> Iterator[pd.DataFrame]:
    ext = os.path.splitext(filepath)[-1].lower()
    file_size = os.path.getsize(filepath)

    # Chunked loading for large streamable files
    if file_size > FILE_SIZE_LIMIT and ext in STREAMABLE:
        sep = ',' if ext == '.csv' else '\t'
        return pd.read_csv(filepath, sep=sep, chunksize=chunksize)

    # Full load + manual chunking
    if ext == '.csv':
        df = pd.read_csv(filepath)
    elif ext == '.tsv':
        df = pd.read_csv(filepath, sep='\t')
    elif ext in ('.xlsx', '.xls'):
        df = pd.read_excel(filepath)
    elif ext == '.json':
        df = pd.read_json(filepath)
    elif ext == '.parquet':
        df = pd.read_parquet(filepath)
    elif ext == '.feather':
        df = pd.read_feather(filepath)
    elif ext == '.xml':
        df = pd.read_xml(filepath)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    # Normalize to generator
    return (df[i:i+chunksize] for i in range(0, len(df), chunksize))


def format_percent(number, total):
    pct = (number / total) * 100 if total else 0
    return f"{pct:6.2f}%"

def truncate(text, width):
    return text[:width-1] + '…' if len(text) > width else text.ljust(width)

def preview_full_dataset(data: Iterator[pd.DataFrame], sample_size: int = 1) -> None:
    headers = list(header_widths.keys())
    print("".join(h.ljust(header_widths[h]) for h in headers))
    print("-" * sum(header_widths.values()))

    stats: dict[str, ColumnStats] = defaultdict(ColumnStats)

    for chunk in data:
        for col in chunk.columns:
            series = chunk[col]
            col_stats = stats[col]

            if col_stats.dtype is None:
                col_stats.dtype = str(series.dtype)

            col_stats.missing += series.isnull().sum()
            col_stats.total += len(series)

            non_null = series.dropna()

            if pd.api.types.is_numeric_dtype(series):
                min_val = non_null.min()
                max_val = non_null.max()
                col_stats.min = str(min(min_val, float(col_stats.min)) if col_stats.min else min_val)
                col_stats.max = str(max(max_val, float(col_stats.max)) if col_stats.max else max_val)
                col_stats.mean_sum += non_null.sum()
                col_stats.mean_count += non_null.shape[0]

            elif pd.api.types.is_datetime64_any_dtype(series):
                min_val = non_null.min()
                max_val = non_null.max()
                col_stats.min = str(min(min_val, pd.to_datetime(col_stats.min)) if col_stats.min else min_val)
                col_stats.max = str(max(max_val, pd.to_datetime(col_stats.max)) if col_stats.max else max_val)

            samples = non_null.astype(str).unique()
            col_stats.samples.update(samples[:sample_size])

    for col, s in stats.items():
        avg = f"{s.mean_sum / s.mean_count:.4f}" if s.mean_count else ""
        sample_val = next(iter(s.samples), "")

        row = {
            "Column": truncate(col, header_widths["Column"]),
            "dtype": (s.dtype or "").ljust(header_widths["dtype"]),
            "Missing": format_percent(s.missing, s.total).rjust(header_widths["Missing"]),
            "Minimum": (s.min or "").rjust(header_widths["Minimum"]),
            "Average": avg.rjust(header_widths["Average"]),
            "Maximum": (s.max or "").rjust(header_widths["Maximum"]),
            "Sample": truncate(sample_val, header_widths["Sample"])
        }

        print("".join(row[h] for h in headers))


def prompt_drop_columns(reader: Iterator[pd.DataFrame]) -> Callable:
    # Preview first chunk to get column names
    first_chunk = next(reader)
    columns = first_chunk.columns.tolist()

    # Ask user which columns to drop
    to_drop = q.checkbox(
        "Select columns to remove:",
        choices=columns
    ).ask()

    if not to_drop:
        print("No columns selected. Skipping column removal.")
        return lambda df: df  # identity function

    print(f"✅ Will drop: {', '.join(to_drop)}")

    # Return transformation function
    def dropper(chunk: pd.DataFrame) -> pd.DataFrame:
        return chunk.drop(columns=to_drop, errors='ignore')

    return dropper


def apply_transformations(chunk: pd.DataFrame, transforms: list[Callable[[pd.DataFrame], pd.DataFrame]]) -> pd.DataFrame:
    for fn in transforms:
        chunk = fn(chunk)
    return chunk

def preview_transformed(reader: Iterator[pd.DataFrame], transforms: list[Callable[[pd.DataFrame], pd.DataFrame]]) -> None:
    row_count = 0
    for chunk in reader:
        transformed = apply_transformations(chunk, transforms)
        preview_full_dataset(iter([transformed]))  # preview just one chunk
        row_count += len(transformed)
    input(f" {row_count:,d} rows previewed. Press Enter to continue...")






# endregion
# region: Main
def cleany(argument: str="") -> None:
    inspect_argument(argument)
    file = find_file(argument)
    if file == "":
        print("Error: No file selected.")
        sys.exit(1)

    # Main loop
    looping = True
    transformations: list[Callable[[pd.DataFrame], pd.DataFrame]] = []
    reader = load_file(file)
    while looping:
        action = q.select("Select an action:", choices=features + ["Exit"]).ask()


        if action == "Exit":
            sys.exit(0)
        elif action == "Preview data (datatypes, sample rows, summary statistics)":
            preview_full_dataset(reader)
            reader = load_file(file)
        elif action == "Remove columns":
            drop_fn = prompt_drop_columns(reader)
            reader = load_file(file)  # Reset reader
            transformations.append(drop_fn)
        







# endregion
if __name__ == "__main__":
    print("This file should not be run directly\n Please use 'cleany' command in the shell or run __main__.py instead.")
    sys.exit(1)

"""
Column          dtype   Missing          Minimum           Average           Maximum            Sample
ProductNumber   str       0.00%            AA001                             ZP800XC          MKL500WP
SKU             str       0.11%           011111                              036110            021626
Length          float    11.82%           0.0100           12.1800           32.0100            6.2200
Width           float    11.82%           0.0100            5.8700           12.0200            3.8000
Height          float    11.82%           0.0100            4.8100            8.1900            1.1000
Weight          float    15.69%           0.0001            0.0300            5.8300            0.0013
Rank            int       6.31%           1.0000        12500.5000        25000.0000          627.0000
LastInvoiceD…   date      6.31%       2001-02-04                          2025-10-03        2025-10-02
"""