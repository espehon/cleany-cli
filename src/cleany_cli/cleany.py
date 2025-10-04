# Copyright (c) 2025 espehon
# MIT License

# region: Imports
# Standard library imports
import sys
import os
from typing import Optional

# Third-party imports
import pandas as pd
import numpy as np

import questionary as q



# endregion
# region: Startup

SUPPORTED_FILE_TYPES = ('.csv', '.xlsx', '.xls', '.json', '.parquet', '.feather', '.xml')








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

def inspect_argument(arg: str) -> str:
    if arg in ('-?', '--help'):
        return 'help'
    elif arg[0] in ('-', '<', '>', ':', '|', '?', '*', '$', '@', '!'):
        return 'invalid'
    else:
        return 'path'


def get_file_list() -> list:
    file_list = [f for f in os.listdir('.') if os.path.isfile(f) and f.lower().endswith(SUPPORTED_FILE_TYPES)]
    return file_list


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


def load_file(filepath: str) -> pd.DataFrame:
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.csv':
        df = pd.read_csv(filepath)
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
    return df
    





# endregion
# region: Main
def cleany(argument: str="") -> None:
    argument_type = inspect_argument(argument)
    if argument_type == 'help':
        help()
        sys.exit(0)
    elif argument_type == 'invalid':
        print("Unrecognized argument. Use -? or --help for usage information.")
    elif argument_type == 'path':
        file = find_file(argument)
        if file == "":
            print("Error: No file selected.")
            sys.exit(1)
        df = load_file(file)
        
    






# endregion
if __name__ == "__main__":
    print("This file should not be run directly\n Please use 'cleany' command in the shell or run __main__.py instead.")
    sys.exit(1)