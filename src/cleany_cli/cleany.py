# Copyright (c) 2025 espehon
# MIT License

# region: Imports
# Standard library imports
import sys
import os
import random
from typing import Union, Iterator, Optional, Callable, Any
from collections import defaultdict
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

# Third-party imports
import pandas as pd
# import numpy as np

import questionary as q
# from halo import Halo


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
    "Normalize currency/percent columns",
    "Remove columns",
    "Remove outliers (IQR)",
    # "Rename columns",
    # "Remove columns",
    # "Remove duplicate rows",
    # "Handle missing values (drop or fill)",
    # "Filter rows based on column values",
    # "Remove outliers (IQR or Z-score method)",
    # "Save cleaned data to new file or overwrite existing (overwrite is not recommended)",
    # "Can save in different formats (see supported file types above)"
    "Export transformed file",
]


# endregion
# region: Transformation Pipeline

class Transform(ABC):
    """Base class for all transformations. Each transform is a reusable operation."""
    
    @abstractmethod
    def apply(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Apply this transformation to a chunk of data."""
        pass
    
    @abstractmethod
    def describe(self) -> str:
        """Return a human-readable description of this transformation."""
        pass


class DropColumnsTransform(Transform):
    """Remove specific columns from the dataframe."""
    
    def __init__(self, columns: list[str]):
        self.columns = columns
    
    def apply(self, chunk: pd.DataFrame) -> pd.DataFrame:
        return chunk.drop(columns=self.columns, errors='ignore')
    
    def describe(self) -> str:
        return f"Drop columns: {', '.join(self.columns)}"


class NormalizeCurrencyPercentTransform(Transform):
    """Normalize currency and percent strings into numeric floats.

    Behavior:
    - Strips leading `$` and thousands separators (commas).
    - If value ends with `%`, removes `%` and divides by 100.
    - Attempts to coerce the cleaned values to numeric; if at least one
      value coerces successfully, it replaces the column with the numeric
      values (NaN where coercion failed).
    """

    def __init__(self, columns: Optional[list[str]] = None):
        # If columns is None, operate on all object columns where pattern matches
        self.columns = columns

    def apply(self, chunk: pd.DataFrame) -> pd.DataFrame:
        df = chunk.copy()
        candidates = self.columns or df.select_dtypes(include=['object', 'string']).columns.tolist()
        for col in candidates:
            if col not in df.columns:
                continue
            try:
                s = df[col].astype(str).str.strip()
            except Exception:
                continue

            # Quick check if any value looks like currency or percent or numeric-like with commas
            mask_currency = s.str.startswith('$', na=False)
            mask_percent = s.str.endswith('%', na=False)
            mask_commas = s.str.contains(',', na=False)
            mask_numeric_like = s.str.match(r'^-?[0-9\.,]+%?$', na=False)

            if not (mask_currency.any() or mask_percent.any() or mask_commas.any() or mask_numeric_like.any()):
                continue

            cleaned = s.str.replace(r'^\$', '', regex=True).str.replace(',', '')
            pct_mask = cleaned.str.endswith('%', na=False)
            cleaned_num = cleaned.str.rstrip('%')
            cleaned_num = cleaned_num.replace({'nan': None, 'None': None})
            coerced = pd.to_numeric(cleaned_num, errors='coerce')
            if pct_mask.any():
                coerced.loc[pct_mask] = coerced.loc[pct_mask] / 100.0

            # If at least one value converted to numeric, replace the column with coerced
            if coerced.notna().any():
                df[col] = coerced

        return df

    def describe(self) -> str:
        if self.columns:
            return f"Normalize currency/percent in columns: {', '.join(self.columns)}"
        return "Normalize currency/percent in inferred columns"


class OutlierRemovalTransform(Transform):
    """Remove rows with outliers based on per-chunk IQR for specified columns.

    This operates per-chunk (streaming-friendly). For each specified column,
    it computes Q1/Q3 and removes rows outside [Q1 - k*IQR, Q3 + k*IQR].
    """

    def __init__(self, columns: list[str], multiplier: float = 1.5):
        self.columns = columns
        self.multiplier = float(multiplier)

    def apply(self, chunk: pd.DataFrame) -> pd.DataFrame:
        if not self.columns:
            return chunk
        mask = pd.Series(True, index=chunk.index)
        for col in self.columns:
            if col not in chunk.columns:
                continue
            # coerce to numeric; non-convertible values become NaN and are preserved
            num = pd.to_numeric(chunk[col], errors='coerce')
            if num.dropna().empty:
                continue
            q1 = num.quantile(0.25)
            q3 = num.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - self.multiplier * iqr
            upper = q3 + self.multiplier * iqr
            # keep NaNs (they are not outliers here)
            mask &= (num.isna()) | ((num >= lower) & (num <= upper))
        # return filtered chunk
        return chunk.loc[mask]

    def describe(self) -> str:
        return f"Remove outliers (IQR x{self.multiplier}) on: {', '.join(self.columns)}"


class TransformationStack:
    """
    Manages a stack of transformations to be applied to data streams.
    
    Instead of modifying data immediately, we build up a list of transformations
    that get applied each time we stream through the file. This preserves the 
    original file and makes it easy to undo/redo operations.
    """
    
    def __init__(self):
        self.transforms: list[Transform] = []
    
    def add(self, transform: Transform) -> None:
        """Add a transformation to the stack."""
        self.transforms.append(transform)
    
    def remove_last(self) -> Optional[Transform]:
        """Undo the last transformation."""
        if self.transforms:
            return self.transforms.pop()
        return None
    
    def apply_to_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Apply all transformations in order to a single chunk."""
        for transform in self.transforms:
            chunk = transform.apply(chunk)
        return chunk
    
    def apply_to_stream(self, reader: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        """
        Apply all transformations to each chunk in a stream.
        This is the key pattern: transformations are applied chunk-by-chunk,
        keeping memory usage low.
        """
        for chunk in reader:
            yield self.apply_to_chunk(chunk)
    
    def describe(self) -> str:
        """Return a list of all transformations in the stack."""
        if not self.transforms:
            return "No transformations applied"
        lines = ["Active transformations:"]
        for i, transform in enumerate(self.transforms, 1):
            lines.append(f"  {i}. {transform.describe()}")
        return "\n".join(lines)


# endregion
# region: Classes


@dataclass
class ColumnStats:
    dtype: Optional[str] = None
    missing: int = 0
    total: int = 0
    # Store native min/max values (numbers, datetimes, or strings) to allow safe comparisons
    min: Optional[Any] = None
    max: Optional[Any] = None
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
    elif arg and arg[0] in ('-', '<', '>', ':', '|', '?', '*', '$', '@', '!'):
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


def detect_leading_zero_columns(filepath: str, sample_size: int = 1000) -> tuple[dict[str, type], list[str]]:
    """
    Scan a file to detect columns that contain:
    1. Numbers with leading zeros (not decimals) → mark as text (IDs, SKUs, codes)
    2. Dates → return as parse_dates list
    
    Returns: (dtype_dict, parse_dates_list)
    """
    ext = os.path.splitext(filepath)[-1].lower()
    
    # Load just the first sample as strings to preserve leading zeros
    if ext == '.csv':
        df_sample = pd.read_csv(filepath, nrows=sample_size, dtype=str)
    elif ext == '.tsv':
        df_sample = pd.read_csv(filepath, sep='\t', nrows=sample_size, dtype=str)
    elif ext in ('.xlsx', '.xls'):
        df_sample = pd.read_excel(filepath, nrows=sample_size)
        df_sample = df_sample.astype(str)
    elif ext == '.json':
        df_sample = pd.read_json(filepath)
        df_sample = df_sample.astype(str)
    elif ext == '.parquet':
        df_sample = pd.read_parquet(filepath)
        df_sample = df_sample.astype(str)
    elif ext == '.feather':
        df_sample = pd.read_feather(filepath)
        df_sample = df_sample.astype(str)
    elif ext == '.xml':
        df_sample = pd.read_xml(filepath)
        df_sample = df_sample.astype(str)
    else:
        return {}, []
    
    def is_numeric_like(val_str: str) -> bool:
        """Check if a string looks like a number."""
        if not val_str:
            return False
        try:
            float(val_str)
            return True
        except ValueError:
            return False
    
    def is_date_like(val_str: str) -> bool:
        """Check if a string looks like a date (YYYY-MM-DD or similar formats)."""
        if not val_str:
            return False
        try:
            pd.to_datetime(val_str)
            # Check if it actually looks like a date pattern (has dashes or slashes)
            return '-' in val_str or '/' in val_str
        except:
            return False
    
    dtype_dict = {}
    date_cols = []
    
    for col in df_sample.columns:
        # First check for dates
        is_date_col = False
        for val in df_sample[col].dropna():
            val_str = str(val).strip()
            if val_str and is_date_like(val_str):
                date_cols.append(col)
                is_date_col = True
                break
        
        if is_date_col:
            continue  # Skip further checks for this column
        
        # Then check for leading zeros
        for val in df_sample[col].dropna():
            val_str = str(val).strip()
            
            if not val_str:
                continue
            
            # If we find ANY numeric value with leading zero (not decimal), mark column as text
            # e.g., "011111" → text, but "0.5" → number (don't mark)
            if is_numeric_like(val_str) and val_str[0] == '0' and len(val_str) > 1 and val_str[1] != '.':
                dtype_dict[col] = str
                break  # Found one, no need to check rest of column
    
    return dtype_dict, date_cols


def detect_currency_percent_columns(filepath: str, sample_size: int = 1000) -> list[str]:
    """Scan a small sample of the file and return columns that look like
    currency or percent (start with '$', end with '%', contain commas, or
    otherwise look numeric with a percent)."""
    ext = os.path.splitext(filepath)[-1].lower()

    # Load a sample as strings
    if ext == '.csv':
        df_sample = pd.read_csv(filepath, nrows=sample_size, dtype=str)
    elif ext == '.tsv':
        df_sample = pd.read_csv(filepath, sep='\t', nrows=sample_size, dtype=str)
    elif ext in ('.xlsx', '.xls'):
        df_sample = pd.read_excel(filepath, nrows=sample_size)
        df_sample = df_sample.astype(str)
    elif ext in ('.json', '.parquet', '.feather', '.xml'):
        try:
            df_sample = pd.read_json(filepath) if ext == '.json' else pd.read_parquet(filepath) if ext == '.parquet' else pd.read_feather(filepath) if ext == '.feather' else pd.read_xml(filepath)
            df_sample = df_sample.astype(str)
        except Exception:
            return []
    else:
        return []

    candidates: list[str] = []
    for col in df_sample.columns:
        try:
            s = df_sample[col].astype(str).str.strip()
        except Exception:
            continue
        mask_currency = s.str.startswith('$', na=False)
        mask_percent = s.str.endswith('%', na=False)
        mask_commas = s.str.contains(',', na=False)
        # Only treat as currency/percent-like when there is an explicit marker
        # such as leading '$', trailing '%', or thousands separators (commas).
        # Avoid flagging plain numeric columns here to prevent false positives
        # (e.g., SKU or length columns that are numeric-looking but not currency).
        if mask_currency.any() or mask_percent.any() or mask_commas.any():
            candidates.append(col)

    return candidates


def load_file(filepath: str, chunksize: int = 10000, dtype: Optional[dict] = None, parse_dates: Optional[list] = None) -> Iterator[pd.DataFrame]:
    ext = os.path.splitext(filepath)[-1].lower()
    file_size = os.path.getsize(filepath)

    # Chunked loading for large streamable files
    if file_size > FILE_SIZE_LIMIT and ext in STREAMABLE:
        sep = ',' if ext == '.csv' else '\t'
        return pd.read_csv(filepath, sep=sep, chunksize=chunksize, dtype=dtype, parse_dates=parse_dates)

    # Full load + manual chunking
    if ext == '.csv':
        df = pd.read_csv(filepath, dtype=dtype, parse_dates=parse_dates)
    elif ext == '.tsv':
        df = pd.read_csv(filepath, sep='\t', dtype=dtype, parse_dates=parse_dates)
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


def format_percent(number: int, total: int) -> str:
    pct = (number / total) * 100 if total else 0
    return f"{pct:6.2f}%"

def format_number(value: Optional[Any], width: int, dtype_hint: Optional[str] = None) -> str:
    """
    Format a value for display according to dtype_hint.
    - For 'int': render as integer (no .00), right-aligned
    - For 'float': round to 2 decimals, scientific notation if too wide
    - For 'date': format as YYYY-MM-DD (short)
    - For others: truncate and right-align
    """
    if value is None or value == "":
        return "".rjust(width)

    # If already a pandas Timestamp or datetime
    try:
        if dtype_hint == 'date':
            dt = pd.to_datetime(value)
            s = dt.strftime('%Y-%m-%d')
            return s.rjust(width)[:width]
    except Exception:
        pass

    # Explicit string handling: don't try to coerce to numeric
    if dtype_hint == 'str':
        return str(value)[:width].rjust(width)

    # Integers
    if dtype_hint == 'int' or (isinstance(value, (int,))):
        try:
            iv = int(float(value))
            s = str(iv)
            return s.rjust(width)[:width]
        except Exception:
            return str(value)[:width].rjust(width)

    # Floats / numeric
    try:
        num = float(value)
        # If dtype_hint explicitly int, we handled above. For float, format with 2 decimals
        formatted = f"{num:.2f}"
        if len(formatted) <= width:
            return formatted.rjust(width)
        sci = f"{num:.2e}"
        if len(sci) <= width:
            return sci.rjust(width)
        return sci[:width].rjust(width)
    except Exception:
        # Fallback: text
        return str(value)[:width].rjust(width)

def truncate(text, width):
    """Return a truncated string (no padding). Caller should pad to width."""
    if text is None:
        return ""
    text = str(text)
    return text[: width - 1] + '…' if len(text) > width else text

def print_table_row(columns: list[str], widths: dict[str, int], is_header: bool = False, is_bottom: bool = False) -> None:
    """Print a row of the table with box characters and column separators."""
    if is_header:
        # Header: use box-drawing characters
        left = "┌"
        mid = "┬"
        right = "┐"
        sep = "─"
    elif is_bottom:
        # Bottom border
        left = "└"
        mid = "┴"
        right = "┘"
        sep = "─"
    else:
        # Data row separator
        left = "├"
        mid = "┼"
        right = "┤"
        sep = "─"
    
    parts = [left]
    for i, col in enumerate(columns):
        parts.append(sep * widths[col])
        if i < len(columns) - 1:
            parts.append(mid)
    parts.append(right)
    print("".join(parts))

def print_table_content(columns: list[str], values: dict[str, str], widths: dict[str, int]) -> None:
    """Print a data row with vertical separators.

    Pads each column to its configured width. Alignment is column-specific:
    - left for `Column` and `dtype`
    - right for numeric columns and `Sample`
    """
    alignments = {
        "Column": "left",
        "dtype": "left",
        "Missing": "right",
        "Minimum": "right",
        "Average": "right",
        "Maximum": "right",
        "Sample": "right",
    }

    parts = ["│"]
    for col in columns:
        raw = values.get(col, "")
        w = widths[col]
        if alignments.get(col, "left") == "right":
            padded = str(raw)[:w].rjust(w)
        else:
            padded = str(raw)[:w].ljust(w)
        parts.append(padded)
        parts.append("│")
    print("".join(parts))

def preview_full_dataset(data: Iterator[pd.DataFrame], sample_size: int = 1, dtype_hints: Optional[dict] = None) -> None:
    headers = list(header_widths.keys())
    
    # Print top border
    print_table_row(headers, header_widths, is_header=True)
    
    # Print header row
    header_row = {h: h.ljust(header_widths[h]) for h in headers}
    print_table_content(headers, header_row, header_widths)
    
    # Print header separator
    print_table_row(headers, header_widths)

    stats: dict[str, ColumnStats] = defaultdict(ColumnStats)
    # For row-based sampling: collect candidate rows with their completeness scores
    candidate_rows = []  # List of (score, row_index, row_dict) tuples

    for chunk_idx, chunk in enumerate(data):
        # Calculate completeness score for each row in this chunk
        # Score = (non-missing columns) / (total columns)
        # Higher score = healthier row
        for row_idx, (_, row) in enumerate(chunk.iterrows()):
            non_missing = sum(1 for val in row if pd.notna(val) and str(val).strip() != '')
            score = non_missing / len(chunk.columns)
            global_row_idx = chunk_idx * len(chunk) + row_idx
            candidate_rows.append((score, global_row_idx, row.to_dict()))
        
        for col in chunk.columns:
            series = chunk[col]
            col_stats = stats[col]

            if col_stats.dtype is None:
                col_stats.dtype = str(series.dtype)

            # Count missing values: empty strings and nulls should be counted once
            mask_missing = series.isnull() | (series.astype(str).str.strip() == '')
            missing_count = mask_missing.sum()
            col_stats.missing += int(missing_count)
            col_stats.total += len(series)

            # Prepare cleaned series for numeric/date detection:
            # If strings contain leading '$' or trailing '%', normalize them into numeric values
            cleaned_series = series
            numeric_series = None
            try:
                s_str = series.astype(str).str.strip()
            except Exception:
                s_str = series

            # Create cleaned numeric candidate by removing $ and commas, stripping %
            try:
                cleaned = (
                    s_str
                    .str.replace(r'^\$', '', regex=True)
                    .str.replace(',', '')
                )
                # mark percent mask
                mask_pct = cleaned.str.endswith('%', na=False)
                cleaned_numeric = cleaned.str.rstrip('%')
                # convert 'nan' back to actual NaN so to_numeric will coerce
                cleaned_numeric = cleaned_numeric.replace({'nan': None, 'None': None})
                numeric_series = pd.to_numeric(cleaned_numeric, errors='coerce')
                if mask_pct.any():
                    numeric_series.loc[mask_pct] = numeric_series.loc[mask_pct] / 100.0
            except Exception:
                numeric_series = None

            # If the column is forced to str by dtype hints, skip numeric normalization
            if dtype_hints and col in dtype_hints and dtype_hints[col] is str:
                numeric_series = None

            # If the column is numeric-like (either inferred dtype or cleaned numeric has values), use numeric_series
            if pd.api.types.is_numeric_dtype(series) or (numeric_series is not None and numeric_series.notna().any()):
                # Prefer numeric_series if available (handles $ and % normalization)
                use_numeric = numeric_series if numeric_series is not None else pd.to_numeric(series, errors='coerce')
                non_null_num = use_numeric[use_numeric.notna()]
                if not non_null_num.empty:
                    min_val = non_null_num.min()
                    max_val = non_null_num.max()
                    # Store native numeric min/max values
                    if col_stats.min is None:
                        col_stats.min = min_val
                    else:
                        try:
                            col_stats.min = min(min_val, float(col_stats.min))
                        except Exception:
                            try:
                                col_stats.min = min(min_val, col_stats.min)
                            except Exception:
                                col_stats.min = min_val

                    if col_stats.max is None:
                        col_stats.max = max_val
                    else:
                        try:
                            col_stats.max = max(max_val, float(col_stats.max))
                        except Exception:
                            try:
                                col_stats.max = max(max_val, col_stats.max)
                            except Exception:
                                col_stats.max = max_val

                    col_stats.mean_sum += non_null_num.sum()
                    col_stats.mean_count += non_null_num.count()

            elif pd.api.types.is_datetime64_any_dtype(series):
                # datetime handling
                try:
                    dt_series = pd.to_datetime(series, errors='coerce')
                    non_null_dt = dt_series[dt_series.notna()]
                    if not non_null_dt.empty:
                        min_val = non_null_dt.min()
                        max_val = non_null_dt.max()
                        if col_stats.min is None:
                            col_stats.min = min_val
                        else:
                            try:
                                existing = pd.to_datetime(col_stats.min)
                                col_stats.min = min(min_val, existing)
                            except Exception:
                                col_stats.min = min_val

                        if col_stats.max is None:
                            col_stats.max = max_val
                        else:
                            try:
                                existing = pd.to_datetime(col_stats.max)
                                col_stats.max = max(max_val, existing)
                            except Exception:
                                col_stats.max = max_val
                except Exception:
                    pass
            else:
                # String-like column: compute alphabetical min/max
                non_null = series[(series.notna()) & (~series.astype(str).isin(['nan', '', 'NaN']))]
                if not non_null.empty:
                    try:
                        smin = non_null.astype(str).min()
                        smax = non_null.astype(str).max()
                        if col_stats.min is None:
                            col_stats.min = smin
                        else:
                            try:
                                col_stats.min = min(col_stats.min, smin)
                            except Exception:
                                col_stats.min = smin
                        if col_stats.max is None:
                            col_stats.max = smax
                        else:
                            try:
                                col_stats.max = max(col_stats.max, smax)
                            except Exception:
                                col_stats.max = smax
                    except Exception:
                        pass

            # Samples will be populated from row-based selection below
            # (this placeholder ensures col_stats.samples is initialized)
            col_stats.samples = set()

    # Select a random row from rows with above-average or perfect completeness
    sample_row_dict = {}
    if candidate_rows:
        # Calculate average completeness score across all rows
        avg_completeness = sum(score for score, _, _ in candidate_rows) / len(candidate_rows)
        # Select rows that have 100% completeness OR are at/above average
        healthy_pool = [(score, idx, row) for score, idx, row in candidate_rows if score >= avg_completeness or score == 1.0]
        # If no rows meet criteria (shouldn't happen), use all rows
        if not healthy_pool:
            healthy_pool = candidate_rows
        # Randomly select one from the healthy pool
        selected_score, selected_idx, selected_row = random.choice(healthy_pool)
        sample_row_dict = selected_row
    
    for col, s in stats.items():
        avg = f"{s.mean_sum / s.mean_count:.2f}" if s.mean_count else ""
        # Get sample value from the selected row and format according to dtype
        sample_val_raw = sample_row_dict.get(col, "") if sample_row_dict else ""
        
        # Friendly dtype name and hint for formatting
        dtype_raw = (s.dtype or "")

        # Try to detect if this is a date column from dtype_raw
        is_date_col = 'datetime64' in dtype_raw
        
        # Format date-like samples properly
        try:
            if pd.notna(sample_val_raw) and str(sample_val_raw).strip() != '':
                if is_date_col:
                    dt = pd.to_datetime(sample_val_raw, errors='coerce')
                    if pd.notna(dt):
                        sample_val = dt.strftime('%Y-%m-%d')
                    else:
                        sample_val = str(sample_val_raw)
                else:
                    sample_val = str(sample_val_raw)
            else:
                sample_val = ""
        except Exception:
            sample_val = str(sample_val_raw)
        
        dtype_hint = None

        # If user provided dtype hints (e.g., leading-zero columns forced to str), respect them
        if dtype_hints and col in dtype_hints and dtype_hints[col] is str:
            friendly_dtype = 'str'
            dtype_hint = 'str'
        elif s.mean_count > 0:
            # Numeric column (we collected numeric observations)
            try:
                # ensure min/max exist before converting
                if s.min is None or s.max is None:
                    raise ValueError("min/max not available")
                min_f = float(s.min)
                max_f = float(s.max)
                avg_f = s.mean_sum / s.mean_count
                # Check if all values are integers, OR if the original dtype was int64/int32/int16/int8
                is_all_integer = float(min_f).is_integer() and float(max_f).is_integer() and float(avg_f).is_integer()
                is_int_dtype = 'int' in dtype_raw.lower()
                if is_all_integer or is_int_dtype:
                    friendly_dtype = 'int'
                    dtype_hint = 'int'
                else:
                    friendly_dtype = 'float'
                    dtype_hint = 'float'
            except Exception:
                friendly_dtype = 'float'
                dtype_hint = 'float'
        elif 'datetime64' in dtype_raw or (s.min and ('-' in str(s.min) or '/' in str(s.min))):
            friendly_dtype = 'date'
            dtype_hint = 'date'
        else:
            friendly_dtype = 'str'

        # Always show average as float (even for int columns, showing the mean)
        avg_display = avg

        row = {
            "Column": truncate(col, header_widths["Column"]),
            "dtype": friendly_dtype.ljust(header_widths["dtype"]),
            "Missing": format_percent(s.missing, s.total).rjust(header_widths["Missing"]),
            "Minimum": format_number(s.min, header_widths["Minimum"], dtype_hint=dtype_hint),
            "Average": format_number(avg_display, header_widths["Average"], dtype_hint='float'),  # Always render average as float
            "Maximum": format_number(s.max, header_widths["Maximum"], dtype_hint=dtype_hint),
            "Sample": truncate(sample_val, header_widths["Sample"]).rjust(header_widths["Sample"])
        }

        print_table_content(headers, row, header_widths)
    
    # Print bottom border
    print_table_row(headers, header_widths, is_bottom=True)



def prompt_drop_columns(reader: Iterator[pd.DataFrame]) -> Optional[DropColumnsTransform]:
    """
    Ask user which columns to drop and return a DropColumnsTransform.
    
    This function CONSUMES the first chunk to get column names, so caller
    must reload the reader after calling this function.
    """
    try:
        first_chunk = next(reader)
    except StopIteration:
        print("No data to preview. Skipping column removal.")
        return None
    
    columns = first_chunk.columns.tolist()

    # Ask user which columns to drop
    to_drop = q.checkbox(
        "Select columns to remove:",
        choices=columns
    ).ask()

    if not to_drop:
        print("No columns selected. Skipping column removal.")
        return None

    print(f"✅ Will drop: {', '.join(to_drop)}")
    return DropColumnsTransform(to_drop)


def prompt_remove_outliers(reader: Iterator[pd.DataFrame]) -> Optional[OutlierRemovalTransform]:
    """Prompt user to select numeric columns to remove outliers from, and multiplier."""
    try:
        first_chunk = next(reader)
    except StopIteration:
        print("No data to preview. Skipping outlier removal.")
        return None

    # Find numeric columns in the first chunk
    numeric_cols = first_chunk.select_dtypes(include=['number']).columns.tolist()
    if not numeric_cols:
        print("No numeric columns detected in the sample. Skipping outlier removal.")
        return None

    to_clean = q.checkbox("Select numeric columns to remove outliers from:", choices=numeric_cols).ask()
    if not to_clean:
        print("No columns selected. Skipping outlier removal.")
        return None

    multiplier_txt = q.text("IQR multiplier (default 1.5):", default="1.5").ask()
    try:
        mult = float(multiplier_txt)
    except Exception:
        print("Invalid multiplier; using 1.5")
        mult = 1.5

    print(f"✅ Will remove outliers on: {', '.join(to_clean)} with multiplier {mult}")
    return OutlierRemovalTransform(columns=to_clean, multiplier=mult)


def preview_with_transformations(
    filepath: str, 
    stack: TransformationStack,
    sample_size: int = 1,
    dtype: Optional[dict] = None,
    parse_dates: Optional[list] = None
) -> None:
    """
    Preview the data with all transformations applied.
    
    This demonstrates the key pattern: we reload the file and apply
    the transformation stack to each chunk as we stream through.
    """
    reader = load_file(filepath, dtype=dtype, parse_dates=parse_dates)
    transformed_reader = stack.apply_to_stream(reader)
    preview_full_dataset(transformed_reader, sample_size, dtype_hints=dtype)


def export_transformed(
    filepath: str,
    stack: TransformationStack,
    dtype_hints: Optional[dict] = None,
    parse_dates: Optional[list] = None,
) -> None:
    """Export the transformed data to a new file.

    Prompts for output format and filename. Streams CSV/TSV; collects for
    binary formats (Excel, Parquet, Feather).
    """
    # Use the SUPPORTED_FILE_TYPES constant to build the export choices
    choice_labels = list(SUPPORTED_FILE_TYPES)
    pick = q.select("Select export format:", choices=choice_labels).ask()
    if pick is None:
        print("Export cancelled.")
        return
    ext = pick

    base = os.path.splitext(os.path.basename(filepath))[0]
    default_name = f"{base}_clean"
    out_name = q.text("Output file name (without extension):", default=default_name).ask()
    if out_name is None:
        print("Export cancelled.")
        return
    out_name = out_name.strip()

    # If user included a supported extension, strip it off
    for e in SUPPORTED_FILE_TYPES:
        if out_name.lower().endswith(e):
            out_name = out_name[: -len(e)]
            break

    out_path = os.path.join(os.path.dirname(filepath) or '.', out_name + ext)
    if os.path.exists(out_path):
        ok = q.confirm(f"File {out_path} exists. Overwrite?", default=False).ask()
        if not ok:
            print("Export cancelled.")
            return

    reader = load_file(filepath, dtype=dtype_hints, parse_dates=parse_dates)
    transformed_reader = stack.apply_to_stream(reader)

    print(f"Exporting transformed data to {out_path} ...")

    # Stream-friendly formats: CSV, TSV, JSON (newline-delimited)
    if ext in ('.csv', '.tsv'):
        sep = ',' if ext == '.csv' else '\t'
        first = True
        for chunk in transformed_reader:
            if first:
                chunk.to_csv(out_path, sep=sep, index=False, mode='w', header=True)
                first = False
            else:
                chunk.to_csv(out_path, sep=sep, index=False, mode='a', header=False)
        print("Export complete.")
        return

    if ext == '.json':
        # Write newline-delimited JSON
        with open(out_path, 'w', encoding='utf-8') as f:
            for chunk in transformed_reader:
                chunk.to_json(f, orient='records', lines=True)
        print("Export complete.")
        return

    # For binary/batch formats we must collect and then write once
    parts = []
    for chunk in transformed_reader:
        parts.append(chunk)
    if not parts:
        print("No data to export.")
        return
    df = pd.concat(parts, ignore_index=True)

    try:
        if ext == '.xlsx' or ext == '.xls':
            df.to_excel(out_path, index=False)
        elif ext == '.parquet':
            df.to_parquet(out_path, index=False)
        elif ext == '.feather':
            df.to_feather(out_path)
        else:
            # Fallback to CSV
            df.to_csv(out_path, index=False)
        print("Export complete.")
    except Exception as e:
        print(f"Export failed: {e}")






# endregion
# region: Main
def cleany() -> None:
    argument = str(sys.argv[1:])
    inspect_argument(argument)
    file = find_file(argument)
    if file == "":
        print("Error: No file selected.")
        sys.exit(1)

    # Auto-detect dtypes for columns with leading zeros and dates
    print("Scanning file for data type hints...")
    detected_dtypes, date_cols = detect_leading_zero_columns(file)
    if detected_dtypes:
        print(f"Detected text columns (with leading zeros): {list(detected_dtypes.keys())}")
    if date_cols:
        print(f"Detected date columns: {date_cols}")

    # Initialize the transformation stack
    stack = TransformationStack()
    file_size_print(file)

    # --- Startup preview + auto-normalization ---
    # Detect currency/percent-like columns in a small sample. If present,
    # auto-add NormalizeCurrencyPercentTransform to the stack so the initial
    # preview reflects the cleaned view (user can remove it later).
    try:
        currency_cols = detect_currency_percent_columns(file, sample_size=500)
        # Exclude any columns that were explicitly hinted as text (leading zeros)
        if detected_dtypes:
            currency_cols = [c for c in currency_cols if not (c in detected_dtypes and detected_dtypes[c] is str)]

        if currency_cols:
            print(f"Detected currency/percent-like columns: {currency_cols}")
            print("Auto-adding NormalizeCurrencyPercentTransform to pipeline for initial preview.")
            stack.add(NormalizeCurrencyPercentTransform(columns=currency_cols))
            print("You can remove this transform later from the pipeline if desired.")
    except Exception:
        # If detection fails, continue without auto-adding
        pass

    # Run an initial preview so user sees the file immediately
    preview_with_transformations(file, stack, dtype=detected_dtypes or None, parse_dates=date_cols or None)
    # --- end startup preview ---

    # Main loop
    looping = True
    while looping:
        print(f"\n{stack.describe()}")
        action = q.select("Select an action:", choices=features + ["Exit"]).ask()

        if action == "Exit":
            sys.exit(0)
        elif action == features[0]: # Preview data
            preview_with_transformations(file, stack, dtype=detected_dtypes or None, parse_dates=date_cols or None)
        elif action == "Normalize currency/percent columns":
            # Add normalization transform to the stack (applies to future streaming reads)
            # Exclude any columns forced to `str` by earlier detection (leading zeros)
            cols = detect_currency_percent_columns(file, sample_size=500)
            if detected_dtypes:
                cols = [c for c in cols if not (c in detected_dtypes and detected_dtypes[c] is str)]

            confirm = q.confirm("Add NormalizeCurrencyPercentTransform to pipeline? This will convert $ and % values to numeric.", default=True).ask()
            if confirm:
                if cols:
                    stack.add(NormalizeCurrencyPercentTransform(columns=cols))
                    print(f"✅ Added NormalizeCurrencyPercentTransform to the pipeline for columns: {cols}")
                else:
                    # No non-forced columns found; add generic transform
                    stack.add(NormalizeCurrencyPercentTransform())
                    print("✅ Added NormalizeCurrencyPercentTransform to the pipeline (no specific columns detected).")
        elif action == "Remove columns":
            reader = load_file(file, dtype=detected_dtypes or None, parse_dates=date_cols or None)
            transformed_reader = stack.apply_to_stream(reader)
            transform = prompt_drop_columns(transformed_reader)
            if transform:
                stack.add(transform)
        elif action == "Remove outliers (IQR)":
            reader = load_file(file, dtype=detected_dtypes or None, parse_dates=date_cols or None)
            transformed_reader = stack.apply_to_stream(reader)
            transform = prompt_remove_outliers(transformed_reader)
            if transform:
                stack.add(transform)
        elif action == "Export transformed file":
            export_transformed(file, stack, dtype_hints=detected_dtypes or None, parse_dates=date_cols or None)

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