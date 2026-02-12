from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Iterator, Any
import pandas as pd


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


class RenameColumnsTransform(Transform):
    """Rename one or more columns in the dataframe."""

    def __init__(self, rename_map: dict[str, str]):
        """
        Args:
            rename_map: Dictionary mapping old column names to new column names.
        """
        self.rename_map = rename_map

    def apply(self, chunk: pd.DataFrame) -> pd.DataFrame:
        return chunk.rename(columns=self.rename_map)

    def describe(self) -> str:
        pairs = [f"{old} → {new}" for old, new in self.rename_map.items()]
        return f"Rename columns: {', '.join(pairs)}"


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
