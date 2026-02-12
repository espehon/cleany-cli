# Copyright (c) 2025 espehon
# MIT License

from __future__ import annotations
from typing import Optional, Iterator, TYPE_CHECKING
import pandas as pd
import questionary as q
from halo import Halo

if TYPE_CHECKING:
    from .transforms import TransformationStack, Transform, DropColumnsTransform, RenameColumnsTransform, OutlierRemovalTransform, NormalizeCurrencyPercentTransform
else:
    # Avoid circular imports at runtime; import inside functions if needed
    pass

spinner = Halo(text='Processing', spinner='dots')


def prompt_drop_columns(reader: Iterator[pd.DataFrame]) -> Optional[DropColumnsTransform]:
    """
    Ask user which columns to drop and return a DropColumnsTransform.
    
    This function CONSUMES the first chunk to get column names, so caller
    must reload the reader after calling this function.
    """
    from .transforms import DropColumnsTransform
    
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

    spinner.succeed(f"Will drop: {', '.join(to_drop)}")
    return DropColumnsTransform(to_drop)


def prompt_rename_columns(reader: Iterator[pd.DataFrame]) -> Optional[RenameColumnsTransform]:
    """
    Ask user which column to rename and what the new name should be.
    Validates the new name and returns a RenameColumnsTransform.
    
    This function CONSUMES the first chunk to get column names, so caller
    must reload the reader after calling this function.
    """
    from .transforms import RenameColumnsTransform
    import re

    try:
        first_chunk = next(reader)
    except StopIteration:
        print("No data to preview. Skipping column rename.")
        return None

    columns = first_chunk.columns.tolist()

    # Ask user which column to rename (single selection)
    col_to_rename = q.select(
        "Select a column to rename:",
        choices=columns
    ).ask()

    if col_to_rename is None or col_to_rename == "":
        print("No column selected. Skipping rename.")
        return None

    # Prompt for new name
    new_name = q.text("Enter new column name:").ask()

    # Validation
    if not new_name or new_name.strip() == "":
        print("Error: New name cannot be empty.")
        return None

    new_name = new_name.strip()

    # Check if new name already exists (in case of other columns)
    if new_name != col_to_rename and new_name in columns:
        print(f"Error: A column named '{new_name}' already exists.")
        return None

    # Check for illegal characters (allow alphanumeric, underscore, space, hyphen, dot)
    if not re.match(r'^[a-zA-Z0-9_\-\.\s]+$', new_name):
        print("Error: Column name contains illegal characters. Allowed: alphanumeric, underscore, hyphen, dot, space.")
        return None

    spinner.succeed(f"Will rename '{col_to_rename}' to '{new_name}'")
    return RenameColumnsTransform(rename_map={col_to_rename: new_name})



def prompt_remove_outliers(reader: Iterator[pd.DataFrame]) -> Optional[OutlierRemovalTransform]:
    """Prompt user to select numeric columns to remove outliers from, and multiplier."""
    from .transforms import OutlierRemovalTransform
    
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

    spinner.succeed(f"Will remove outliers on: {', '.join(to_clean)} with multiplier {mult}")
    return OutlierRemovalTransform(columns=to_clean, multiplier=mult)


def prompt_remove_transforms(stack: TransformationStack) -> Optional[list[Transform]]:
    """Prompt the user to select one or more transforms to remove from the stack.

    Returns a list of removed transforms, or None if nothing was removed.
    """
    if not stack.transforms:
        print("No transformations in the pipeline to remove.")
        return None

    titles = [f"{i+1}. {t.describe()}" for i, t in enumerate(stack.transforms)]
    selected = q.checkbox("Select transforms to remove (checked):", choices=titles).ask()
    if not selected:
        print("No transforms selected. No changes made.")
        return None

    # Parse indices and remove in reverse order to avoid shifting
    try:
        indices = sorted([int(s.split('.', 1)[0]) - 1 for s in selected], reverse=True)
    except Exception:
        print("Could not parse selection. No changes made.")
        return None

    removed: list[Transform] = []
    for idx in indices:
        try:
            removed.append(stack.transforms.pop(idx))
        except Exception:
            continue

    if removed:
        spinner.succeed(f"Removed {len(removed)} transform(s): {', '.join(r.describe() for r in removed)}")
        return removed

    print("No transforms were removed.")
    return None


def prompt_edit_normalize_transform(stack: TransformationStack, filepath: str, detected_dtypes: Optional[dict] = None) -> bool:
    """Allow the user to edit which columns a NormalizeCurrencyPercentTransform applies to.

    Returns True if a transform was updated, False otherwise.
    """
    from .transforms import NormalizeCurrencyPercentTransform
    from .cleany import load_file, detect_currency_percent_columns
    
    # Find normalize transform instances in the stack
    norm_items = [(i, t) for i, t in enumerate(stack.transforms) if isinstance(t, NormalizeCurrencyPercentTransform)]
    if not norm_items:
        print("No NormalizeCurrencyPercentTransform instances found in the pipeline.")
        return False

    # If multiple, let user pick which one to edit
    if len(norm_items) == 1:
        idx, transform = norm_items[0]
    else:
        titles = [f"{i+1}. {t.describe()}" for i, t in norm_items]
        choice = q.select("Select a NormalizeCurrencyPercentTransform to edit:", choices=titles).ask()
        try:
            sel = int(choice.split('.', 1)[0]) - 1
            idx, transform = norm_items[sel]
        except Exception:
            print("Invalid selection. Aborting.")
            return False

    # Always present all file columns as choices. Use the transform's explicit
    # `columns` as the checked defaults when present; otherwise use the
    # auto-detected currency/percent candidates as the default checked set.
    try:
        reader = load_file(filepath, dtype=detected_dtypes or None)
        first = next(reader)
        candidates = first.columns.tolist()
    except Exception:
        print("Could not determine any columns to present for editing.")
        return False

    # Determine which columns should be checked by default
    if transform.columns:
        default_checked = set(transform.columns)
    else:
        try:
            auto = detect_currency_percent_columns(filepath, sample_size=500)
            if detected_dtypes:
                auto = [c for c in auto if not (c in detected_dtypes and detected_dtypes[c] is str)]
            default_checked = set(auto)
        except Exception:
            default_checked = set()

    choices = [q.Choice(c, checked=(c in default_checked)) for c in candidates]
    selected = q.checkbox("Select columns to apply normalization to (checked):", choices=choices).ask()
    if selected is None:
        print("No selection made. Aborting.")
        return False

    # If user selected no explicit columns, interpret as using inferred detection
    if len(selected) == 0:
        transform.columns = None
        spinner.succeed("Normalize transform set to inferred (auto-detect) columns")
    else:
        transform.columns = selected
        spinner.succeed(f"Normalize transform will apply to: {', '.join(selected)}")

    # Update stack (in-place modification already applied)
    stack.transforms[idx] = transform
    return True
