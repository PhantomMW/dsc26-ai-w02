"""Data audit for Rawaj Pay transfer extracts.

Allowed dependencies: numpy, pandas, scikit-learn (and the standard library).
"""
from __future__ import annotations

import pandas as pd
from sklearn.metrics import roc_auc_score


def audit(df: pd.DataFrame, *, target: str, id_col: str, time_col: str) -> dict:
    """Audit a transfer extract before anyone trains a model on it."""

    # Required columns must exist.
    for column in (target, id_col, time_col):
        if column not in df.columns:
            raise KeyError(column)

    # NaN, None, empty strings, and whitespace-only strings are missing.
    def missing_mask(series: pd.Series) -> pd.Series:
        mask = series.isna()
        blank_strings = series.map(
            lambda value: isinstance(value, str) and not value.strip()
        )
        return mask | blank_strings

    missing_masks = {
        column: missing_mask(df[column])
        for column in df.columns
    }

    n_rows = len(df)

    missing = {
        column: float(mask.mean()) if n_rows else 0.0
        for column, mask in missing_masks.items()
    }

    # Find duplicated, non-missing IDs.
    id_missing = missing_masks[id_col]
    repeated_ids = df.loc[~id_missing, id_col]
    repeated_ids = repeated_ids.loc[
        repeated_ids.duplicated(keep=False)
    ]
    duplicate_ids = sorted(repeated_ids.drop_duplicates().tolist())

    # Normalize recognized missing values before counting duplicate rows.
    normalized = df.copy()

    for column, mask in missing_masks.items():
        if mask.any():
            normalized[column] = normalized[column].astype(object)
            normalized.loc[mask, column] = None

    duplicate_rows = int(normalized.duplicated(keep="first").sum())

    # Constant columns ignore missing values; exclude the target.
    constant_columns = []

    for column in df.columns:
        if column == target:
            continue

        non_missing = df.loc[~missing_masks[column], column]

        if non_missing.nunique(dropna=True) <= 1:
            constant_columns.append(column)

    constant_columns.sort()

    # Label statistics.
    target_missing = missing_masks[target]
    labelled = df.loc[~target_missing]
    labelled_target = labelled[target]

    unlabelled_rows = int(target_missing.sum())

    positive_rate = (
        float((labelled_target == 1).mean())
        if len(labelled) > 0
        else 0.0
    )

    # Parse non-missing timestamps and count invalid values.
    time_missing = missing_masks[time_col]

    parsed_times = pd.to_datetime(
        df.loc[~time_missing, time_col],
        errors="coerce",
    )

    bad_timestamps = int(parsed_times.isna().sum())
    valid_times = parsed_times.dropna()

    if valid_times.empty:
        time_range = [None, None]
    else:
        time_range = [
            pd.Timestamp(valid_times.min()).isoformat(),
            pd.Timestamp(valid_times.max()).isoformat(),
        ]

    # Detect possible target leakage using value and missingness AUC.
    leakage_suspects = []

    if len(labelled) >= 20 and labelled_target.nunique(dropna=True) == 2:
        y = labelled_target.astype(int).to_numpy()

        for column in df.columns:
            if column in (target, id_col, time_col):
                continue

            series = labelled[column]
            column_missing = missing_masks[column].loc[
                ~target_missing
            ].to_numpy()

            # Skip columns with one distinct value across the full dataset.
            # Recognized missing values have already been normalized.
            if normalized[column].nunique(dropna=False) <= 1:
                continue

            suspect = False

            # Check whether missingness predicts the label.
            if len(set(column_missing.tolist())) > 1:
                auc_missing = roc_auc_score(
                    y,
                    column_missing.astype(int),
                )

                if auc_missing >= 0.98 or auc_missing <= 0.02:
                    suspect = True

            # Check numeric/boolean values using labelled, non-missing rows.
            if (
                not suspect
                and (
                    pd.api.types.is_numeric_dtype(series.dtype)
                    or pd.api.types.is_bool_dtype(series.dtype)
                )
            ):
                usable = ~column_missing

                if int(usable.sum()) >= 20:
                    y_usable = y[usable]

                    if len(set(y_usable.tolist())) == 2:
                        values = series.loc[usable].to_numpy()

                        auc_value = roc_auc_score(
                            y_usable,
                            values,
                        )

                        if auc_value >= 0.98 or auc_value <= 0.02:
                            suspect = True

            if suspect:
                leakage_suspects.append(column)

    return {
        "n_rows": int(n_rows),
        "missing": missing,
        "duplicate_ids": duplicate_ids,
        "duplicate_rows": duplicate_rows,
        "constant_columns": constant_columns,
        "unlabelled_rows": unlabelled_rows,
        "positive_rate": positive_rate,
        "bad_timestamps": bad_timestamps,
        "time_range": time_range,
        "leakage_suspects": sorted(leakage_suspects),
    }