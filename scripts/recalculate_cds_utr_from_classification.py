#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Recalculate CDS/UTR features from a SQANTI3 classification file.

Purpose
-------
This script is intended to be used after add_gff_features().

The input dataframe may already contain:
    - cds_length
    - utr5_len
    - utr3_len

These existing columns are removed first, then recalculated using the
SQANTI3 classification file:

    cds_length = CDS_length
    utr5_len   = CDS_start - 1
    utr3_len   = length - CDS_end

The transcript IDs are matched using:

    df["transcript_id"] <-> classification["isoform"]

Notes
-----
1. The definitions intentionally follow the SQANTI3 classification-based
   calculation used in the reference plotting script.
2. Missing/non-numeric CDS information is retained as NaN.
3. Negative UTR lengths are converted to NaN.
4. The original row order and row count of df are preserved.
"""

import argparse
import numpy as np
import pandas as pd


FEATURE_COLUMNS = ["cds_length", "utr5_len", "utr3_len"]


def read_table_robust(path: str) -> pd.DataFrame:
    """
    Read a tabular file robustly.

    First tries tab-separated parsing. If only one column is obtained,
    falls back to arbitrary whitespace separation.
    """
    try:
        df = pd.read_csv(path, sep="\t", dtype=str)
        df.columns = [c.strip() for c in df.columns]

        if df.shape[1] <= 1:
            raise ValueError("Tab parsing returned only one column.")

        return df

    except Exception:
        df = pd.read_csv(
            path,
            sep=r"\s+",
            engine="python",
            dtype=str,
        )
        df.columns = [c.strip() for c in df.columns]
        return df


def build_cds_utr_table(classification_file: str) -> pd.DataFrame:
    """
    Build transcript-level CDS/UTR features from a SQANTI3 classification file.

    Required columns
    ----------------
    isoform
    length
    CDS_start
    CDS_end
    CDS_length

    Returns
    -------
    pd.DataFrame with columns:
        transcript_id
        cds_length
        utr5_len
        utr3_len
    """
    cls = read_table_robust(classification_file)

    required = {
        "isoform",
        "length",
        "CDS_start",
        "CDS_end",
        "CDS_length",
    }

    missing = sorted(required - set(cls.columns))
    if missing:
        raise ValueError(
            "classification_file is missing required columns: "
            + ", ".join(missing)
        )

    cls = cls.copy()

    # Match the numeric conversion logic used in the reference script.
    for col in ["length", "CDS_start", "CDS_end", "CDS_length"]:
        cls[col] = pd.to_numeric(cls[col], errors="coerce")

    # Keep only the fields needed for recalculation.
    feat = cls[
        [
            "isoform",
            "length",
            "CDS_start",
            "CDS_end",
            "CDS_length",
        ]
    ].copy()

    # SQANTI3 classification-based definitions.
    feat["cds_length"] = feat["CDS_length"]
    feat["utr5_len"] = feat["CDS_start"] - 1
    feat["utr3_len"] = feat["length"] - feat["CDS_end"]

    # Same handling as the reference code:
    # biologically invalid negative UTR values -> NaN.
    feat.loc[feat["utr5_len"] < 0, "utr5_len"] = np.nan
    feat.loc[feat["utr3_len"] < 0, "utr3_len"] = np.nan

    feat = feat.rename(columns={"isoform": "transcript_id"})

    feat = feat[
        [
            "transcript_id",
            "cds_length",
            "utr5_len",
            "utr3_len",
        ]
    ].copy()

    # A classification file should normally contain one row per isoform.
    # Explicitly reject duplicated isoform IDs to avoid accidental row
    # multiplication during merge.
    duplicated = feat["transcript_id"].duplicated(keep=False)
    if duplicated.any():
        examples = (
            feat.loc[duplicated, "transcript_id"]
            .astype(str)
            .drop_duplicates()
            .head(10)
            .tolist()
        )
        raise ValueError(
            "Duplicated isoform IDs were found in classification_file. "
            "Examples: " + ", ".join(examples)
        )

    return feat


def recalculate_cds_utr_features(
    df: pd.DataFrame,
    classification_file: str,
) -> pd.DataFrame:
    """
    Remove previously calculated CDS/UTR features from df and replace them
    with SQANTI3 classification-based values.

    Parameters
    ----------
    df
        DataFrame returned by add_gff_features(), or any dataframe containing
        a 'transcript_id' column.

    classification_file
        SQANTI3 classification file containing:
        isoform, length, CDS_start, CDS_end, CDS_length.

    Returns
    -------
    pd.DataFrame
        Copy of the original dataframe with recalculated:
            cds_length
            utr5_len
            utr3_len

    Important
    ---------
    Missing classification matches remain NaN. They are NOT automatically
    converted to zero, because this function follows the classification-based
    definition directly.
    """
    if "transcript_id" not in df.columns:
        raise ValueError("Input df must contain a 'transcript_id' column.")

    out = df.copy()

    # Preserve original row order exactly.
    helper_col = "__original_row_order__"
    while helper_col in out.columns:
        helper_col = "_" + helper_col

    out[helper_col] = np.arange(len(out))

    # Remove CDS/UTR definitions previously generated by add_gff_features().
    cols_to_drop = [c for c in FEATURE_COLUMNS if c in out.columns]
    if cols_to_drop:
        out = out.drop(columns=cols_to_drop)

    # Build new SQANTI3-derived features.
    feature_df = build_cds_utr_table(classification_file)

    # Left merge preserves all transcripts from the original dataframe.
    out = out.merge(
        feature_df,
        on="transcript_id",
        how="left",
        validate="many_to_one",
    )

    # Restore original order.
    out = (
        out.sort_values(helper_col, kind="stable")
        .drop(columns=helper_col)
        .reset_index(drop=True)
    )

    return out


# Optional alias with a shorter name.
replace_cds_utr_from_classification = recalculate_cds_utr_features


def main():
    """
    Optional command-line interface.

    Example
    -------
    python recalculate_cds_utr_from_classification.py \
        --input features_from_add_gff.tsv \
        --classification_file allCA_classification.txt \
        --output features_recalculated.tsv
    """
    parser = argparse.ArgumentParser(
        description=(
            "Replace cds_length, utr5_len and utr3_len using "
            "SQANTI3 classification-file definitions."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input TSV containing the dataframe previously returned by add_gff_features().",
    )

    parser.add_argument(
        "--classification_file",
        required=True,
        help="SQANTI3 classification file.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output TSV.",
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input, sep="\t")

    out = recalculate_cds_utr_features(
        df=df,
        classification_file=args.classification_file,
    )

    out.to_csv(
        args.output,
        sep="\t",
        index=False,
        na_rep="NA",
    )

    n_total = len(out)
    n_cds = out["cds_length"].notna().sum()
    n_utr5 = out["utr5_len"].notna().sum()
    n_utr3 = out["utr3_len"].notna().sum()

    print(f"[OK] saved: {args.output}")
    print(f"[INFO] total transcripts: {n_total}")
    print(f"[INFO] transcripts with cds_length: {n_cds}")
    print(f"[INFO] transcripts with utr5_len: {n_utr5}")
    print(f"[INFO] transcripts with utr3_len: {n_utr3}")


if __name__ == "__main__":
    main()
