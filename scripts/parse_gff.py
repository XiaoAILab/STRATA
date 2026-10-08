# ./scripts/parse_gff.py

import pandas as pd
from tqdm import tqdm


def parse_attributes(attr_str):
    attrs = {}
    for item in attr_str.strip().split(";"):
        item = item.strip()
        if item == "":
            continue
        if " " in item:
            key, val = item.split(" ", 1)
            attrs[key] = val.replace('"', '')
    return attrs


def load_gff(gff_file):
    cols = ["chr", "source", "feature", "start", "end",
            "score", "strand", "phase", "attributes"]

    df = pd.read_csv(gff_file, sep="\t", comment="#", names=cols)

    attr_df = df["attributes"].apply(parse_attributes).apply(pd.Series)
    df = pd.concat([df, attr_df], axis=1)

    return df


# ===============================
# CDS length
# ===============================
def compute_cds_length(gff_df):
    cds_df = gff_df[gff_df["feature"] == "CDS"].copy()

    if cds_df.empty:
        return pd.DataFrame(columns=["transcript_id", "cds_length"])

    cds_df["cds_len"] = cds_df["end"] - cds_df["start"] + 1

    return (
        cds_df.groupby("transcript_id")["cds_len"]
        .sum()
        .reset_index()
        .rename(columns={"cds_len": "cds_length"})
    )


# ===============================
# UTR length（严格 strand-aware）
# ===============================
def compute_utr_lengths(gff_df):
    exon_df = gff_df[gff_df["feature"] == "exon"]
    cds_df = gff_df[gff_df["feature"] == "CDS"]

    tx_span = (
        exon_df.groupby("transcript_id")
        .agg(
            tx_start=("start", "min"),
            tx_end=("end", "max"),
            strand=("strand", "first"),
        )
        .reset_index()
    )

    cds_span = (
        cds_df.groupby("transcript_id")
        .agg(
            cds_start=("start", "min"),
            cds_end=("end", "max"),
        )
        .reset_index()
    )

    df = tx_span.merge(cds_span, on="transcript_id", how="left")

    df[["cds_start", "cds_end"]] = df[["cds_start", "cds_end"]].fillna(0)

    def calc(row):
        if row["cds_start"] == 0:
            return 0, 0

        if row["strand"] == "+":
            utr5 = row["cds_start"] - row["tx_start"]
            utr3 = row["tx_end"] - row["cds_end"]
        else:
            utr5 = row["tx_end"] - row["cds_end"]
            utr3 = row["cds_start"] - row["tx_start"]

        return max(0, utr5), max(0, utr3)

    df[["utr5_len", "utr3_len"]] = df.apply(
        lambda r: pd.Series(calc(r)), axis=1
    )

    return df[["transcript_id", "utr5_len", "utr3_len"]]


# ===============================
# NMD distance proxy
# ===============================
def compute_nmd_distance(gff_df):
    exon_df = gff_df[gff_df["feature"] == "exon"]
    cds_df = gff_df[gff_df["feature"] == "CDS"]

    results = []

    for tid, exons in tqdm(exon_df.groupby("transcript_id")):
        exons = exons.sort_values("start")
        strand = exons["strand"].iloc[0]

        cds = cds_df[cds_df["transcript_id"] == tid]

        # 无 CDS 或单 exon → 无 NMD
        if cds.empty or len(exons) < 2:
            results.append((tid, 0))
            continue

        if strand == "+":
            stop = cds["end"].max()
            last_junction = exons.iloc[-2]["end"]
            dist = last_junction - stop
        else:
            stop = cds["start"].min()
            last_junction = exons.iloc[1]["start"]
            dist = stop - last_junction

        results.append((tid, max(0, dist)))

    return pd.DataFrame(results, columns=["transcript_id", "NMD_distance"])


# ===============================
# 主接口
# ===============================
def add_gff_features(df, gff_file):
    print("=== (1) Loading gff ===")
    gff_df = load_gff(gff_file)

    print("=== (2) Computing cds length ===")
    cds_df = compute_cds_length(gff_df)
    print("=== (3) Computing utr length ===")
    utr_df = compute_utr_lengths(gff_df)
    print("=== (4) Computing nmd distance ===")
    nmd_df = compute_nmd_distance(gff_df)

    df = df.merge(cds_df, on="transcript_id", how="left")
    df = df.merge(utr_df, on="transcript_id", how="left")
    df = df.merge(nmd_df, on="transcript_id", how="left")

    df["cds_length"] = df["cds_length"].fillna(0)
    df[["utr5_len", "utr3_len", "NMD_distance"]] = df[
        ["utr5_len", "utr3_len", "NMD_distance"]
    ].fillna(0)

    return df
