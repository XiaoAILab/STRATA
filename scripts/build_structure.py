import pandas as pd

def build_structure(exon_df):

    exon_df["is_microexon"] = exon_df["len"] <= 27

    g = exon_df.groupby("transcript_id")

    df = g.agg(
        exon_count=("len", "count"),
        exon_len_mean=("len", "mean"),
        exon_len_std=("len", "std"),
        num_microexons=("is_microexon", "sum"),
        gene_id=("gene_id", "first"),
        chr=("chr", "first"),
        strand=("strand", "first"),
        tx_start=("start", "min"),
        tx_end=("end", "max"),
    ).reset_index()

    # transcript_length = genomic span (NOT sum of exons)
    df["genomic_span"] = df["tx_end"] - df["tx_start"] + 1
    df["transcript_length"] = df["exon_count"] * df["exon_len_mean"]

    # exon density
    df["exon_density"] = df["exon_count"] / ((df["tx_end"] - df["tx_start"] + 1) + 1)
    
    # exon length CV
    df["exon_len_cv"] = df["exon_len_std"] / (df["exon_len_mean"] + 1e-6)

    # intron proxy length
    df["intron_proxy"] = df["genomic_span"] - df["transcript_length"]

    return df
