#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Add corrected-transcript 3' end architecture features.

Features:
    APA_distance
    termi3_novelty

Input:
    1. Feature table CSV/TSV
       required columns:
           transcript_id
           gene_id

    2. Corrected transcript GTF
       (IsoQuant/SQANTI3 corrected GTF)

Output:
    CSV with added features

Designed for:
    FSM / ISM / NIC / NNC transcripts

python add_corrected_transcript_apa_features_v2.py \
    --input corrected_id/input_v4.csv \
    --corrected_gtf corrected_id/sqanti3/allCA_corrected.gtf \
    --output corrected_id/input_v4_APA.csv

python add_corrected_transcript_apa_features_v2.py \
    --input new_model_figures/gencodeV49_input2.csv \
    --corrected_gtf /home/lichang/References/gencode.v49.annotation.gtf \
    --output new_model_figures/gencodeV49_input2_APA.csv

"""

import argparse
import pandas as pd
from collections import Counter



def parse_gtf_attributes(attr):

    attrs = {}

    for item in attr.strip().split(";"):

        item = item.strip()

        if not item:
            continue

        if " " in item:

            key, value = item.split(" ", 1)

            attrs[key] = value.replace('"', '')

    return attrs



def load_feature_table(path):

    """
    Automatically detect csv or tsv
    """

    return pd.read_csv(
        path,
        sep=None,
        engine="python"
    )



def extract_corrected_transcript_ends(gtf_file):

    """
    Extract transcript-level 3' end from corrected GTF.

    Return:
        transcript_id
        gene_id
        tx_end
    """

    records = []


    with open(gtf_file) as f:

        for line in f:

            if line.startswith("#"):
                continue


            fields = line.rstrip("\n").split("\t")


            if len(fields) != 9:
                continue


            (
                chrom,
                source,
                feature,
                start,
                end,
                score,
                strand,
                frame,
                attr
            ) = fields


            if feature != "transcript":
                continue


            attrs = parse_gtf_attributes(attr)


            if "transcript_id" not in attrs:
                continue


            transcript_id = attrs["transcript_id"]

            gene_id = attrs.get(
                "gene_id",
                None
            )


            start = int(start)
            end = int(end)


            if strand == "+":

                tx_end = end

            else:

                tx_end = start



            records.append({

                "transcript_id": transcript_id,
                "gene_id": gene_id,
                "tx_end": tx_end

            })


    tx_df = pd.DataFrame(records)


    return tx_df



def clean_transcript_table(tx_df):

    """
    Remove duplicated transcript IDs.

    Check whether one transcript_id
    maps to multiple genes.
    """


    print(
        "Transcript models before cleaning:",
        len(tx_df)
    )


    duplicate_ids = (
        tx_df["transcript_id"]
        .duplicated(keep=False)
    )


    print(
        "Duplicated transcript IDs:",
        duplicate_ids.sum()
    )


    # check transcript assigned to multiple genes

    multi_gene = (
        tx_df
        .groupby("transcript_id")
        ["gene_id"]
        .nunique()
    )


    multi_gene = multi_gene[
        multi_gene > 1
    ]


    print(
        "Transcript IDs assigned to multiple genes:",
        len(multi_gene)
    )


    if len(multi_gene) > 0:

        print(
            "Warning: keeping first occurrence"
        )



    # keep one transcript model

    tx_df = (
        tx_df
        .drop_duplicates(
            subset=[
                "transcript_id"
            ],
            keep="first"
        )
        .copy()
    )


    print(
        "Transcript models after cleaning:",
        len(tx_df)
    )


    return tx_df



def calculate_apa_features(tx_df):

    """
    Calculate:

    APA_distance:
        distance from transcript 3' end
        to predominant gene-level end

    termi3_novelty:
        0 shared end
        1 unique end
    """


    results = []


    for gene_id, group in tx_df.groupby(
        "gene_id"
    ):


        end_counts = Counter(
            group["tx_end"]
        )


        # most frequent transcript end
        reference_end = (
            end_counts
            .most_common(1)[0][0]
        )


        for _, row in group.iterrows():


            tx_end = row["tx_end"]


            results.append({

                "transcript_id":
                    row["transcript_id"],

                "APA_distance":
                    abs(
                        tx_end -
                        reference_end
                    ),

                "termi3_novelty":
                    1
                    if end_counts[tx_end] == 1
                    else 0

            })


    apa_df = pd.DataFrame(results)


    return apa_df



def main():

    parser = argparse.ArgumentParser(
        description=
        "Add APA features from corrected transcript GTF"
    )


    parser.add_argument(
        "--input",
        required=True,
        help="Input feature CSV/TSV"
    )


    parser.add_argument(
        "--corrected_gtf",
        required=True,
        help="Corrected transcript GTF"
    )


    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV"
    )


    args = parser.parse_args()



    print(
        "[1] Loading feature table"
    )


    df = load_feature_table(
        args.input
    )


    required = [
        "transcript_id",
        "gene_id"
    ]


    for col in required:

        if col not in df.columns:

            raise ValueError(
                f"Missing required column: {col}"
            )



    input_rows = len(df)



    print(
        "[2] Extracting corrected transcript ends"
    )


    tx_df = extract_corrected_transcript_ends(
        args.corrected_gtf
    )



    tx_df = clean_transcript_table(
        tx_df
    )



    print(
        "[3] Calculating APA features"
    )


    apa_df = calculate_apa_features(
        tx_df
    )


    print(
        "APA feature rows:",
        len(apa_df)
    )



    print(
        "[4] Checking uniqueness before merge"
    )


    assert (
        apa_df["transcript_id"]
        .duplicated()
        .sum()
        == 0
    ), (
        "APA table contains duplicated transcript IDs"
    )



    print(
        "[5] Merging features"
    )


    df = df.merge(
        apa_df,
        on="transcript_id",
        how="left",
        validate="one_to_one"
    )



    output_rows = len(df)


    if output_rows != input_rows:

        raise RuntimeError(
            f"Row number changed after merge: "
            f"{input_rows} -> {output_rows}"
        )



    missing = (
        df["APA_distance"]
        .isna()
        .sum()
    )


    print(
        "Missing APA annotations:",
        missing
    )


    df["APA_distance"] = (
        df["APA_distance"]
        .fillna(0)
    )


    df["termi3_novelty"] = (
        df["termi3_novelty"]
        .fillna(0)
        .astype(int)
    )



    print(
        "[6] Saving output"
    )


    df.to_csv(
        args.output,
        sep="\t",
        index=False
    )


    print(
        "Finished:",
        args.output
    )



if __name__ == "__main__":

    main()