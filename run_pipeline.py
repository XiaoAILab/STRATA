# run_pipeline.py

import pandas as pd
import torch

from scripts.parse_gtf import load_exons
from scripts.build_structure import build_structure
from scripts.build_features import load_hsic, build_feature_matrix

from scripts.parse_gff import add_gff_features
from scripts.parse_fasta import add_fasta_features

from scripts.add_corrected_transcript_apa_features_v2 import (
    load_feature_table,
    extract_corrected_transcript_ends,
    clean_transcript_table,
    calculate_apa_features,
)

from scripts.recalculate_cds_utr_from_classification import recalculate_cds_utr_features

from scripts.strata_length_baseline_benchmark_v8 import run_benchmark

import time


# =========================
# main pipeline
# =========================
if __name__ == "__main__":


    work_dir = "./corrected_id/"

    print("Start loading load_exons...")
    exon = load_exons(work_dir + "sqanti3/allCA_corrected.gtf")

    print("Start building structures...")
    tumor_struct_1 = build_structure(exon)
    tumor_struct_1.to_csv(work_dir + "input1_v4.csv", sep="\t", index=False)

    # tumor_struct_1 = pd.read_csv(work_dir + "input1_v4.csv", sep="\t")

    print("Start adding gff3 features...")
    tumor_struct_2 = add_gff_features(tumor_struct_1, work_dir + "sqanti3/allCA_corrected.cds.gff3")

    print("Start adding fasta features...")
    tumor_struct = add_fasta_features(tumor_struct_2, work_dir + "sqanti3/allCA_corrected.fasta")
    tumor_struct.to_csv(work_dir + "input2_v4.csv", sep="\t", index=False)

    # tumor_struct = pd.read_csv(work_dir + "input2_v4.csv", sep="\t")

    # print("Start adding hsic scores...")
    # hsic = load_hsic(work_dir + "discovered_transcript_grouped_tpm.hsic_score.tsv")

    print("Start building feature matrix...")
    df = build_feature_matrix(tumor_struct, hsic)
    df.to_csv(work_dir + "input_v4.csv", sep="\t", index=False)


    print("Start adding 3' features...")
    input_file = work_dir + "input_v4.csv"
    corrected_gtf = work_dir + "sqanti3/allCA_corrected.gtf"
    output_file = work_dir + "input_v4_APA.csv"

    df = load_feature_table(input_file)
    input_rows = len(df)

    for col in ["transcript_id", "gene_id"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    tx_df = extract_corrected_transcript_ends(corrected_gtf)
    tx_df = clean_transcript_table(tx_df)
    apa_df = calculate_apa_features(tx_df)

    assert not apa_df["transcript_id"].duplicated().any()

    df = df.merge(
        apa_df,
        on="transcript_id",
        how="left",
        validate="one_to_one"
    )

    if len(df) != input_rows:
        raise RuntimeError("Row number changed after merge")

    print("Missing APA annotations:", df["APA_distance"].isna().sum())

    df["APA_distance"] = df["APA_distance"].fillna(0)
    df["termi3_novelty"] = df["termi3_novelty"].fillna(0).astype(int)

    df.to_csv(output_file, sep="\t", index=False)

    print("APA features saved:", output_file)

    # df = pd.read_csv(work_dir + "input_v4_APA.csv", sep="\t")

    print(len(df))
    df = recalculate_cds_utr_features(
        df,
        classification_file=work_dir + "sqanti3/allCA_classification.txt"
    )


    df.to_csv(work_dir + "input_v4_APA_recalUTR.csv", sep="\t", index=False)

    df_dropped = df.drop(columns=['hsic_score'])
    df_dropped = df_dropped.drop(columns=['liverCA_hsic_score'])
    df_dropped = df_dropped.drop(columns=['colorectalCA_hsic_score'])
    df_dropped = df_dropped.drop(columns=['lungCA_hsic_score'])
    df_dropped = df_dropped.drop(columns=['inLiverCA'])
    df_dropped = df_dropped.drop(columns=['incolorectalCA'])
    df_dropped = df_dropped.drop(columns=['inLungCA'])
    
    
    # score_cols = "liverCA_hsic_score"
    # score_cols = "colorectalCA_hsic_score"
    # score_cols = "lungCA_hsic_score"

    score_cols = "hsic_score"
    
    hsic_df = pd.read_csv(work_dir + "discovered_transcript_grouped_tpm.hsic_score_new_standarized.tsv", sep="\t")
    hsic_score = hsic_df
    
    # hsic_score = hsic_score[hsic_score["inLiverCA"] == 1].copy()
    # hsic_score = hsic_score[hsic_score["incolorectalCA"] == 1].copy()
    # hsic_score = hsic_score[hsic_score["inLungCA"] == 1].copy()

    df = df_dropped.merge(hsic_score, on="transcript_id", how="inner")
    
    print(len(df))

    features = [
        "transcript_length",
        "exon_count",
        "utr5_len",
        "utr3_len",
        "exon_len_std",
        # "exon_len_mean",
        "num_microexons",
        "intron_proxy",
        "cds_length",
        "GC_ratio",
        "NMD_distance",
        "APA_distance",
        "termi3_novelty",
#        "ESE_density",
#        "ESS_density",
#        "exon_density",
#        "avg_exon_len",
#        "logFC",
#        "IF_mean",
   #     "normal_mean",
#        "parent_gene_isoform_count",
    #    "Normal_IF_variance",
#        "exon_len_cv",
#        "struct_novelty",
#        "splicing_complexity"
    ]

    # print(df[features])

    print("Start training...")

    benchmark = run_benchmark(
        df=df,
        features=features,
        output_dir="./strata_length_benchmark_All_v8",

        group_col="gene_id",
        label_col=score_cols,

        # Figure 4D export
        model_label="All_standarized",
        figure4d_out="./figure4D_All_input_v8.tsv",
        # figure4d_extra_cols=("transcript_id", "gene_id"),

        n_splits=3,
        seed=111111,
        n_boot=1000,
        ci=0.95,
        top_fractions=(0.01, 0.05, 0.10),

        include_xgb=True,

        strata_model_kwargs=dict(
            d_model=32,
            nhead=4,
            num_layers=2,
            dropout=0.15,
        ),

        lr=3e-3,
        weight_decay=1e-4,
        max_epochs=500,
        patience=40,
        num_pairs=8192,
        tie_eps=1e-4,
        margin=0.0,
        mse_weight=0.1,

        device=torch.device("cuda:1"),

        verbose=True,
    )
    

