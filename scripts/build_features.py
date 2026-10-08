# scripts/04_build_features.py
import pandas as pd

def load_hsic(path):
    df = pd.read_csv(path, sep="\t")
    df["hsic_flag"] = 1
    return df[["transcript_id", "hsic_score", "hsic_flag"]]


# def build_feature_matrix(struct_df, expr_df, hsic_df):
def build_feature_matrix(struct_df, hsic_df):

    # df = struct_df.merge(expr_df, on="transcript_id", how="left")
    # df = df.merge(hsic_df, on="transcript_id", how="left")
    df = struct_df.merge(hsic_df, on="transcript_id", how="left")
    #print(hsic_df)
    df["hsic_flag"] = df["hsic_flag"].fillna(0)

    #print(df)
    df["hsic_score"] = df["hsic_score"].fillna(0)

#    df["label"] = (df["IF"] > 0.5).astype(int)
#    df["label"] = (df["logFC"] > 1).astype(int)
    threshold = df["hsic_score"].median()
    print(threshold)
    df["label"] = (df["hsic_score"] > threshold).astype(int)

    # ===== MAFS core features =====
    df["struct_novelty"] = (
        df["exon_count"] * 0.3 +
        df["transcript_length"] * 0.0001
#        df["hsic_flag"] * 1.2
    )

    df["splicing_complexity"] = df["exon_count"] / (df["transcript_length"] + 1)

    # df["tumor_specificity"] = abs(df["logFC"])

    # df["mafs_score"] = (
    #     0.4 * df["struct_novelty"] +
    #     0.6 * df["tumor_specificity"]
    # )

    return df
