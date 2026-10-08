# ./scripts/parse_fasta.py

import pandas as pd
import re
from tqdm import tqdm


# ===============================
# FASTA 读取
# ===============================
def read_fasta(fasta_file):
    seq_dict = {}
    current_id = None
    seq_chunks = []

    with open(fasta_file) as f:
        for line in f:
            line = line.strip()

            if line.startswith(">"):
                if current_id:
                    seq_dict[current_id] = "".join(seq_chunks)

                current_id = line[1:].split()[0]
                seq_chunks = []
            else:
                seq_chunks.append(line)

        if current_id:
            seq_dict[current_id] = "".join(seq_chunks)

    return seq_dict


# ===============================
# GC ratio
# ===============================
def compute_gc_ratio(seq):
    seq = seq.upper()
    if len(seq) == 0:
        return 0
    gc = seq.count("G") + seq.count("C")
    return gc / len(seq)


# ===============================
# motif density
# ===============================
def compute_density(seq, motifs):
 #   seq = seq.upper().replace("T", "U")
    seq = seq.upper()

    total_hits = 0
    for m in motifs:
        total_hits += len(re.findall(m, seq))

    return total_hits / len(seq) if len(seq) > 0 else 0


# ===============================
# 常见 ESE / ESS motif（简化版）
# ===============================

ESE_MOTIFS = [
    "GAAGAA", "GGAGGA", "AAGGAA", "GAAGGA"
]

ESS_MOTIFS = [
    "TTTTC", "TTCTT", "CTTTT", "TCTTT"
]


# ===============================
# 主接口
# ===============================
def add_fasta_features(df, fasta_file):
    seq_dict = read_fasta(fasta_file)

    records = []

    for tid, seq in tqdm(seq_dict.items()):
        gc = compute_gc_ratio(seq)

        ese = compute_density(seq, ESE_MOTIFS)
        ess = compute_density(seq, ESS_MOTIFS)

        records.append({
            "transcript_id": tid,
            "GC_ratio": gc,
            "seq_length": len(seq),
            "ESE_density": ese,
            "ESS_density": ess,
        })

    feat_df = pd.DataFrame(records)

    return df.merge(feat_df, on="transcript_id", how="left")
