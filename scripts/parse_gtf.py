import pandas as pd
import re

def parse_attr(attr):
    return dict(re.findall(r'(\S+) "([^"]+)"', attr))


def load_exons(gtf):
    rows = []

    with open(gtf) as f:
        for line in f:
            if line.startswith("#"):
                continue

            c = line.strip().split("\t")
            if len(c) != 9:
                continue

            chrom, _, feature, start, end, _, strand, _, attr = c
            if feature != "exon":
                continue

            a = parse_attr(attr)

            tid = a.get("transcript_id")
            gid = a.get("gene_id")

            if tid is None:
                continue

            rows.append({
                "transcript_id": tid,
                "gene_id": gid,
                "chr": chrom,
                "start": int(start),
                "end": int(end),
                "len": int(end) - int(start) + 1,
                "strand": strand
            })

    return pd.DataFrame(rows)
