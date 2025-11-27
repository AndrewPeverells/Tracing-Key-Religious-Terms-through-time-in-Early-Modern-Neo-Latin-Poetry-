"""
1. Extract KWIC windows around target lemmas from lemmatised texts.
2. Compute collocations (logDice) per target × time_bin × region_bin.
3. Enrich metadata with religion and add religion_group to collocates.
4. Compute global diachronic drift for each target.
5. Define semantic core (clusters + mean logDice thresholds).
6. Compute per-religion drift with adaptive thresholds.
7. Combine time-only and religion-specific drift into final CSVs.
8. Export summary Excel files and provide plotting/inspection helpers.
"""

import math
import re
from collections import Counter
from pathlib import Path

import fasttext
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openpyxl import Workbook
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import umap

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

BASE = Path("/mnt/c/exp")

METADATA_CSV = BASE / "metadata.csv"
LEMM_DIR     = BASE / "lemmatised"

TARGETS      = ["gratia", "fides"]
TARGETS_SET  = set(t.lower() for t in TARGETS)

WINDOW_SIZE      = 7
EXPERIMENT_LABEL = "gratia_fides"

KWIC_CSV        = BASE / f"windows_w{WINDOW_SIZE}_{EXPERIMENT_LABEL}.csv"
COLLOCATES_CSV  = BASE / f"collocates_w{WINDOW_SIZE}_{EXPERIMENT_LABEL}.csv"

STOPWORDS_FILE  = BASE / "stopwords_lat_lemmas.txt"
FASTTEXT_MODEL  = BASE / "cc.la.300.bin"

KWIC_ENCODING       = "utf-8"
METADATA_ENCODING   = "utf-8-sig"
COLLOCATES_ENCODING = "utf-8-sig"

TOKEN_SPLIT_RE = re.compile(r"\s+")


# =============================================================================
# 1. KWIC EXTRACTION (WINDOWS AROUND TARGET LEMMAS)
# =============================================================================

"""
KWIC extraction:
- Read metadata and corresponding lemmatised text files.
- For each target lemma, extract ±WINDOW_SIZE (end of verse) token windows.
- Output one row per hit: metadata + target + token_index + left_context + keyword + right_context.
"""


def read_tokens(path: Path):
    with path.open("r", encoding=KWIC_ENCODING) as f:
        text = f.read()
    tokens = TOKEN_SPLIT_RE.split(text.strip())
    return [t for t in tokens if t]


def extract_kwic_windows():
    print(f"[KWIC] Loading metadata from: {METADATA_CSV}")
    df_meta = pd.read_csv(METADATA_CSV, encoding=METADATA_ENCODING)

    filename_col = df_meta.columns[0]
    print(f"[KWIC] Filename column detected as: {filename_col}")

    all_rows = []

    for idx, row in df_meta.iterrows():
        xml_name = str(row[filename_col])
        stem = re.sub(r"\.\w+$", "", xml_name)
        txt_name = stem + ".txt"
        txt_path = LEMM_DIR / txt_name

        if not txt_path.is_file():
            print(f"[KWIC] WARNING: Lemma file not found for {xml_name} -> {txt_name}")
            continue

        tokens = read_tokens(txt_path)
        if not tokens:
            continue

        tokens_lower = [t.lower() for t in tokens]
        meta_dict = row.to_dict()

        for i, tok in enumerate(tokens_lower):
            if tok not in TARGETS_SET:
                continue

            start = max(0, i - WINDOW_SIZE)
            end   = min(len(tokens), i + WINDOW_SIZE + 1)

            left_tokens  = tokens[start:i]
            right_tokens = tokens[i+1:end]

            left_context  = " ".join(left_tokens)
            right_context = " ".join(right_tokens)
            keyword       = tokens[i]

            out_row = {
                **meta_dict,
                "target": keyword.lower(),
                "token_index": i,
                "left_context": left_context,
                "keyword": keyword,
                "right_context": right_context,
            }
            all_rows.append(out_row)

        if (idx + 1) % 50 == 0:
            print(f"[KWIC] Processed {idx+1} / {len(df_meta)} metadata rows…")

    if not all_rows:
        print("[KWIC] No target occurrences found. Check TARGETS, paths, and encodings.")
        return

    df_out = pd.DataFrame(all_rows)
    print(f"[KWIC] Total windows extracted: {len(df_out)}")
    print(f"[KWIC] Writing to: {KWIC_CSV}")
    df_out.to_csv(KWIC_CSV, index=False, encoding="utf-8-sig")
    print("[KWIC] Done.")


# =============================================================================
# 2. COLLOCATE COMPUTATION (logDice PER SLICE)
# =============================================================================

"""
Collocate computation:
- Read KWIC windows.
- Tokenise left/right context.
- Count collocate frequencies per target × time_bin × region_bin.
- Compute logDice association scores, excluding stopwords and the target itself.
- Output collocates table.
"""

with STOPWORDS_FILE.open("r", encoding="utf-8") as f:
    STOPWORDS = {line.strip() for line in f if line.strip()}


def tokenize_context(left_context, right_context):
    left = [] if pd.isna(left_context) else TOKEN_SPLIT_RE.split(str(left_context).strip())
    right = [] if pd.isna(right_context) else TOKEN_SPLIT_RE.split(str(right_context).strip())
    tokens = [t for t in left + right if t]
    return tokens


def compute_collocates():
    print(f"[COLL] Loading KWIC windows from: {KWIC_CSV}")
    df = pd.read_csv(KWIC_CSV, encoding="utf-8-sig")

    title_col = df.columns[0]  # first metadata column = file name / title
    print(f"[COLL] Using title column: {title_col}")

    for col in ["target", "left_context", "right_context", "time_bin", "region_bin"]:
        if col not in df.columns:
            raise ValueError(f"[COLL] Missing required column: {col}")

    records = []
    group_cols = ["target", "time_bin", "region_bin"]

    for keys, df_slice in df.groupby(group_cols):
        target, time_bin, region_bin = keys
        freq_target = len(df_slice)
        if freq_target == 0:
            continue

        colloc_counter = Counter()
        n_texts = df_slice[title_col].nunique()

        for _, row in df_slice.iterrows():
            ctx_tokens = tokenize_context(row["left_context"], row["right_context"])
            for tok in ctx_tokens:
                t = tok.lower()
                if not t:
                    continue
                if t == target:
                    continue
                if t in STOPWORDS:
                    continue
                colloc_counter[t] += 1

        if not colloc_counter:
            continue

        for colloc, f_wc in colloc_counter.items():
            f_c = f_wc
            logdice = 14 + math.log2((2 * f_wc) / (freq_target + f_c))

            records.append({
                "target": target,
                "time_bin": time_bin,
                "region_bin": region_bin,
                "collocate": colloc,
                "freq_target": freq_target,
                "freq_collocate": f_c,
                "logDice": logdice,
                "window_size": WINDOW_SIZE,
                "n_texts_in_slice": n_texts,
            })

    if not records:
        print("[COLL] No collocations found. Check input file and columns.")
        return

    df_out = pd.DataFrame(records).sort_values(
        by=["target", "time_bin", "region_bin", "logDice"],
        ascending=[True, True, True, False]
    )

    print(f"[COLL] Total collocate rows: {len(df_out)}")
    print(f"[COLL] Writing collocation table to: {COLLOCATES_CSV}")
    df_out.to_csv(COLLOCATES_CSV, index=False, encoding="utf-8-sig")
    print("[COLL] Done.")


# =============================================================================
# 3. METADATA ENRICHMENT WITH RELIGION (METADATA + COLLOCATES)
# =============================================================================

"""
Metadata enrichment:
- Read metadata and custom city→religion mapping.
- Add a 'religion' column to metadata.
- Join slice-level 'places' and 'religions' onto the collocates table.
- Derive 'religion_group' (Catholic / Lutheran / Calvinist / Mixed / Unknown).
"""


def add_religion_to_metadata(mapping_path: Path):
    df = pd.read_csv(METADATA_CSV, encoding=METADATA_ENCODING)

    religion_map = {}
    with mapping_path.open("r", encoding="utf-8") as f:
        for line in f:
            if "-" in line:
                city, religion = line.split("-", 1)
                religion_map[city.strip()] = religion.strip()

    df["religion"] = df["place_of_publication"].map(religion_map)
    print("[REL] Unmatched cities:")
    print(df[df["religion"].isna()]["place_of_publication"].unique())

    df.to_csv(METADATA_CSV, index=False, encoding=METADATA_ENCODING)
    print("[REL] Updated metadata.csv with 'religion' column.")


def enrich_collocates_with_places_and_religions():
    coll = pd.read_csv(COLLOCATES_CSV, encoding=COLLOCATES_ENCODING)
    meta = pd.read_csv(METADATA_CSV, encoding=METADATA_ENCODING)

    for col in ["time_bin", "region_bin", "place_of_publication", "religion"]:
        if col not in meta.columns:
            raise ValueError(f"[REL] Column '{col}' not found in metadata.csv")

    slice_info = (
        meta
        .groupby(["time_bin", "region_bin"], dropna=False)
        .agg(
            places=("place_of_publication",
                    lambda x: "; ".join(sorted({str(v) for v in x.dropna()}))),
            religions=("religion",
                       lambda x: "; ".join(sorted({str(v) for v in x.dropna()}))),
        )
        .reset_index()
    )

    coll_enriched = coll.merge(slice_info, on=["time_bin", "region_bin"], how="left")
    coll_enriched.to_csv(COLLOCATES_CSV, index=False, encoding=COLLOCATES_ENCODING)
    print("[REL] Updated collocates CSV with 'places' and 'religions' columns.")


def normalise_religion_cell(s):
    """
    Normalise raw 'religions' string to one of:
    - 'Catholic'
    - 'Protestant - Lutheran'
    - 'Protestant - Calvinist'
    - 'Mixed'
    - 'Unknown'
    """
    if pd.isna(s):
        return "Unknown"

    s = str(s).strip()
    if not s:
        return "Unknown"

    labels = [p.strip() for p in s.split(";") if p.strip()]
    atomic = set(labels)

    if len(atomic) > 1:
        return "Mixed"

    label = list(atomic)[0]
    low = label.lower()
    if "catholic" in low:
        return "Catholic"
    if "lutheran" in low:
        return "Protestant - Lutheran"
    if "calvinist" in low or "reformed" in low:
        return "Protestant - Calvinist"
    if "mixed" in low:
        return "Mixed"
    return label


def add_religion_group_column():
    df = pd.read_csv(COLLOCATES_CSV, encoding=COLLOCATES_ENCODING)
    df["religion_group"] = df["religions"].apply(normalise_religion_cell)
    print("[REL] religion_group distribution:")
    print(df["religion_group"].value_counts(dropna=False))
    df.to_csv(COLLOCATES_CSV, index=False, encoding=COLLOCATES_ENCODING)


# =============================================================================
# 4. GLOBAL DIACHRONIC DRIFT (TIME ONLY)
# =============================================================================

"""
Global drift (time-only):
- Compute diachronic logDice trajectories for collocates of a given target.
- Filter by:
  - minimum number of time slices
  - minimum mean logDice
  - minimum total frequency
"""


def shared_collocates_filtered(
    df,
    target,
    min_slices=4,
    min_mean_logdice=8.5,
    min_total_freq=8,
):
    """
    Compute diachronic collocate drift for a target, with stability filters.
    Output includes:
      collocate, time_bins, logdice_values, n_slices, max/min/mean_logdice,
      total_freq, first, last, delta, abs_delta, trend.
    """
    sub = df[df["target"] == target].copy()

    sub_mean = (
        sub.groupby(["collocate", "time_bin"], as_index=False)
           .agg(logDice=("logDice", "mean"))
    )

    time_order = {tb: i for i, tb in enumerate(sorted(sub_mean["time_bin"].unique()))}
    sub_mean["time_index"] = sub_mean["time_bin"].map(time_order)

    sub_sorted = sub_mean.sort_values(["collocate", "time_index"])

    grouped = (
        sub_sorted.groupby("collocate")
                  .agg(
                      time_bins=("time_bin", list),
                      logdice_values=("logDice", list),
                      n_slices=("time_bin", "nunique"),
                      max_logdice=("logDice", "max"),
                      min_logdice=("logDice", "min"),
                      mean_logdice=("logDice", "mean")
                  )
                  .reset_index()
    )

    total_freq = df.groupby("collocate")["freq_collocate"].sum()
    grouped["total_freq"] = grouped["collocate"].map(total_freq)

    filtered = grouped[
        (grouped["n_slices"] >= min_slices) &
        (grouped["mean_logdice"] >= min_mean_logdice) &
        (grouped["total_freq"] >= min_total_freq)
    ].copy()

    filtered[["first", "last"]] = filtered["logdice_values"].apply(
        lambda lst: pd.Series([lst[0], lst[-1]])
    )

    filtered["delta"] = filtered["last"] - filtered["first"]
    filtered["abs_delta"] = filtered["delta"].abs()

    def _trend(x):
        if x > 0: return "↑"
        if x < 0: return "↓"
        return "="

    filtered["trend"] = filtered["delta"].apply(_trend)

    return filtered.sort_values("abs_delta", ascending=False)


def semantic_drift_top_collocates(
    df,
    target,
    min_slices=4,
    min_total_freq=20,
    logdice_quantile=0.9,
):
    """
    Drift top collocates via:
      1) ≥ min_slices time bins
      2) total_freq ≥ min_total_freq
    """
    sub = df[df["target"] == target].copy()

    sub_mean = (
        sub.groupby(["collocate", "time_bin"], as_index=False)
           .agg(
               logDice=("logDice", "mean"),
               freq_collocate=("freq_collocate", "sum")
           )
    )

    time_order = {tb: i for i, tb in enumerate(sorted(sub_mean["time_bin"].unique()))}
    sub_mean["time_index"] = sub_mean["time_bin"].map(time_order)

    sub_sorted = sub_mean.sort_values(["collocate", "time_index"])

    grouped = (
        sub_sorted.groupby("collocate")
                  .agg(
                      time_bins=("time_bin", list),
                      logdice_values=("logDice", list),
                      n_slices=("time_bin", "nunique"),
                      max_logdice=("logDice", "max"),
                      min_logdice=("logDice", "min"),
                      mean_logdice=("logDice", "mean"),
                      total_freq=("freq_collocate", "sum"),
                  )
                  .reset_index()
    )

    cutoff = grouped["max_logdice"].quantile(logdice_quantile)

    filtered = grouped[
        (grouped["n_slices"] >= min_slices) &
        (grouped["total_freq"] >= min_total_freq) &
        (grouped["max_logdice"] >= cutoff)
    ].copy()

    filtered[["first", "last"]] = filtered["logdice_values"].apply(
        lambda lst: pd.Series([lst[0], lst[-1]])
    )

    filtered["delta"] = filtered["last"] - filtered["first"]
    filtered["abs_delta"] = filtered["delta"].abs()

    def _trend(d):
        if d > 0: return "↑"
        if d < 0: return "↓"
        return "="

    filtered["trend"] = filtered["delta"].apply(_trend)
    filtered = filtered.sort_values("abs_delta", ascending=False)
    filtered.attrs["logdice_cutoff"] = cutoff
    return filtered


def global_drift_overview(df, delta_threshold=2.0, top_n=30):
    """
    Produce simple ranked views of the drift table:
      - top increases
      - top decreases
      - top volatile (|delta|)
      - threshold_filtered (abs_delta >= delta_threshold)
    """
    up = df.sort_values("delta", ascending=False)
    down = df.sort_values("delta", ascending=True)
    volatile = df.sort_values("abs_delta", ascending=False)
    filtered = df[df["abs_delta"] >= delta_threshold].sort_values(
        "abs_delta", ascending=False
    )

    return {
        "top_increases": up[["collocate", "delta", "abs_delta", "trend", "first", "last"]].head(top_n),
        "top_decreases": down[["collocate", "delta", "abs_delta", "trend", "first", "last"]].head(top_n),
        "top_volatile": volatile[["collocate", "delta", "abs_delta", "trend", "first", "last"]].head(top_n),
        "threshold_filtered": filtered[["collocate", "delta", "abs_delta", "trend", "first", "last"]],
    }


def make_drift_csvs(shared_df, target, clusters, out_prefix=None):
    """
    Write CSVs for:
      - strong increasing
      - strong decreasing
      - stable collocates
    restricted to a semantic core defined by `clusters` and mean_logdice >= 9.
    """
    if out_prefix is None:
        out_prefix = target

    core_mask = (
        shared_df["cluster"].isin(clusters) &
        (shared_df["mean_logdice"] >= 9.0)
    )
    core = shared_df.loc[core_mask].copy()

    core["drift_class"] = pd.cut(
        core["abs_delta"],
        bins=[0, 1.0, 2.0, core["abs_delta"].max()],
        labels=["stable", "mild", "strong"]
    )

    core_strong_inc = (
        core[(core["drift_class"] == "strong") & (core["delta"] > 0)]
        .sort_values("delta", ascending=False)
    )
    core_strong_dec = (
        core[(core["drift_class"] == "strong") & (core["delta"] < 0)]
        .sort_values("delta", ascending=True)
    )
    core_stable = (
        core[core["drift_class"] == "stable"]
        .sort_values("mean_logdice", ascending=False)
    )

    core_strong_inc.to_csv(
        BASE / f"{out_prefix}_increasing.csv",
        index=False, encoding="utf-8-sig"
    )
    core_strong_dec.to_csv(
        BASE / f"{out_prefix}_decreasing.csv",
        index=False, encoding="utf-8-sig"
    )
    core_stable.to_csv(
        BASE / f"{out_prefix}_stable.csv",
        index=False, encoding="utf-8-sig"
    )


# =============================================================================
# 5. RELIGION-SPECIFIC DRIFT
# =============================================================================

"""
Religion drift (adaptive thresholds):
- For each target and religion_group, compute logDice trajectories.
- Adaptive thresholds depend on global frequency with this target:
    global_freq >= 200 → min_slices=1, min_total_freq=5
    50 ≤ global_freq < 200 → min_slices=2, min_total_freq=10
    global_freq < 50 → base_min_slices, base_min_total_freq
- Output dict: {religion_group: DataFrame}
"""


def collocate_drift_by_religion(
    df,
    target,
    base_min_slices=4,
    base_min_total_freq=15,
):
    results = {}

    sub_all = df[df["target"] == target]
    global_freq = (
        sub_all.groupby("collocate")["freq_collocate"]
               .sum()
    )

    def choose_thresholds(freq):
        if pd.isna(freq):
            return base_min_slices, base_min_total_freq
        if freq >= 200:
            return 1, 5
        elif freq >= 50:
            return 2, 10
        else:
            return base_min_slices, base_min_total_freq

    for rel in sorted(df["religion_group"].dropna().unique()):
        sub = sub_all[sub_all["religion_group"] == rel]
        if sub.empty:
            continue

        sub_mean = (
            sub.groupby(["collocate", "time_bin"], as_index=False)
               .agg(
                   logDice=("logDice", "mean"),
                   freq=("freq_collocate", "sum"),
               )
        )

        time_order = {tb: i for i, tb in enumerate(sorted(sub_mean["time_bin"].unique()))}
        sub_mean["time_index"] = sub_mean["time_bin"].map(time_order)
        sub_mean = sub_mean.sort_values(["collocate", "time_index"])

        grouped = (
            sub_mean.groupby("collocate")
                    .agg(
                        time_bins=("time_bin", list),
                        logdice_values=("logDice", list),
                        n_slices=("time_bin", "nunique"),
                        total_freq=("freq", "sum"),
                    )
                    .reset_index()
        )

        grouped["global_freq"] = grouped["collocate"].map(global_freq)

        def row_passes(row):
            ms, mf = choose_thresholds(row["global_freq"])
            return (row["n_slices"] >= ms) and (row["total_freq"] >= mf)

        filtered = grouped[grouped.apply(row_passes, axis=1)].copy()
        if filtered.empty:
            continue

        filtered["first"] = filtered["logdice_values"].apply(lambda xs: xs[0])
        filtered["last"]  = filtered["logdice_values"].apply(lambda xs: xs[-1])
        filtered["delta"] = filtered["last"] - filtered["first"]
        filtered["abs_delta"] = filtered["delta"].abs()
        filtered["religion_group"] = rel
        filtered = filtered.sort_values("abs_delta", ascending=False)

        results[rel] = filtered

    return results


def combine_religion_drifts(drift_dict):
    """
    Combine per-religion drift tables into a single DataFrame.
    """
    rows = []
    for rel, df_rel in drift_dict.items():
        tmp = df_rel.copy()
        tmp["religion_group"] = rel
        rows.append(tmp)
    return pd.concat(rows, ignore_index=True)


def pivot_religion_trends(df_combined):
    """
    From combined religion drift table, build:
      - deltas: collocate × religion_group → delta
      - trends: collocate × religion_group → trend (derived from delta)
    """
    df_combined = df_combined.copy()

    if "trend" not in df_combined.columns:
        def _trend(d):
            if d > 0: return "↑"
            elif d < 0: return "↓"
            else: return "="
        df_combined["trend"] = df_combined["delta"].apply(_trend)

    deltas = df_combined.pivot_table(
        index="collocate",
        columns="religion_group",
        values="delta"
    )

    trends = df_combined.pivot_table(
        index="collocate",
        columns="religion_group",
        values="trend",
        aggfunc=lambda x: x.iloc[0]
    )

    return deltas, trends


def classify_cross_religion(trends_row):
    """
    Qualitative pattern across religions based on trend symbols.
    """
    unique = set(trends_row.dropna())

    if unique == {"↑"}:
        return "shared_increase"
    if unique == {"↓"}:
        return "shared_decrease"
    if unique == {"="}:
        return "stable_all"
    if "↑" in unique and "↓" in unique:
        return "opposite_directions"
    if "↑" in unique and "=" in unique:
        return "rise_in_some"
    if "↓" in unique and "=" in unique:
        return "decline_in_some"
    return "complex"


# =============================================================================
# 6. COMBINED TIME + RELIGION DRIFT, CSV/EXCEL EXPORTS
# =============================================================================

"""
Integration of global time-only drift and religion-specific drift:
- Merge global drift metrics onto religion trends.
- Compute delta_range across religions.
- Classify combined patterns (global vs confessional directions).
- Write final CSVs:
    - fides_gratia_results.csv
    - gratia_fides_religion_patterns.csv
"""

REL_COLS = ["Catholic", "Protestant - Lutheran", "Protestant - Calvinist", "Mixed"]


def compute_delta_range_partial(deltas):
    """
    Compute per-collocate delta_range using only available (non-NaN) religions.
    """
    ranges = {}
    for coll in deltas.index:
        row = deltas.loc[coll].dropna()
        if len(row) < 2:
            ranges[coll] = None
        else:
            ranges[coll] = row.max() - row.min()
    return ranges


def merge_general_and_religion(general_drift, religion_trends, religion_deltas):
    """
    Merge:
      - general time drift (global)
      - religion trends table
      - delta ranges across religions
    into a collocate-indexed DataFrame with trend_general / delta_general / abs_delta_general.
    """
    out = religion_trends.copy()

    out = out.merge(
        general_drift[["collocate", "delta", "abs_delta"]],
        on="collocate",
        how="left"
    )
    out = out.rename(columns={
        "delta": "delta_general",
        "abs_delta": "abs_delta_general",
    })

    if "trend" in general_drift.columns:
        out = out.merge(
            general_drift[["collocate", "trend"]],
            on="collocate",
            how="left",
            suffixes=("", "_general_tmp")
        )
        out = out.rename(columns={"trend": "trend_general"})
    else:
        def _trend(d):
            if d > 0: return "↑"
            elif d < 0: return "↓"
            else: return "="
        out["trend_general"] = out["delta_general"].apply(_trend)

    partial_ranges = compute_delta_range_partial(religion_deltas)
    out["delta_range"] = out.index.map(partial_ranges)

    out["mismatch"] = "consistent"
    return out


def combined_time_religion_pattern(row):
    """
    Combined pattern classifier:
    - global up/down/stable vs confessional up/down/stable.
    """
    g = row["trend_general"]
    rels = {k: row.get(k) for k in REL_COLS if k in row.index}
    rels = {k: v for k, v in rels.items() if isinstance(v, str)}

    if g == "↑" and set(rels.values()) == {"↑"}:
        return "global_up_shared_up"
    if g == "↓" and set(rels.values()) == {"↓"}:
        return "global_down_shared_down"
    if g == "↑" and "↓" in rels.values():
        return "global_up_religion_down"
    if g == "↓" and "↑" in rels.values():
        return "global_down_religion_up"
    if g == "=" and len(set(rels.values())) > 1:
        return "global_stable_religion_divergent"
    if g in {"↑", "↓"} and set(rels.values()) == {"="}:
        return "global_shift_religion_stable"
    return "complex"


def inspect_religion_patterns(df_rel, words):
    """
    Helper: extract time + religion pattern rows for selected collocates.
    """
    cols = [
        "collocate",
        "trend_general",
        "delta_general",
        "Catholic",
        "Protestant - Lutheran",
        "Protestant - Calvinist",
        "Mixed",
        "combined_pattern",
    ]
    sub = df_rel[df_rel["collocate"].isin(words)].copy()
    cols = [c for c in cols if c in sub.columns]
    return sub[cols]


def export_group_patterns_to_excel():
    """
    Build Excel file with 6 sheets summarising pre-defined groups of collocates
    (gratia/fides × increasing/decreasing/stable).
    """
    df_gratia_rel = pd.read_csv(BASE / "gratia_time_religion_all.csv")
    df_fides_rel  = pd.read_csv(BASE / "fides_time_religion_all.csv")

    gratia_increasing = [
        "canticum", "dea", "amo", "pallas", "musa", "mitis", "anima", "sors",
        "caelum", "sacrum", "salus", "felix", "flos", "frons", "certus", "uotum",
        "castus", "lex", "decus", "credo", "beatus", "tutus", "amor", "decet",
        "consto", "decor", "filius", "paeniteo", "doceo lux", "gloria", "pietas",
    ]
    gratia_decreasing = [
        "faustus", "uerbum", "columba", "saeculum", "vita", "poeta", "mater",
        "pater", "pax", "ars", "christus", "benignus", "caelestis", "mens",
        "spiritus",
    ]
    gratia_stable = [
        "deus", "ops", "fides", "pius", "dignus", "laus", "meritum", "superus",
        "miser", "dominus", "spes", "donum", "sacer", "cura", "poena",
        "licet", "sanctus", "aeternus", "mysterium", "fateor", "diuinus", "culpa",
    ]

    fides_increasing = [
        "caelum", "iuro", "decus", "tutus", "aeternus", "dubius", "apostolus",
        "princeps", "credo", "foedus", "pater", "animus",
    ]
    fides_decreasing = [
        "christus", "constantia", "constans", "sacrum", "probo", "sincerus",
        "mirus", "pariter", "salus", "virtus", "iustitia", "verbum", "opus",
        "sanctus",
    ]
    fides_stable = [
        "castus", "concordia", "dominus", "candor", "bene", "mereo", "gloria",
        "regnum", "laus", "gratia", "mater", "fatum",
    ]

    groups = [
        ("gratia_increasing", df_gratia_rel, gratia_increasing),
        ("gratia_decreasing", df_gratia_rel, gratia_decreasing),
        ("gratia_stable",     df_gratia_rel, gratia_stable),
        ("fides_increasing",  df_fides_rel,  fides_increasing),
        ("fides_decreasing",  df_fides_rel,  fides_decreasing),
        ("fides_stable",      df_fides_rel,  fides_stable),
    ]

    results = {}
    for sheet_name, df_rel, word_list in groups:
        word_list_unique = list(dict.fromkeys(word_list))
        results[sheet_name] = inspect_religion_patterns(df_rel, word_list_unique)

    wb = Workbook()
    default_sheet = wb.active
    wb.remove(default_sheet)

    for sheet_name, df_out in results.items():
        ws = wb.create_sheet(title=sheet_name[:31])
        ws.append(list(df_out.columns))
        for row in df_out.itertuples(index=False, name=None):
            ws.append(list(row))

    output_path = BASE / "gratia_fides_religion_patterns.xlsx"
    wb.save(str(output_path))
    print("[REL] Saved Excel:", output_path)


# =============================================================================
# 7. VISUALISATION (TIME + RELIGION PLOTS)
# =============================================================================

"""
Visualisation helpers:
- plot_many_collocates_religion_drift:
    small multiples (one subplot per collocate) with lines per religion.
"""


REL_COLOR_MAP = {
    "Catholic":               "#d62728",
    "Protestant - Lutheran":  "#1f77b4",
    "Protestant - Calvinist": "#2ca02c",
    "Mixed":                  "#F97306",
}


def plot_many_collocates_religion_drift(
    drift_dict,
    collocates,
    title_prefix="fides",
    figsize_per_plot=(7, 4),
    ncols=2,
    lw=2.0,
    ms=8,
    fontsize=13,
):
    """
    Plot semantic drift by religion for multiple collocates.
    drift_dict: {religion_group: df} from collocate_drift_by_religion
    collocates: list of collocate strings
    """
    all_tb = set()
    for df_rel in drift_dict.values():
        for tb_list in df_rel["time_bins"]:
            all_tb.update(tb_list)
    TIME_ORDER = sorted(all_tb)
    TIME_INDEX = {tb: i for i, tb in enumerate(TIME_ORDER)}

    n = len(collocates)
    nrows = math.ceil(n / ncols)
    fig_width = figsize_per_plot[0] * ncols
    fig_height = figsize_per_plot[1] * nrows

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(fig_width, fig_height),
        sharex=False, sharey=False
    )
    axes = axes.flatten()

    active_religions = set()

    for ax, coll in zip(axes, collocates):
        has_any = False

        for rel, df_rel in drift_dict.items():
            row = df_rel[df_rel["collocate"] == coll]
            if row.empty:
                continue

            time_bins = row["time_bins"].iloc[0]
            logd_vals = row["logdice_values"].iloc[0]
            x = [TIME_INDEX[t] for t in time_bins]

            color = REL_COLOR_MAP.get(rel, "black")
            ax.plot(x, logd_vals, marker="o", ms=ms, lw=lw,
                    color=color, label=rel)

            active_religions.add(rel)
            has_any = True

        ax.set_title(coll, fontsize=fontsize+2)
        ax.tick_params(axis="both", labelsize=fontsize)

        if has_any:
            ax.set_xticks(range(len(TIME_ORDER)))
            ax.set_xticklabels(TIME_ORDER, rotation=45,
                               ha="right", fontsize=fontsize)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(0.5, 0.5, "no data",
                    ha="center", va="center", fontsize=fontsize)

    for ax in axes[n:]:
        ax.set_visible(False)

    handles = [
        plt.Line2D([0], [0], color=REL_COLOR_MAP[r], lw=lw)
        for r in active_religions
    ]
    fig.legend(
        handles,
        list(active_religions),
        title="Religion",
        loc="upper center",
        ncol=2,
        fontsize=fontsize+4,
        title_fontsize=fontsize+6,
        frameon=True,
        borderpad=1.2,
        labelspacing=1.2,
        handlelength=2.5,
    )

    plt.tight_layout(rect=[0.05, 0.08, 0.95, 0.88])
    fig.suptitle(f"{title_prefix}: semantic drift by religion",
                 fontsize=fontsize+4, y=0.98)
    plt.show()