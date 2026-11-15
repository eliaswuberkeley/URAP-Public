"""Rebuild aggregate charts and researcher-facing summaries from frozen CSVs."""
from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "runtime/matplotlib"))
sys.path.insert(0, str(ROOT / "src"))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from storage import atomic_bytes, read_csv, sha, write_csv, write_json  # noqa: E402

COLORS = ["#197A72", "#CB526B", "#577EBB", "#8D729C"]
GROUPS = [("Black", "Female"), ("White", "Female"), ("Black", "Male"), ("White", "Male")]


def values(rows, key, race=None, gender=None):
    return np.array([float(r[key]) for r in rows if (race is None or r["source_race"] == race)
                     and (gender is None or r["source_gender"] == gender)])


def smd(rows, key, denominator, gender=None):
    b, w = values(rows, key, "Black", gender), values(rows, key, "White", gender)
    return float((b.mean() - w.mean()) / max(denominator, 1e-8))


def grouped(categories, series, ylabel, xlabel="", **kwargs):
    return {"kind": "grouped_bar", "categories": categories, "series": series,
            "ylabel": ylabel, "xlabel": xlabel, **kwargs}


def count_series(rows, field, categories):
    result = []
    for i, (race, gender) in enumerate(GROUPS):
        counts = Counter(r[field] for r in rows if r["source_race"] == race and r["source_gender"] == gender)
        result.append({"label": race + " " + gender, "values": [counts[k] for k in categories], "color": COLORS[i]})
    return result


def render(item):
    panels = item["panels"]
    fig, axes = plt.subplots(1, len(panels), figsize=(12, 4.6), squeeze=False)
    for ax, panel in zip(axes[0], panels, strict=True):
        categories, series = panel["categories"], panel["series"]
        x = np.arange(len(categories))
        width = .78 / len(series)
        for j, s in enumerate(series):
            pos = x + (j - (len(series) - 1) / 2) * width
            if panel["kind"] == "horizontal_bar":
                bars = ax.barh(pos, s["values"], height=width, color=s["color"], label=s["label"])
            else:
                bars = ax.bar(pos, s["values"], width=width, color=s["color"], label=s["label"])
            if panel.get("labels"):
                ax.bar_label(bars, fmt="%.0f", padding=3, fontsize=10)
        if panel["kind"] == "horizontal_bar":
            ax.set_yticks(x, categories)
            ax.invert_yaxis()
            ax.set_xlabel(panel["xlabel"])
            ax.grid(axis="x", alpha=.18)
            if panel.get("limit"):
                ax.set_xlim(0, panel["limit"])
        else:
            ax.set_xticks(x, categories)
            ax.set_ylabel(panel["ylabel"])
            ax.set_xlabel(panel["xlabel"])
            ax.grid(axis="y", alpha=.18)
            ax.set_ylim(0, max(max(s["values"], default=0) for s in series) * 1.2 + .1)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        if panel.get("title"):
            ax.set_title(panel["title"], fontsize=12)
        if len(series) > 1:
            ax.legend(frameon=False, fontsize=10, ncol=2, loc="upper right")
    fig.tight_layout(pad=1.7)
    for ext in ("png", "pdf"):
        dest = ROOT / f"plots/{item['id']}.{ext}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        temp = dest.with_suffix(".part")
        fig.savefig(temp, format=ext, dpi=200, facecolor="white")
        os.replace(temp, dest)
    plt.close(fig)


def main():
    ready = json.loads((ROOT / "data/DATASET_READY.json").read_text())
    for name, checksum in ready["files"].items():
        if sha(ROOT / name) != checksum:
            raise ValueError("Dataset artifact changed: " + name)
    rows = read_csv(ROOT / "data/selected_images.csv")
    pool = read_csv(ROOT / "data/matching_candidate_pool.csv")
    audit = read_csv(ROOT / "data/candidate_audit.csv")
    inventory = read_csv(ROOT / "data/source_inventory.csv")
    pairs = read_csv(ROOT / "data/matched_pairs.csv")
    cfg = json.loads((ROOT / "config/curation.json").read_text())
    report = json.loads((ROOT / "data/selection_report.json").read_text())
    screening = Counter(r["screening_status"] for r in audit)
    selection = Counter(r["split"] for r in rows)
    count_table = [{"source_race": race, "source_gender": gender, "split": split,
                    "images": sum(r["source_race"] == race and r["source_gender"] == gender and r["split"] == split for r in rows)}
                   for race, gender in GROUPS for split in ("train", "development", "test")]
    write_csv(ROOT / "analysis/group_split_counts.csv", count_table)
    quality = []
    features = [("contrast", "Contrast"), ("log_edge_energy", "Log image detail"),
                ("expression_confidence", "Expression confidence"), ("brightness", "Brightness"),
                ("clipped_dark_fraction", "Dark clipping"), ("clipped_light_fraction", "Light clipping")]
    for gender in ["all", *cfg["genders"]]:
        g = None if gender == "all" else gender
        for key, label in features:
            b, w = values(pool, key, "Black", g), values(pool, key, "White", g)
            sd = float(np.sqrt((b.var(ddof=1) + w.var(ddof=1)) / 2))
            for name, data in [("candidate_pool", pool), ("selected", rows)]:
                quality.append({"population": name, "source_gender": gender, "metric": key,
                                "label": label, "black_mean": float(values(data, key, "Black", g).mean()),
                                "white_mean": float(values(data, key, "White", g).mean()),
                                "reference_pooled_sd": sd, "signed_smd_black_minus_white": smd(data, key, sd, g),
                                "absolute_smd": abs(smd(data, key, sd, g)),
                                "used_in_matching": key in cfg["quality_features"] or key == "expression_confidence"})
    write_csv(ROOT / "analysis/quality_balance.csv", quality)
    dist = []
    for field in ("age_band", "expression_class", "expression_label"):
        for race, gender in GROUPS:
            sub = [r for r in rows if r["source_race"] == race and r["source_gender"] == gender]
            for level, n in sorted(Counter(r[field] for r in sub).items()):
                dist.append({"field": field, "level": level, "source_race": race,
                             "source_gender": gender, "images": n, "percent": 100 * n / len(sub)})
    write_csv(ROOT / "analysis/age_expression_counts.csv", dist)
    uncertain = sum(r["expression_class"] == "uncertain" for r in rows)
    pstats = []
    for k in ("confidence_abs_difference", "contrast_difference_sd", "log_edge_energy_difference_sd", "match_cost"):
        v = np.abs([float(p[k]) for p in pairs])
        pstats.append({"metric": k, "median": float(np.median(v)), "p95": float(np.quantile(v, .95)), "maximum": float(max(v))})
    write_csv(ROOT / "analysis/pair_distance_summary.csv", pstats)
    sizes = {r["image_id"]: r for r in read_csv(ROOT / "data/image_file_sizes.csv")}
    if set(sizes) != {r["image_id"] for r in rows}:
        raise ValueError("Image-size manifest membership changed")
    for row in rows:
        if sizes[row["image_id"]]["image_sha256"] != row["image_sha256"]:
            raise ValueError("Image-size manifest hash mismatch")
    size = sum(int(sizes[r["image_id"]]["bytes"]) for r in rows)
    summary = {**report, "screening_counts": dict(screening), "images_by_split": dict(selection),
               "source_inventory_counts": dict(Counter(r["inventory_status"] for r in inventory)),
               "uncertain_expression_images": uncertain, "uncertain_expression_percent": 100 * uncertain / len(rows),
               "selected_image_bytes": size, "historical_selected_overlap": 0,
               "no_detected_duplicate_images": True, "person_disjointness_verified": False,
               "human_review": False, "api_cost_usd": 0, "gpu_rental_cost_usd": 0}
    write_json(ROOT / "analysis/summary.json", summary)
    items = []
    def add(ident, title, finding, panels, sources):
        items.append({"id": ident, "title": title, "finding": finding, "panels": panels,
                      "sources": sources, "plot": f"plots/{ident}.png"})
    flow = [len(audit), screening["eligible"], len(pool), len(rows)]
    add("01_sample_selection", "FairFace matched sample",
        f"{len(pairs):,} Black-White pairs contain {len(rows):,} distinct images. Selection uses image attributes, not model bias scores.",
        [grouped(["Screened", "Passed screening", "Matching pool", "Selected"],
                 [{"label": "Images", "values": flow, "color": COLORS[0]}], "Images", labels=True)],
        ["data/candidate_audit.csv", "data/matching_candidate_pool.csv", "data/selected_images.csv"])
    split_series = [{"label": s.capitalize(), "values": [next(r["images"] for r in count_table if
                       (r["source_race"], r["source_gender"], r["split"]) == (race, gender, s)) for race, gender in GROUPS],
                     "color": COLORS[i]} for i, s in enumerate(("train", "development", "test"))]
    add("02_groups_and_splits", "Race, gender and research splits",
        f"All four groups have equal counts in each split: {selection['train']:,} training, {selection['development']:,} development and {selection['test']:,} test images overall.",
        [grouped([r + "\n" + g for r, g in GROUPS], split_series, "Images", labels=True)],
        ["analysis/group_split_counts.csv"])
    ages = cfg["age_bands"]
    add("03_age_balance", "Matched age coverage",
        "Age-band counts are identical across race/gender groups. Coverage is limited by the least available matching strata; this is not a population-representative sample.",
        [grouped(ages, count_series(rows, "age_band", ages), "Images per group", "Publisher age band")],
        ["analysis/age_expression_counts.csv", "data/common_support_quotas.csv"])
    expressions = ["neutral", "positive", "negative", "uncertain"]
    add("04_expression_balance", "Matched expression estimates",
        f"Expression counts match across groups, including detailed labels within each class. {100 * uncertain / len(rows):.1f}% of images remain uncertain; estimates are not human-verified emotions.",
        [grouped([e.capitalize() for e in expressions], count_series(rows, "expression_class", expressions), "Images per group")],
        ["analysis/age_expression_counts.csv"])
    quality_series = [{"label": label, "values": [next(r["absolute_smd"] for r in quality if r["population"] == pop and
                         r["source_gender"] == "all" and r["metric"] == key) for key, _ in features], "color": COLORS[i]}
                      for i, (pop, label) in enumerate((("candidate_pool", "Before matching"), ("selected", "Selected")))]
    add("05_residual_quality_balance", "Measured quality and remaining differences",
        "Bars show absolute Black-White mean differences in candidate standard-deviation units; closer to zero means closer averages. Brightness and clipping are reported, not forced equal.",
        [{"kind": "horizontal_bar", "categories": [label for _, label in features], "series": quality_series,
          "xlabel": "Absolute standardized mean difference", "ylabel": ""}],
        ["analysis/quality_balance.csv"])
    conf_edges = np.linspace(0, .15, 6)
    detail_edges = np.linspace(0, 2, 6)
    panels = []
    for key, edges, title in [("confidence_abs_difference", conf_edges, "Expression confidence difference"),
                              ("contrast_difference_sd", detail_edges, "Contrast difference (SD)")]:
        counts, _ = np.histogram(np.abs([float(p[key]) for p in pairs]), bins=edges)
        digits = 2 if key == "confidence_abs_difference" else 1
        labels = [f"{a:.{digits}f}-{b:.{digits}f}" for a, b in zip(edges[:-1], edges[1:], strict=True)]
        panels.append(grouped(labels, [{"label": "Pairs", "values": counts.tolist(), "color": COLORS[0]}],
                              "Matched pairs", title=title))
    add("06_pair_match_distances", "Pair-level matching distances",
        "Each pair shares age band, gender and estimated expression label. Confidence differences are at most 0.15; contrast and log-detail differences are each at most 2 candidate SDs.",
        panels, ["data/matched_pairs.csv", "analysis/pair_distance_summary.csv"])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    for item in items:
        render(item)
    write_json(ROOT / "slides/slide_manifest.json", items)
    selected_quality = {r["metric"]: r["absolute_smd"] for r in quality if r["population"] == "selected" and r["source_gender"] == "all"}
    text = f"""# October 6 FairFace curation results

## Outcome

- Selected **{len(pairs):,} matched Black-White pairs / {len(rows):,} images**.
- Train: {selection['train']:,} images; development: {selection['development']:,}; test: {selection['test']:,}.
- Each race/gender cell has {len(rows)//4:,} images. Age and expression distributions match exactly within each split.
- Screened {len(audit):,} metadata-eligible candidates; {screening['eligible']:,} passed technical/duplicate screening.
- Matching used a frozen pool of {len(pool):,} annotated candidates from wave {report['annotation_wave']} per age/race/gender stratum.
- Uncertain expression estimates: {uncertain:,} images ({100*uncertain/len(rows):.1f}%).
- Image bytes: {size/1e6:.1f} MB before ZIP packaging.
- Model API spend and GPU rental spend: $0. This is local image curation, not model fine-tuning.

## Interpretation

The target {'was met' if not report['shortfall_pairs'] else 'was not fully met'} without changing the frozen matching rules.
Pairing is exact on measured categorical strata and constrained on confidence and two quality proxies.
It does not establish complete fairness, person-disjointness, or improved model behavior.
No IAT outputs influenced selection. No independent human review was performed.

Selected absolute standardized race-mean differences: contrast {selected_quality['contrast']:.3f},
log image detail {selected_quality['log_edge_energy']:.3f}, expression confidence
{selected_quality['expression_confidence']:.3f}, brightness {selected_quality['brightness']:.3f}.
Brightness is deliberately diagnostic-only and can reflect skin/background as well as lighting.
The comparison uses the same frozen candidate denominator before and after matching.

## Figures

"""
    for item in items:
        text += f"### {item['title']}\n\n![{item['title']}]({item['plot']})\n\n{item['finding']}\n\nSource: " + ", ".join(f"`{p}`" for p in item["sources"]) + ".\n\n"
    text += """## Important files

- `data/selected_images.csv`: one row per image, hashes, matching variables and split.
- `data/matched_pairs.csv`: one row per pair, image IDs, distances and split.
- `data/train_images.csv`, `data/development_images.csv`, `data/test_images.csv`: immutable research allocations.
- `private/dataset/images/`: selected image files, kept out of public GitHub.
- `data/validation.json`, `data/DATASET_READY.json`: validation and source-output checksums.
- `config/curation.json`, `src/`, `scripts/`, `requirements.lock`: reproducible implementation.
- `docs/METHODS.md`, `docs/COLUMN_GLOSSARY.csv`: rationale, limitations and variable definitions.

## Next research steps

Keep development/test images out of future training and preference generation. Construct grounded
training responses only from the training allocation, retaining accurate descriptions of negative
expressions. Compare chosen-only SFT and revised DPO with fresh LoRA adapters using the same pinned
base model. Select training settings with development data, then perform a final held-out evaluation
of bias, grounding, validity, and response diversity. Dataset balance alone is not a training result.
Report uncertain-expression and gender-stratified sensitivity results. Consider later human review
as a separately versioned dataset rather than silently editing this release.
"""
    atomic_bytes(ROOT / "RESULTS.md", text.encode())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
