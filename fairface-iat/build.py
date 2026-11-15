"""Replot frozen FairFace responses. No inference, image access, or network calls."""

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RACES = ("Black", "White")
EXPRESSIONS = ("negative", "neutral", "positive")


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(workspace):
    study = workspace / "gemma3-fairface-debiasing-october6"
    curation = workspace / "image-dataset-curation-october6"
    sources = {
        "data/base_responses.csv": study / "results/evaluation/base/development_screen/responses.csv",
        "data/curated_images.csv": curation / "data/selected_images.csv",
        "data/matched_pairs.csv": curation / "data/matched_pairs.csv",
        "iat_protocol.py": study / "scripts/iat_protocol.py",
        "reference/evaluation_manifest.json": study / "results/evaluation/base/development_screen/manifest.json",
        "reference/curation_config.json": curation / "config/curation.json",
    }
    if (ROOT / "reference/provenance.json").exists():
        raise ValueError("Inputs already frozen; run without --freeze-from.")
    provenance = {}
    for destination, source in sources.items():
        target = ROOT / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        provenance[destination] = {"source": str(source.relative_to(workspace)), "sha256": sha(source)}
    metrics_source = study / "results/analysis/model_metrics.csv"
    metrics = [r for r in read_csv(metrics_source)
               if r["panel"] == "development_screen" and r["model_key"] == "base"]
    write_csv(ROOT / "reference/recorded_base_metrics.csv", metrics)
    provenance["reference/recorded_base_metrics.csv"] = {
        "source": str(metrics_source.relative_to(workspace)), "source_sha256": sha(metrics_source),
        "selection": "panel=development_screen AND model_key=base",
        "sha256": sha(ROOT / "reference/recorded_base_metrics.csv"),
    }
    glossary = read_csv(curation / "docs/COLUMN_GLOSSARY.csv")
    definitions = {r["column"]: r["definition"] for r in glossary}
    dictionary = []
    for filename in ("curated_images.csv", "matched_pairs.csv"):
        for field in read_csv(ROOT / "data" / filename)[0]:
            dictionary.append({"file": "data/" + filename, "column": field, "definition": definitions[field]})
    write_csv(ROOT / "reference/dataset_columns.csv", dictionary)
    provenance["reference/dataset_columns.csv"] = {
        "source": str((curation / "docs/COLUMN_GLOSSARY.csv").relative_to(workspace)),
        "source_sha256": sha(curation / "docs/COLUMN_GLOSSARY.csv"),
        "sha256": sha(ROOT / "reference/dataset_columns.csv"),
        "selection": "Definitions for columns in the two included dataset tables",
    }
    (ROOT / "reference/provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def validate_inputs():
    from iat_protocol import NEGATIVE, annotate

    provenance = json.loads((ROOT / "reference/provenance.json").read_text())
    for filename, record in provenance.items():
        if sha(ROOT / filename) != record["sha256"]:
            raise ValueError("Frozen input changed: " + filename)
    responses = read_csv(ROOT / "data/base_responses.csv")
    images = read_csv(ROOT / "data/curated_images.csv")
    pairs = read_csv(ROOT / "data/matched_pairs.csv")
    assert len(responses) == 600 and len(images) == 4000 and len(pairs) == 2000
    index = {r["image_id"]: r for r in images}
    pair_index = {r["pair_id"]: r for r in pairs}
    assert len(index) == 4000 and len(pair_index) == 2000
    assert len({r["image_sha256"] for r in images}) == 4000
    members = []
    for pair in pairs:
        for race in RACES:
            image = index[pair[race.lower() + "_image_id"]]
            members.append(image["image_id"])
            assert image["source_race"] == race
            for field in ("pair_id", "split", "source_gender", "age_band", "expression_class", "expression_label"):
                assert image[field] == pair[field]
    assert len(set(members)) == 4000
    assert Counter(r["split"] for r in images) == {"train": 2400, "development": 800, "test": 800}
    assert Counter((r["source_race"], r["source_gender"]) for r in images) == {
        (race, gender): 1000 for race in RACES for gender in ("Female", "Male")}
    assert len({(r["image_id"], r["iteration"]) for r in responses}) == 600
    assert len({r["image_id"] for r in responses}) == 120
    assert len({r["match_id"] for r in responses}) == 60
    iterations = defaultdict(set)
    for row in responses:
        source = index[row["image_id"]]
        assert source["split"] == row["source_split"] == "development"
        assert row["role"] == "development_screen" and row["model_key"] == "base"
        assert row["image_sha256"] == source["image_sha256"]
        assert row["match_id"] == source["pair_id"]
        assert row["race"] == source["source_race"]
        assert row["gender"] == source["source_gender"]
        assert row["expression_class"] == source["expression_class"]
        assert row["age_band"] == source["age_band"]
        actual = annotate(row["sampled_raw"], row["fallback_raw"])
        for key, value in actual.items():
            assert str(value) == row[key], (row["image_id"], key)
        assert row["valid_three"] == "True"
        assert int(row["negative_count"]) == sum(w in NEGATIVE for w in row["words"].split("|"))
        iterations[row["image_id"]].add(int(row["iteration"]))
    assert all(values == {1, 2, 3, 4, 5} for values in iterations.values())
    assert Counter(r["race"] for r in responses) == {"Black": 300, "White": 300}
    return responses, images, pairs


def word_counts(responses):
    from iat_protocol import NEGATIVE, POSITIVE, WORDS

    counters = {race: Counter(w for r in responses if r["race"] == race and r["valid_three"] == "True"
                             for w in r["words"].split("|")) for race in RACES}
    result = []
    for valence, group in (("positive", POSITIVE), ("negative", NEGATIVE)):
        order = sorted(group, key=lambda w: (-sum(counters[r][w] for r in RACES), WORDS.index(w)))
        for rank, word in enumerate(order, 1):
            result.append({"plot_position": len(result) + 1, "valence": valence, "within_valence_rank": rank,
                           "word": word, "black_count": counters["Black"][word],
                           "white_count": counters["White"][word],
                           "pooled_count": sum(counters[r][word] for r in RACES)})
    return result


def rates(responses):
    result = []
    for expression in ("all_pooled",) + EXPRESSIONS:
        for race in RACES:
            selected = [r for r in responses if r["race"] == race
                        and (expression == "all_pooled" or r["expression_class"] == expression)]
            nwords = sum(len(r["words"].split("|")) for r in selected if r["valid_three"] == "True")
            negatives = sum(int(r["negative_count"]) for r in selected if r["valid_three"] == "True")
            result.append({"expression_group": expression, "race": race,
                           "images": len({r["image_id"] for r in selected}), "responses": len(selected),
                           "selected_words": nwords, "negative_words": negatives,
                           "negative_rate_percent": 100 * negatives / nwords})
    return result


def plots(counts, images, evaluation):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    destination = ROOT / "plots"
    destination.mkdir(exist_ok=True)

    def save(fig, name):
        for suffix in ("png", "pdf"):
            fig.savefig(destination / (name + "." + suffix), dpi=220,
                        facecolor="white", metadata={"Creator": "FairFace IAT replot", "CreationDate": None}
                        if suffix == "pdf" else {"Software": "FairFace IAT replot"})
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(23, 7))
    positions = np.arange(61)
    for race, offset, color in (("Black", -.2, "#315CA8"), ("White", .2, "#D47A32")):
        ax.bar(positions + offset, [r[race.lower() + "_count"] for r in counts], width=.38,
               color=color, label=race, zorder=3)
    ax.axvline(29.5, color="#737373", lw=1, linestyle="--")
    ax.set_xticks(positions, [r["word"] for r in counts], rotation=65, ha="right", fontsize=9)
    ax.set(xlim=(-.8, 60.8), ylabel="Word selections (count)")
    ax.grid(axis="y", color="#E1E5E8", linewidth=.7, zorder=0)
    ax.set_ylim(0, max(max(r["black_count"], r["white_count"]) for r in counts) * 1.13)
    ax.text(.245, 1.02, "Positive words", transform=ax.transAxes, ha="center", fontsize=12)
    ax.text(.755, 1.02, "Negative words", transform=ax.transAxes, ha="center", fontsize=12)
    ax.legend(frameon=False, loc="upper right", ncol=2)
    fig.suptitle("Gemma 3 12B IT: FairFace IAT-style word counts", y=.96, fontsize=19)
    fig.text(.5, .902, "120 development portraits | 60 matched pairs | 5 answers per image",
             ha="center", fontsize=12, color="#505860")
    fig.subplots_adjust(left=.055, right=.992, bottom=.24, top=.81)
    save(fig, "fairface_base_61_word_counts_sorted")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    split_counts = Counter(r["split"] for r in images)
    values = [split_counts[s] // 2 for s in ("train", "development", "test")]
    bars = axes[0].bar(["Training", "Development", "Test"], values, color=["#7C8792", "#315CA8", "#7C8792"])
    axes[0].bar_label(bars, padding=5, fmt="%d")
    axes[0].set(title="Curated dataset: 2,000 pairs", ylabel="Matched pairs", ylim=(0, 1450))
    expression_counts = Counter(r["expression_class"] for r in evaluation)
    values = [expression_counts[e] // 2 for e in EXPRESSIONS]
    bars = axes[1].bar(["Negative", "Neutral", "Positive"], values, color="#315CA8")
    axes[1].bar_label(bars, padding=5, fmt="%d")
    axes[1].set(title="IAT sample: 60 development pairs", ylabel="Matched pairs", ylim=(0, 27))
    axes[1].set_xlabel("Automated expression class")
    for ax in axes:
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    fig.subplots_adjust(left=.08, right=.98, bottom=.17, top=.83, wspace=.28)
    save(fig, "fairface_dataset_and_iat_sample")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-from", type=Path, help="One-time snapshot from the original research workspace")
    args = parser.parse_args()
    if args.freeze_from:
        freeze(args.freeze_from.resolve())
    responses, images, pairs = validate_inputs()
    counts = word_counts(responses)
    rate_rows = rates(responses)
    evaluation_ids = {r["image_id"] for r in responses}
    evaluation = [r for r in images if r["image_id"] in evaluation_ids]
    columns = ("image_id", "image_sha256", "image_path", "pair_id", "split", "source_race", "source_gender",
               "age_band", "expression_class", "expression_label", "expression_confidence")
    write_csv(ROOT / "data/evaluated_images.csv", [{k: r[k] for k in columns} for r in evaluation])
    write_csv(ROOT / "results/sorted_word_counts.csv", counts)
    write_csv(ROOT / "results/negative_rates.csv", rate_rows)
    recorded = read_csv(ROOT / "reference/recorded_base_metrics.csv")[0]
    standardized = {race: sum(r["negative_rate_percent"] for r in rate_rows
                             if r["race"] == race and r["expression_group"] != "all_pooled") / 3 for race in RACES}
    for race in RACES:
        assert math.isclose(standardized[race], float(recorded[race.lower() + "_rate"]) * 100, abs_tol=1e-12)
    summary = {
        "model": "google/gemma-3-12b-it",
        "revision": "96b6f1eccf38110c56df3a15bffe176da04bfd80",
        "source_run": "October 6, 2026; base, development_screen",
        "curated_images": len(images), "curated_pairs": len(pairs),
        "evaluated_images": len(evaluation), "evaluated_pairs": len(evaluation) // 2,
        "responses": len(responses), "words_per_response": 3,
        "parsed_valid_responses": sum(r["valid_three"] == "True" for r in responses),
        "raw_strict_valid_responses": sum(r["sampled_strict_valid"] == "True" for r in responses),
        "final_strict_valid_responses": sum(r["strict_valid"] == "True" for r in responses),
        "fallback_responses": sum(bool(r["fallback_raw"]) for r in responses),
        "original_generation_calls": sum(int(r["generation_calls"]) for r in responses),
        "new_generation_calls": 0,
        "image_version": "original; backgrounds not normalized",
        "expression_standardized_negative_percent": standardized,
        "pooled_negative_percent": {r["race"]: r["negative_rate_percent"] for r in rate_rows
                                    if r["expression_group"] == "all_pooled"},
        "sort_rule": "Descending Black + White count within each valence; ties follow original vocabulary order",
        "checks": "All 120 image IDs, hashes, split, labels and pair IDs match the frozen 4,000-image manifest",
        "interpretation": "Exploratory development sample, not all 4,000 images or a confirmatory fairness result",
    }
    (ROOT / "results/summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plots(counts, images, evaluation)
    paths = sorted(p for p in ROOT.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt"
                   and not any(part.startswith(".") or part == "__pycache__" for part in p.relative_to(ROOT).parts)
                   and p.suffix != ".pyc")
    (ROOT / "SHA256SUMS.txt").write_text("".join(f"{sha(p)}  {p.relative_to(ROOT)}\n" for p in paths))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
