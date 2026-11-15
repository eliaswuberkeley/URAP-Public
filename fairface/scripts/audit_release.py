"""Independent metadata/leakage checks, separate from the matching implementation."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from storage import read_csv, sha, write_json  # noqa: E402


def main():
    rows = read_csv(ROOT / "data/selected_images.csv")
    pairs = read_csv(ROOT / "data/matched_pairs.csv")
    inventory = {r["source_index"]: r for r in read_csv(ROOT / "data/source_inventory.csv")}
    prior = {r["source_index"] for r in read_csv(ROOT / "data/frozen/september29_selected_images.csv") if r["dataset"] == "fairface"}
    excluded = {r["source_index"] for r in read_csv(ROOT / "data/frozen/excluded_fairface_indices.csv")}
    selected = {r["source_index"] for r in rows}
    assert not (selected & prior), "September 29 overlap"
    assert not (selected & excluded), "Historical index overlap"
    assert len(selected) == len(rows)
    groups = defaultdict(set)
    sets = defaultdict(list)
    for r in rows:
        src = inventory[r["source_index"]]
        assert src["inventory_status"] == "eligible"
        for k in ("source_id", "source_race", "source_gender", "age_band", "source_revision", "source_split"):
            assert src[k] == r[k], (r["image_id"], k)
        groups[r["split"]].add(r["image_sha256"])
        sets[r["match_set_id"]].append(r)
    for name, members in groups.items():
        assert all(not (members & other) for key, other in groups.items() if key != name)
    for group in sets.values():
        assert len(group) == 4
        assert len({(r["source_race"], r["source_gender"]) for r in group}) == 4
        assert len({r["split"] for r in group}) == 1
    assert len({p["pair_id"] for p in pairs}) == len(pairs)
    assert len(pairs) * 2 == len(rows)
    for split in ("train", "development", "test"):
        subset = read_csv(ROOT / f"data/{split}_images.csv")
        assert {r["image_id"] for r in subset} == {r["image_id"] for r in rows if r["split"] == split}
    results = {"passed": True, "selected_images": len(rows), "pairs": len(pairs),
               "prior_september29_source_index_overlap": len(selected & prior),
               "historical_source_index_overlap": len(selected & excluded),
               "pair_set_count": len(sets), "images_by_split": dict(Counter(r["split"] for r in rows)),
               "publisher_labels_match_inventory": True, "person_identity_overlap_verified": False,
               "selected_manifest_sha256": sha(ROOT / "data/selected_images.csv")}
    write_json(ROOT / "analysis/independent_audit.json", results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
