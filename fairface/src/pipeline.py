"""Local FairFace curation with durable, resumable per-image processing."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import io
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np
from PIL import Image, ImageOps
from scipy.fft import dctn

from matching import match, stable
from storage import Store, atomic_bytes, digest, read_csv, sha, write_csv, write_json, writer_lock

ROOT = Path(__file__).resolve().parents[1]
STOP = False
FER = {"neutral": "neutral", "happy": "positive", "sad": "negative", "angry": "negative",
       "disgust": "negative", "fear": "negative"}
FROZEN_INPUTS = {
    "historical_image_fingerprints.csv": "data/frozen/historical_image_fingerprints.csv",
    "excluded_fairface_indices.csv": "data/frozen/excluded_fairface_indices.csv",
    "historical_sources.json": "data/frozen/historical_sources.json",
    "source_files.json": "data/frozen/source_files.json",
    "expression_model.json": "data/frozen/expression_model.json",
    "september29_selected_images.csv": "data/selected_images.csv",
    "september29_candidate_audit.csv": "data/candidate_audit.csv",
    "september29_expression_annotations.csv": "data/expression_annotations.csv",
}


def event(stage, **kwargs):
    obj = {"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stage": stage, **kwargs}
    print(json.dumps(obj, sort_keys=True), flush=True)
    write_json(ROOT / "runtime/progress.json", obj)


def stop_requested(*_):
    global STOP
    STOP = True


def stop_check(store):
    if STOP:
        store.backup(ROOT / "runtime/checkpoint.backup.sqlite")
        event("paused", reason="signal_received_at_completed_batch", resume="scripts/run_local.sh run")
        raise SystemExit(130)


def metrics(image):
    g = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    small = np.asarray(image.convert("L").resize((32, 32)), dtype=np.float32)
    coeff = dctn(small, type=2, norm="ortho")[:8, :8]
    mask = coeff > np.median(coeff.ravel()[1:])
    phash = int("".join("1" if b else "0" for b in mask.ravel()), 2)
    dh = np.asarray(image.convert("L").resize((9, 8)), dtype=np.int16)
    bits = dh[:, 1:] > dh[:, :-1]
    dhash = int("".join("1" if b else "0" for b in bits.ravel()), 2)
    edge = float(np.abs(np.diff(g, axis=0)).mean() + np.abs(np.diff(g, axis=1)).mean())
    return {"width": image.width, "height": image.height,
            "pixel_sha256": digest(f"{image.width}x{image.height}|RGB|".encode() + image.tobytes()),
            "perceptual_hash": f"{phash:016x}", "difference_hash": f"{dhash:016x}",
            "brightness": float(g.mean()), "contrast": float(g.std()), "edge_energy": edge,
            "log_edge_energy": math.log(max(edge, 1e-8)),
            "clipped_dark_fraction": float((g <= 5 / 255).mean()),
            "clipped_light_fraction": float((g >= 250 / 255).mean())}


class HashIndex:
    def __init__(self):
        self.buckets = defaultdict(set)
        self.items = []
        self.pixels = {}

    def add(self, row):
        i = len(self.items)
        self.items.append(row)
        self.pixels[row["pixel_sha256"]] = row
        h = int(row["perceptual_hash"], 16)
        for shift, width in [(0, 13), (13, 13), (26, 13), (39, 13), (52, 12)]:
            self.buckets[shift, (h >> shift) & ((1 << width) - 1)].add(i)

    def matches(self, row, config):
        if row["pixel_sha256"] in self.pixels:
            return self.pixels[row["pixel_sha256"]]
        h, d = int(row["perceptual_hash"], 16), int(row["difference_hash"], 16)
        ids = set()
        for shift, width in [(0, 13), (13, 13), (26, 13), (39, 13), (52, 12)]:
            ids.update(self.buckets.get((shift, (h >> shift) & ((1 << width) - 1)), ()))
        for i in sorted(ids):
            other = self.items[i]
            if ((h ^ int(other["perceptual_hash"], 16)).bit_count() <= config["perceptual_hash_max_distance"]
                    and (d ^ int(other["difference_hash"], 16)).bit_count() <= config["difference_hash_max_distance"]):
                return other
        return None


def freeze(config):
    root = ROOT.parent
    prior = root / "image-dataset-curation-september29"
    records = []
    for name, rel in FROZEN_INPUTS.items():
        source, dest = prior / rel, ROOT / "data/frozen" / name
        if dest.exists():
            if sha(dest) != sha(source):
                raise ValueError(f"Historical input drift: {name}")
        else:
            atomic_bytes(dest, source.read_bytes())
        records.append({"file": str(dest.relative_to(ROOT)), "sha256": sha(dest),
                        "source": str(source.relative_to(root))})
    write_json(ROOT / "data/frozen/input_manifest.json", records)
    source_dir = root / ("gemma3-fairface-debiasing-august3/data/august3/huggingface_cache/datasets/"
                         "HuggingFaceM4___fair_face/0.25/0.0.0/" + config["fairface_revision"])
    model_dir = root / ("multimodel-bias-benchmark-august31/data/model-cache/"
                        "models--trpakov--vit-face-expression/snapshots/" + config["expression_model_revision"])
    paths = {"source_dir": str(source_dir), "model_dir": str(model_dir)}
    paths_file = ROOT / "runtime/local_paths.json"
    if paths_file.exists():
        paths = json.loads(paths_file.read_text())
    else:
        write_json(paths_file, paths)
    return records, {k: Path(v) for k, v in paths.items()}


def load_source(paths, config):
    import pyarrow as pa
    import pyarrow.ipc as ipc
    spec = json.loads((ROOT / "data/frozen/source_files.json").read_text())
    info = json.loads((paths["source_dir"] / "dataset_info.json").read_text())
    if info["features"] != spec["fairface_dataset_info"]["features"]:
        raise ValueError("Source label mapping drift")
    tables = []
    for item in spec["fairface_arrow"]:
        p = paths["source_dir"] / item["filename"]
        if sha(p) != item["sha256"]:
            raise ValueError("Source Arrow SHA-256 mismatch: " + p.name)
        with pa.memory_map(str(p), "r") as f:
            tables.append(ipc.open_stream(f).read_all())
    table = pa.concat_tables(tables)
    if len(table) != 86744 or spec["source_revisions"]["fairface"] != config["fairface_revision"]:
        raise ValueError("Source revision/size mismatch")
    return table, info


def history():
    rows = read_csv(ROOT / "data/frozen/historical_image_fingerprints.csv")
    rows += read_csv(ROOT / "data/frozen/september29_selected_images.csv")
    index = HashIndex()
    for r in rows:
        index.add(r)
    return index


def inventory(table, info, config):
    excluded = {int(r["source_index"]) for r in read_csv(ROOT / "data/frozen/excluded_fairface_indices.csv")}
    old = {int(r["source_index"]) for r in read_csv(ROOT / "data/frozen/september29_selected_images.csv")
           if r["dataset"] == "fairface"}
    rows = []
    for i, r in enumerate(table.select(["age", "gender", "race"]).to_pylist()):
        labels = {k: info["features"][k]["names"][r[k]] for k in r}
        if labels["race"] not in config["races"]:
            continue
        status = "eligible"
        if labels["age"] not in config["age_bands"]:
            status = "age_out_of_scope"
        elif i in excluded:
            status = "historical_index_exclusion"
        elif i in old:
            status = "september29_selection_exclusion"
        source_id = f"train:{i:06d}"
        rows.append({"image_id": f"ff_train_{i:06d}", "source_id": source_id, "source_index": i,
                     "source_split": "train", "source_revision": config["fairface_revision"],
                     "source_race": labels["race"], "source_gender": labels["gender"],
                     "age_band": labels["age"], "inventory_status": status,
                     "screen_rank": stable(config["seed"], source_id)})
    write_csv(ROOT / "data/source_inventory.csv", rows)
    event("inventory", counts=dict(Counter(r["inventory_status"] for r in rows)))
    return sorted((r for r in rows if r["inventory_status"] == "eligible"), key=lambda r: r["screen_rank"])


def raw_image(table, row):
    value = table["image"][int(row["source_index"])].as_py()["bytes"]
    if value is None:
        raise ValueError("Missing embedded source image")
    return value


def image_png(raw):
    with Image.open(io.BytesIO(raw)) as im:
        im.load()
        rgb = ImageOps.exif_transpose(im).convert("RGB")
    out = io.BytesIO()
    rgb.save(out, format="PNG", optimize=False)
    return rgb, out.getvalue()


def check_cached_images(store, table):
    records = store.get("screen")
    repairs = 0
    for row in records.values():
        if row["screening_status"] != "eligible":
            continue
        p = ROOT / row["candidate_path"]
        if not p.is_file() or sha(p) != row["image_sha256"]:
            raw = raw_image(table, row)
            if digest(raw) != row["source_image_sha256"]:
                raise ValueError("Cannot repair from changed source")
            _, data = image_png(raw)
            if digest(data) != row["image_sha256"]:
                raise ValueError("PNG encoding drift during repair")
            atomic_bytes(p, data)
            repairs += 1
    event("resume_validation", committed_screen_records=len(records), repaired_images=repairs)
    return records


def screen(store, rows, table, config, stop_after_batches=None):
    prior = check_cached_images(store, table)
    history_index, current = history(), HashIndex()
    for r in sorted(prior.values(), key=lambda r: r["screen_rank"]):
        if r["screening_status"] == "eligible":
            current.add(r)
    todo = [r for r in rows if r["image_id"] not in prior]
    # Committed screening must be a prefix of the fixed order, never a partial
    # arbitrary subset, because greedy duplicate representatives depend on order.
    if set(prior) != {r["image_id"] for r in rows[:len(prior)]}:
        raise ValueError("Screening checkpoint is not a deterministic prefix")
    batch_size = config["screen_batch_size"]
    start = time.monotonic()
    for offset in range(0, len(todo), batch_size):
        batch = []
        for original in todo[offset:offset + batch_size]:
            r = dict(original, screening_status="eligible", duplicate_of="")
            try:
                raw = raw_image(table, r)
                rgb, png = image_png(raw)
            except (OSError, ValueError) as exc:
                r["screening_status"] = "unreadable"
                r["error_type"] = type(exc).__name__
            else:
                r["source_image_sha256"] = digest(raw)
                if min(rgb.size) < config["min_side_pixels"]:
                    r["screening_status"] = "undersized"
                else:
                    r.update(metrics(rgb))
                    old, seen = history_index.matches(r, config), current.matches(r, config)
                    if old:
                        r["screening_status"] = "historical_duplicate"
                        r["duplicate_of"] = old.get("image_id", old.get("source_id", "historical"))
                    elif seen:
                        r["screening_status"] = "candidate_duplicate"
                        r["duplicate_of"] = seen["image_id"]
                    else:
                        r["candidate_path"] = f"private/candidates/{r['image_id']}.png"
                        r["image_sha256"] = digest(png)
                        atomic_bytes(ROOT / r["candidate_path"], png)
                        current.add(r)
            batch.append(r)
        store.put("screen", batch)
        if offset % (batch_size * 8) == 0 or offset + batch_size >= len(todo):
            event("screen", completed=len(prior) + min(offset + batch_size, len(todo)), total=len(rows),
                  images_per_second=round(min(offset + batch_size, len(todo)) / max(time.monotonic() - start, .001), 2))
        stop_check(store)
        if stop_after_batches and offset // batch_size + 1 >= stop_after_batches:
            event("intentional_stop", stage_completed="screen_batch")
            return False
    store.backup(ROOT / "runtime/checkpoint.backup.sqlite")
    write_csv(ROOT / "data/candidate_audit.csv", list(store.get("screen").values()))
    return True


def annotation(row, label, confidence, config, provenance):
    label = label.lower()
    klass = FER.get(label, "uncertain") if confidence >= config["expression_confidence_for_class"] else "uncertain"
    return {"image_id": row["image_id"], "image_sha256": row["image_sha256"],
            "expression_label": label, "expression_confidence": float(confidence), "expression_class": klass,
            "expression_model_id": config["expression_model_id"],
            "expression_model_revision": config["expression_model_revision"], "expression_provenance": provenance}


def verify_annotation(row, ann, config):
    if ann["image_sha256"] != row["image_sha256"]:
        raise ValueError("Annotation/image checksum mismatch")
    for k in ("expression_model_id", "expression_model_revision"):
        if ann[k] != config[k]:
            raise ValueError("Expression model drift")
    expected = annotation(row, ann["expression_label"], float(ann["expression_confidence"]), config, ann["expression_provenance"])
    if expected != ann or not 0 <= float(ann["expression_confidence"]) <= 1:
        raise ValueError("Invalid annotation")


def reuse_annotations(store, rows, config):
    previous = {r["image_sha256"]: r for r in read_csv(ROOT / "data/frozen/september29_expression_annotations.csv")}
    existing = store.get("annotation")
    reused = []
    for r in rows:
        if r["image_id"] in existing:
            verify_annotation(r, existing[r["image_id"]], config)
            continue
        old = previous.get(r["image_sha256"])
        if old and all(old[k] == config[k] for k in ("expression_model_id", "expression_model_revision")):
            reused.append(annotation(r, old["expression_label"], float(old["expression_confidence"]), config,
                                     "september29_same_image_and_model_hash"))
    store.put("annotation", reused)
    event("annotation_cache", reused=len(reused), existing=len(existing))


def annotate_wave(store, rows, paths, config, max_new_batches=None):
    prior = store.get("annotation")
    todo = [r for r in rows if r["image_id"] not in prior]
    if not todo:
        return True
    import torch
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    spec = json.loads((ROOT / "data/frozen/expression_model.json").read_text())
    for item in spec["files"]:
        if sha(paths["model_dir"] / item["filename"]) != item["sha256"]:
            raise ValueError("Expression model file hash mismatch")
    torch.set_num_threads(config["torch_threads"])
    torch.manual_seed(config["torch_seed"])
    torch.use_deterministic_algorithms(True)
    processor = AutoImageProcessor.from_pretrained(paths["model_dir"], local_files_only=True, use_fast=False)
    model = AutoModelForImageClassification.from_pretrained(paths["model_dir"], local_files_only=True).eval().to(config["device"])
    batch_size = config["expression_batch_size"]
    start = time.monotonic()
    for offset in range(0, len(todo), batch_size):
        batch = todo[offset:offset + batch_size]
        images = []
        for r in batch:
            with Image.open(ROOT / r["candidate_path"]) as im:
                images.append(im.convert("RGB").copy())
        # Pad the final batch so every inference uses a fixed batch shape.
        padded = images + [images[-1]] * (batch_size - len(images))
        inputs = processor(images=padded, return_tensors="pt").to(config["device"])
        with torch.inference_mode():
            probs = model(**inputs).logits.softmax(dim=-1).cpu().numpy()
        result = []
        for r, prob in zip(batch, probs[:len(batch)], strict=True):
            i = int(prob.argmax())
            result.append(annotation(r, model.config.id2label[i], float(prob[i]), config, "october6_local_cpu"))
        store.put("annotation", result)
        done = min(offset + batch_size, len(todo))
        speed = done / max(time.monotonic() - start, .001)
        if offset % (batch_size * 4) == 0 or done == len(todo):
            event("annotate", new_completed=done, new_total=len(todo), images_per_second=round(speed, 3),
                  estimated_remaining_minutes=round((len(todo) - done) / speed / 60, 1))
        if offset % (batch_size * 32) == 0:
            store.backup(ROOT / "runtime/checkpoint.backup.sqlite")
        stop_check(store)
        if max_new_batches and offset // batch_size + 1 >= max_new_batches:
            event("intentional_stop", stage_completed="annotation_batch", new_completed=done)
            return False
    store.backup(ROOT / "runtime/checkpoint.backup.sqlite")
    return True


def wave_rows(rows, wave, config):
    groups = defaultdict(list)
    for r in rows:
        groups[r["source_race"], r["source_gender"], r["age_band"]].append(r)
    out = []
    for _, pool in sorted(groups.items()):
        out.extend(sorted(pool, key=lambda r: r["screen_rank"])[:wave])
    return sorted(out, key=lambda r: r["screen_rank"])


def combined(store, rows, config):
    anns = store.get("annotation")
    out = []
    for r in rows:
        a = anns[r["image_id"]]
        verify_annotation(r, a, config)
        out.append({**r, **a})
    return out


def save_selection(pairs, quotas, report, rows, wave):
    by_id = {r["image_id"]: r for r in rows}
    selected = []
    for p in pairs:
        for side in ("black", "white"):
            r = dict(by_id[p[side + "_image_id"]])
            r.update(pair_id=p["pair_id"], match_set_id=p["match_set_id"], split=p["split"],
                     image_path="images/" + r["image_id"] + ".png", review_status="automated_only")
            selected.append(r)
    selected.sort(key=lambda r: r["image_id"])
    report.update(annotation_wave=wave, human_validated=False, person_disjointness_verified=False,
                  bias_tests_run=0, training_run=False)
    write_csv(ROOT / "data/matched_pairs.csv", pairs)
    write_csv(ROOT / "data/selected_images.csv", selected)
    write_csv(ROOT / "data/matching_candidate_pool.csv", rows)
    write_csv(ROOT / "data/common_support_quotas.csv", quotas)
    write_json(ROOT / "data/selection_report.json", report)
    for split in ("train", "development", "test"):
        write_csv(ROOT / f"data/{split}_images.csv", [r for r in selected if r["split"] == split])
        write_csv(ROOT / f"data/{split}_pairs.csv", [p for p in pairs if p["split"] == split])
    event("selection", **report)


def materialize():
    selected = read_csv(ROOT / "data/selected_images.csv")
    for row in selected:
        dest = ROOT / "private/dataset" / row["image_path"]
        if not dest.exists() or sha(dest) != row["image_sha256"]:
            source = ROOT / row["candidate_path"]
            if sha(source) != row["image_sha256"]:
                raise ValueError("Candidate changed before packaging")
            atomic_bytes(dest, source.read_bytes())
    atomic_bytes(ROOT / "private/dataset/manifest.csv", (ROOT / "data/selected_images.csv").read_bytes())
    atomic_bytes(ROOT / "private/dataset/matched_pairs.csv", (ROOT / "data/matched_pairs.csv").read_bytes())


def validate(config):
    rows = read_csv(ROOT / "data/selected_images.csv")
    pairs = read_csv(ROOT / "data/matched_pairs.csv")
    report = json.loads((ROOT / "data/selection_report.json").read_text())
    index, ids = history(), {r["image_id"]: r for r in rows}
    errors = []
    for key in ("image_id", "source_index", "pixel_sha256", "image_sha256"):
        if len({r[key] for r in rows}) != len(rows):
            errors.append("Nonunique " + key)
    for r in rows:
        p = ROOT / "private/dataset" / r["image_path"]
        if not p.is_file() or sha(p) != r["image_sha256"]:
            errors.append("Image checksum failure " + r["image_id"])
        if index.matches(r, config):
            errors.append("Historical or within-release duplicate " + r["image_id"])
        index.add(r)
    used = []
    for p in pairs:
        b, w = ids[p["black_image_id"]], ids[p["white_image_id"]]
        used += [b["image_id"], w["image_id"]]
        if b["source_race"] != "Black" or w["source_race"] != "White":
            errors.append("Pair race-label mismatch")
        for k in ("age_band", "source_gender", "expression_class", "expression_label", "split", "pair_id", "match_set_id"):
            if b[k] != w[k] or b[k] != p[k]:
                errors.append("Pair mismatch " + k)
        if abs(float(b["expression_confidence"]) - float(w["expression_confidence"])) > config["expression_confidence_caliper"] + 1e-10:
            errors.append("Confidence caliper violated")
        for k in config["quality_features"]:
            if abs(float(b[k]) - float(w[k])) / report["quality_scales"][k] > config["quality_caliper_sd"] + 1e-10:
                errors.append("Quality caliper violated")
    if len(used) != len(set(used)) or set(used) != set(ids):
        errors.append("Pair assignment does not partition images")
    balance = []
    for split in ("all", "train", "development", "test"):
        subset = [r for r in rows if split == "all" or r["split"] == split]
        cells = [Counter((r["age_band"], r["expression_class"], r["expression_label"]) for r in subset
                         if r["source_race"] == race and r["source_gender"] == gender)
                 for race in config["races"] for gender in config["genders"]]
        equal = all(c == cells[0] for c in cells)
        balance.append({"split": split, "images": len(subset), "joint_balance_equal": equal})
        if not equal:
            errors.append("Joint balance failure " + split)
    result = {"passed": not errors, "errors": errors, "images": len(rows), "pairs": len(pairs),
              "target_reached": len(pairs) == config["target_pairs"], "balance": balance,
              "person_disjointness_verified": False, "human_validated": False,
              "config_sha256": sha(ROOT / "config/curation.json")}
    write_json(ROOT / "data/validation.json", result)
    event("validation", **result)
    if errors:
        raise ValueError("Validation failed")
    write_json(ROOT / "data/DATASET_READY.json", {
        "status": "validated_target_reached" if result["target_reached"] else "validated_shortfall",
        "files": {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / "data").glob("*.csv"))},
        "validation_sha256": sha(ROOT / "data/validation.json")})


def run(args):
    config = json.loads((ROOT / "config/curation.json").read_text())
    if not 0 <= config["perceptual_hash_max_distance"] <= 4:
        raise ValueError("Hash index supports distance <=4")
    inputs, paths = freeze(config)
    table, info = load_source(paths, config)
    import importlib.metadata
    versions = {p: importlib.metadata.version(p) for p in ("numpy", "Pillow", "scipy", "torch", "transformers", "pyarrow")}
    contract = {"config": config, "inputs": inputs, "versions": versions,
                "code": {p.name: sha(p) for p in sorted((ROOT / "src").glob("*.py"))}}
    write_json(ROOT / "runtime/environment.json", {"python": sys.version, "versions": versions})
    store = Store(ROOT / "runtime/checkpoint.sqlite", contract)
    try:
        rows = inventory(table, info, config)
        if not screen(store, rows, table, config, args.screen_batches):
            return
        eligible = [r for r in store.get("screen").values() if r["screening_status"] == "eligible"]
        reuse_annotations(store, eligible, config)
        for wave in config["annotation_waves_per_age_race_gender"]:
            candidates = wave_rows(eligible, wave, config)
            event("annotation_wave", cap_per_age_race_gender=wave, candidates=len(candidates))
            if not annotate_wave(store, candidates, paths, config, args.annotation_batches):
                return
            annotated = combined(store, candidates, config)
            pairs, quotas, report = match(annotated, config)
            event("matching_support", wave=wave, **report)
            write_json(ROOT / f"data/wave_{wave}_support.json", report)
            if len(pairs) == config["target_pairs"] or wave == config["annotation_waves_per_age_race_gender"][-1]:
                write_csv(ROOT / "data/expression_annotations.csv", list(store.get("annotation").values()))
                save_selection(pairs, quotas, report, annotated, wave)
                materialize()
                validate(config)
                break
            stop_check(store)
        store.backup(ROOT / "runtime/checkpoint.backup.sqlite")
    finally:
        store.close()


def main():
    global ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["run", "status", "validate"])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--screen-batches", type=int)
    parser.add_argument("--annotation-batches", type=int)
    args = parser.parse_args()
    ROOT = args.root.resolve()
    if args.stage == "status":
        p = ROOT / "runtime/progress.json"
        print(p.read_text() if p.exists() else "Not started")
        return
    for s in (signal.SIGINT, signal.SIGTERM):
        signal.signal(s, stop_requested)
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[key] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    with writer_lock(ROOT):
        if args.stage == "run":
            run(args)
        else:
            validate(json.loads((ROOT / "config/curation.json").read_text()))


if __name__ == "__main__":
    main()
