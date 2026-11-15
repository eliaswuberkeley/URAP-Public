"""Deterministic measured-covariate matching; not race-only counterfactuals."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib

import numpy as np
from scipy.optimize import linear_sum_assignment


def stable(seed, value):
    return hashlib.sha256(f"{seed}|{value}".encode()).hexdigest()


def capped_allocation(total, capacities, weights=None):
    if total < 0 or total > sum(capacities.values()):
        raise ValueError("Insufficient common support")
    weights = weights or {k: 1 for k in capacities}
    out = {k: 0 for k in capacities}
    for _ in range(total):
        k = min((k for k in capacities if out[k] < capacities[k]),
                key=lambda k: ((out[k] + 1) / max(weights.get(k, 0), 1e-9), str(k)))
        out[k] += 1
    return out


def expression_stratum(row):
    return row["expression_class"] + ":" + row["expression_label"]


def scales_for(rows, config):
    return {k: max(float(np.std([float(r[k]) for r in rows], ddof=1)), 1e-8)
            for k in config["quality_features"]}


def pair_pool(black, white, config, scales):
    if not black or not white:
        return []
    black = sorted(black, key=lambda r: r["image_id"])
    white = sorted(white, key=lambda r: r["image_id"])
    confidence = np.abs(np.array([r["expression_confidence"] for r in black])[:, None]
                        - np.array([r["expression_confidence"] for r in white])[None, :])
    valid = confidence <= config["expression_confidence_caliper"] + 1e-12
    cost = (confidence / config["expression_confidence_caliper"]) ** 2
    diffs = {}
    for k in config["quality_features"]:
        delta = (np.array([float(r[k]) for r in black])[:, None]
                 - np.array([float(r[k]) for r in white])[None, :]) / scales[k]
        valid &= np.abs(delta) <= config["quality_caliper_sd"] + 1e-12
        cost += delta ** 2
        diffs[k] = delta
    # A disallowed edge costs more than every allowed edge combined. The
    # assignment therefore maximizes feasible cardinality before minimizing cost.
    penalty = (min(len(black), len(white)) + 1) * (2 + len(diffs) * config["quality_caliper_sd"] ** 2)
    a, b = linear_sum_assignment(np.where(valid, cost, penalty))
    pairs = []
    for i, j in zip(a, b, strict=True):
        if not valid[i, j]:
            continue
        x, y = black[i], white[j]
        pairs.append({
            "pair_id": "ffpair_" + stable(config["seed"], x["image_id"] + "|" + y["image_id"])[:20],
            "black_image_id": x["image_id"], "white_image_id": y["image_id"],
            "source_gender": x["source_gender"], "age_band": x["age_band"],
            "expression_stratum": expression_stratum(x),
            "expression_class": x["expression_class"], "expression_label": x["expression_label"],
            "match_cost": float(cost[i, j]), "confidence_abs_difference": float(confidence[i, j]),
            **{k + "_difference_sd": float(v[i, j]) for k, v in diffs.items()},
        })
    return sorted(pairs, key=lambda p: (p["match_cost"], stable(config["seed"], p["pair_id"])))


def match(rows, config):
    if len(rows) < 2:
        raise ValueError("Insufficient annotated candidates")
    scales = scales_for(rows, config)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["age_band"], expression_stratum(r), r["source_gender"], r["source_race"])].append(r)
    strata = sorted({(r["age_band"], expression_stratum(r)) for r in rows})
    pool, caps = {}, {}
    for age, expr in strata:
        for gender in config["genders"]:
            pool[age, expr, gender] = pair_pool(groups[age, expr, gender, "Black"],
                                               groups[age, expr, gender, "White"], config, scales)
        caps[age, expr] = min(len(pool[age, expr, g]) for g in config["genders"])
    age_caps = {a: sum(n for (age, _), n in caps.items() if age == a) for a in config["age_bands"]}
    target = min(config["target_pairs"] // 2, sum(caps.values()))
    age_quotas = capped_allocation(target, age_caps)
    quota = {}
    for age in config["age_bands"]:
        cap = {e: n for (a, e), n in caps.items() if a == age}
        weights = {e: sum(len(groups[age, e, g, r]) for g in config["genders"] for r in config["races"])
                   for e in cap}
        quota.update({(age, e): n for e, n in capped_allocation(age_quotas[age], cap, weights).items()})
    units = [((age, e), i) for (age, e), n in sorted(quota.items()) for i in range(n)]
    units.sort(key=lambda u: stable(config["seed"], repr(u)))
    nt = int(target * config["split_fractions"]["train"])
    nd = int(target * config["split_fractions"]["development"])
    split_by = {u: "train" if i < nt else "development" if i < nt + nd else "test"
                for i, u in enumerate(units)}
    pairs = []
    for (age, expr), n in sorted(quota.items()):
        for gender in config["genders"]:
            for i, p in enumerate(pool[age, expr, gender][:n]):
                pairs.append({**p, "split": split_by[((age, expr), i)],
                              "match_set_id": "ffset_" + stable(config["seed"], f"{age}|{expr}|{i}")[:16]})
    pairs.sort(key=lambda r: r["pair_id"])
    quotas = [{"age_band": a, "expression_stratum": e, "common_pair_capacity_per_gender": caps[a, e],
               "selected_pairs_per_gender": n} for (a, e), n in sorted(quota.items())]
    return pairs, quotas, {
        "target_pairs": config["target_pairs"], "selected_pairs": len(pairs),
        "selected_images": len(pairs) * 2, "shortfall_pairs": config["target_pairs"] - len(pairs),
        "common_pair_capacity_per_gender": sum(caps.values()),
        "annotated_candidates": len(rows), "quality_scales": scales,
        "pairs_by_split": dict(Counter(p["split"] for p in pairs)),
        "pairs_by_gender": dict(Counter(p["source_gender"] for p in pairs)),
        "selected_age_counts_per_race_gender": age_quotas,
    }
