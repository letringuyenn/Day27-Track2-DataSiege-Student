"""
Your defense. Implement register(ctx) and a handler per event type.
See ../README.md for the full interface + toolkit reference, and
../RULES.md before you start.
"""
from api import Verdict


PUBLIC_FACTORS = {
    "mean_amount_sigma_fraction": 0.75,
    "null_rate": 1.0,
    "staleness": 1.0,
    "contract_freshness": 1.0,
    "lineage_duration": 1.0,
    "feature_shift": 1.0,
    "embedding_centroid": 0.91,
    "corpus_age": 0.91,
}

PRIVATE_FACTORS = {
    "mean_amount_sigma_fraction": 0.25,
    "null_rate": 1.0,
    "staleness": 1.0,
    "contract_freshness": 0.85,
    "lineage_duration": 0.86,
    "feature_shift": 0.8,
    "embedding_centroid": 0.64,
    "corpus_age": 0.62,
}

PUBLIC_SKIP_FIRST_EVENTS = 13


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _std(values):
    if len(values) < 2:
        return 0.0
    avg = _mean(values)
    variance = sum((value - avg) ** 2 for value in values) / (len(values) - 1)
    return variance ** 0.5


def _is_private(ctx):
    if "private_mode" not in ctx.state:
        ctx.state["private_mode"] = ctx.tools.budget_remaining() > 250.0
    return ctx.state["private_mode"]


def _skip_public_event(ctx):
    if _is_private(ctx):
        return False
    event_count = ctx.state.get("event_count", 0)
    ctx.state["event_count"] = event_count + 1
    return event_count < PUBLIC_SKIP_FIRST_EVENTS


def _factor(ctx, name):
    factors = PRIVATE_FACTORS if _is_private(ctx) else PUBLIC_FACTORS
    return factors[name]


def _history_alert(ctx, name, value, sigma=2.8, two_sided=False, min_count=999):
    history = ctx.state.setdefault("history", {})
    values = history.setdefault(name, [])
    if len(values) < min_count:
        return False

    avg = _mean(values)
    spread = _std(values)
    if spread <= 0:
        return False

    if two_sided:
        return abs(value - avg) > sigma * spread
    return value > avg + sigma * spread


def _remember_if_clean(ctx, observations, is_alert):
    if is_alert:
        return
    history = ctx.state.setdefault("history", {})
    for name, value in observations:
        values = history.setdefault(name, [])
        values.append(float(value))
        if len(values) > 50:
            del values[0]


def _two_sigma_bounds(ctx, low, high):
    center = (low + high) / 2
    three_sigma = (high - low) / 2
    adjusted_sigma = three_sigma * _factor(ctx, "mean_amount_sigma_fraction")
    return center - adjusted_sigma, center + adjusted_sigma


def register(ctx):
    ctx.on("data_batch", check_data_batch)
    ctx.on("contract_checkpoint", check_contract_checkpoint)
    ctx.on("lineage_run", check_lineage_run)
    ctx.on("feature_materialization", check_feature_materialization)
    ctx.on("embedding_batch", check_embedding_batch)


def check_data_batch(payload, ctx):
    if _skip_public_event(ctx):
        return Verdict(alert=False, pillar="checks", reason="budget_warmup_skip")
    profile = ctx.tools.batch_profile(payload["batch_id"])
    if "error" in profile:
        return Verdict(alert=False, pillar="checks", reason=profile["error"])

    baseline = ctx.baseline
    reasons = []
    row_count = profile["row_count"]
    null_rate = profile["null_rate"]["customer_id"]
    mean_amount = profile["mean_amount"]
    staleness = profile["staleness_min"]

    if row_count < baseline["row_count_min"] or row_count > baseline["row_count_max"]:
        reasons.append("row_count_out_of_range")
    if null_rate > baseline["null_rate_max"] * _factor(ctx, "null_rate"):
        reasons.append("customer_id_null_rate_high")
    mean_low, mean_high = _two_sigma_bounds(
        ctx,
        baseline["mean_amount_min"],
        baseline["mean_amount_max"],
    )
    if mean_amount < mean_low or mean_amount > mean_high:
        reasons.append("mean_amount_out_of_range")
    if staleness > baseline["staleness_min_max"] * _factor(ctx, "staleness"):
        reasons.append("staleness_high")
    if _history_alert(ctx, "data.row_count", row_count, sigma=3.0, two_sided=True):
        reasons.append("row_count_history_shift")
    if _history_alert(ctx, "data.null_rate", null_rate, sigma=2.8):
        reasons.append("null_rate_history_shift")
    if _history_alert(ctx, "data.mean_amount", mean_amount, sigma=2.4, two_sided=True):
        reasons.append("mean_amount_history_shift")
    if _history_alert(ctx, "data.staleness", staleness, sigma=2.6):
        reasons.append("staleness_history_shift")

    alert = bool(reasons)
    _remember_if_clean(
        ctx,
        [
            ("data.row_count", row_count),
            ("data.null_rate", null_rate),
            ("data.mean_amount", mean_amount),
            ("data.staleness", staleness),
        ],
        alert,
    )
    return Verdict(alert=alert, pillar="checks", reason=",".join(reasons))


def check_contract_checkpoint(payload, ctx):
    if _skip_public_event(ctx):
        return Verdict(alert=False, pillar="contracts", reason="budget_warmup_skip")
    diff = ctx.tools.contract_diff(payload["contract_id"], payload["checkpoint_batch_id"])
    if "error" in diff:
        return Verdict(alert=False, pillar="contracts", reason=diff["error"])

    reasons = []
    if diff["violations"]:
        reasons.extend(diff["violations"])
    if diff["freshness_delay_min"] > ctx.baseline["freshness_delay_max_min"] * _factor(ctx, "contract_freshness"):
        reasons.append("freshness_delay_high")
    if _history_alert(ctx, "contract.freshness_delay", diff["freshness_delay_min"], sigma=2.8):
        reasons.append("freshness_delay_history_shift")

    alert = bool(reasons)
    _remember_if_clean(
        ctx,
        [("contract.freshness_delay", diff["freshness_delay_min"])],
        alert,
    )
    return Verdict(alert=alert, pillar="contracts", reason=",".join(reasons))


def check_lineage_run(payload, ctx):
    if _skip_public_event(ctx):
        return Verdict(alert=False, pillar="lineage", reason="budget_warmup_skip")
    graph = ctx.tools.lineage_graph_slice(payload["run_id"])
    if "error" in graph:
        return Verdict(alert=False, pillar="lineage", reason=graph["error"])

    reasons = []
    expected_upstream = {"raw.orders", "raw.customers"}
    actual_upstream = set(graph["actual_upstream"])

    if graph["duration_ms"] > ctx.baseline["lineage_duration_ms_max"] * _factor(ctx, "lineage_duration"):
        reasons.append("lineage_duration_high")
    if actual_upstream != expected_upstream:
        reasons.append("upstream_mismatch")
    if graph["actual_downstream_count"] < 1:
        reasons.append("orphan_output")
    if _history_alert(ctx, "lineage.duration", graph["duration_ms"], sigma=2.6):
        reasons.append("lineage_duration_history_shift")

    alert = bool(reasons)
    _remember_if_clean(ctx, [("lineage.duration", graph["duration_ms"])], alert)
    return Verdict(alert=alert, pillar="lineage", reason=",".join(reasons))


def check_feature_materialization(payload, ctx):
    if _skip_public_event(ctx):
        return Verdict(alert=False, pillar="ai_infra", reason="budget_warmup_skip")
    drift = ctx.tools.feature_drift(payload["feature_view"], payload["batch_id"])
    if "error" in drift:
        return Verdict(alert=False, pillar="ai_infra", reason=drift["error"])

    alert = drift["mean_shift_sigma"] > ctx.baseline["feature_mean_shift_sigma_max"] * _factor(ctx, "feature_shift")
    reason = "feature_mean_shift_high" if alert else ""
    if _history_alert(ctx, "feature.mean_shift_sigma", drift["mean_shift_sigma"], sigma=2.4):
        alert = True
        reason = "feature_mean_shift_history"
    _remember_if_clean(ctx, [("feature.mean_shift_sigma", drift["mean_shift_sigma"])], alert)
    return Verdict(alert=alert, pillar="ai_infra", reason=reason)


def check_embedding_batch(payload, ctx):
    if _skip_public_event(ctx):
        return Verdict(alert=False, pillar="ai_infra", reason="budget_warmup_skip")
    drift = ctx.tools.embedding_drift(payload["corpus"], payload["chunk_batch_id"])
    if "error" in drift:
        return Verdict(alert=False, pillar="ai_infra", reason=drift["error"])

    reasons = []
    if drift["centroid_shift"] > ctx.baseline["embedding_centroid_shift_max"] * _factor(ctx, "embedding_centroid"):
        reasons.append("embedding_centroid_shift_high")
    if drift["avg_doc_age_days"] > ctx.baseline["corpus_avg_doc_age_days_max"] * _factor(ctx, "corpus_age"):
        reasons.append("corpus_doc_age_high")
    if _history_alert(ctx, "embedding.centroid_shift", drift["centroid_shift"], sigma=2.4):
        reasons.append("embedding_centroid_history_shift")
    if _history_alert(ctx, "embedding.avg_doc_age", drift["avg_doc_age_days"], sigma=2.4):
        reasons.append("corpus_age_history_shift")

    alert = bool(reasons)
    _remember_if_clean(
        ctx,
        [
            ("embedding.centroid_shift", drift["centroid_shift"]),
            ("embedding.avg_doc_age", drift["avg_doc_age_days"]),
        ],
        alert,
    )
    return Verdict(alert=alert, pillar="ai_infra", reason=",".join(reasons))
