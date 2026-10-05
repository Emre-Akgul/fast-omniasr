"""Comparison logic between predicted and reference command lists."""


def _normalize_command(cmd: dict) -> tuple:
    channel = str(cmd.get("channel", "")).strip().lower()
    action = str(cmd.get("action", "")).strip().upper()
    params = cmd.get("params")
    # normalize numeric-ish params so 20 == 20.0 == "20"
    params_key = _normalize_value(params)
    return (channel, action, params_key)


def _normalize_value(value):
    if isinstance(value, dict):
        return tuple(sorted((k, _normalize_value(v)) for k, v in value.items()))
    if isinstance(value, list):
        return tuple(_normalize_value(v) for v in value)
    if isinstance(value, (int, float)):
        return round(float(value), 3)
    if isinstance(value, str):
        try:
            return round(float(value), 3)
        except ValueError:
            return value.strip().lower()
    return value


def compare(predicted: list, reference: list) -> dict:
    """Returns per-example match info at two granularities:

    - loose: predicted and reference agree on the multiset of (channel, action) pairs
    - strict: predicted and reference agree on the multiset of (channel, action, params)
    """
    pred_loose = sorted((c.get("channel", ""), c.get("action", "")) for c in predicted)
    ref_loose = sorted((c.get("channel", ""), c.get("action", "")) for c in reference)

    pred_strict = sorted(_normalize_command(c) for c in predicted)
    ref_strict = sorted(_normalize_command(c) for c in reference)

    return {
        "loose_match": pred_loose == ref_loose,
        "strict_match": pred_strict == ref_strict,
        "predicted_count": len(predicted),
        "reference_count": len(reference),
    }


def _sum_times(results: list, key: str):
    values = [r[key] for r in results if r.get(key) is not None]
    return sum(values) if values else None


def _avg(total, count):
    return total / count if total is not None and count else None


def summarize(results: list) -> dict:
    n = len(results)
    if n == 0:
        return {"total": 0}

    n_loose = sum(1 for r in results if r["comparison"]["loose_match"])
    n_strict = sum(1 for r in results if r["comparison"]["strict_match"])
    n_errors = sum(1 for r in results if r.get("error"))

    total_time_s = _sum_times(results, "total_time_s")
    stt_time_s = _sum_times(results, "stt_time_s")
    ttc_time_s = _sum_times(results, "ttc_time_s")

    return {
        "total": n,
        "errors": n_errors,
        # "num_correct" = strict match (channel, action, and params all agree)
        "num_correct": n_strict,
        "num_correct_loose": n_loose,
        "loose_accuracy": n_loose / n,
        "strict_accuracy": n_strict / n,
        "total_time_s": total_time_s,
        "stt_time_s": stt_time_s,
        "ttc_time_s": ttc_time_s,
        "avg_total_time_s": _avg(total_time_s, n),
        "avg_stt_time_s": _avg(stt_time_s, n),
        "avg_ttc_time_s": _avg(ttc_time_s, n),
    }
