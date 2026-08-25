from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


PARAM_KEYS: Tuple[str, ...] = ("E", "nu", "density", "yield_stress")
LOG_PARAM_KEYS = {"E", "density", "yield_stress"}
TRIM_FRACTION = 0.10


def _eval_space_value(key: str, value: float) -> float:
    if key in LOG_PARAM_KEYS:
        return math.log1p(max(0.0, float(value)))
    return float(value)


def build_param_sample_record(
    *,
    sample_id: str,
    action: str,
    material: str,
    object_name: str,
    gt_raw: Sequence[float],
    pred_raw: Sequence[float],
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "sample_id": sample_id,
        "action": action,
        "material": material,
        "object_name": object_name,
    }
    for idx, key in enumerate(PARAM_KEYS):
        gt_v = float(gt_raw[idx])
        pred_v = float(pred_raw[idx])
        gt_eval = _eval_space_value(key, gt_v)
        pred_eval = _eval_space_value(key, pred_v)
        abs_err = abs(pred_eval - gt_eval)
        sq_err = (pred_eval - gt_eval) ** 2
        mape = abs(pred_v - gt_v) / abs(gt_v) * 100.0 if abs(gt_v) > 1e-12 else float("nan")
        row[f"{key}_gt_raw"] = gt_v
        row[f"{key}_pred_raw"] = pred_v
        row[f"{key}_gt_eval"] = gt_eval
        row[f"{key}_pred_eval"] = pred_eval
        row[f"{key}_abs_err"] = abs_err
        row[f"{key}_sq_err"] = sq_err
        row[f"{key}_mape_percent"] = mape
    return row


def _group_key(row: Dict[str, object], group_fields: Sequence[str]) -> Tuple[object, ...]:
    return tuple(row.get(k, "") for k in group_fields)


def _is_valid_param_row(row: Dict[str, object], param: str) -> bool:
    gt_eval = float(row.get(f"{param}_gt_eval", float("nan")))
    pred_eval = float(row.get(f"{param}_pred_eval", float("nan")))
    if not (math.isfinite(gt_eval) and math.isfinite(pred_eval)):
        return False
    if param == "yield_stress":
        gt_raw = float(row.get("yield_stress_gt_raw", 0.0))
        return gt_raw > 0.0
    return True


def _finite_float(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _mean(vals: Sequence[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _median(vals: Sequence[float]) -> float:
    if not vals:
        return float("nan")
    xs = sorted(vals)
    n = len(xs)
    mid = n // 2
    if n % 2:
        return float(xs[mid])
    return float((xs[mid - 1] + xs[mid]) / 2.0)


def _percentile(vals: Sequence[float], q: float) -> float:
    if not vals:
        return float("nan")
    xs = sorted(vals)
    if len(xs) == 1:
        return float(xs[0])
    pos = max(0.0, min(1.0, float(q))) * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(xs[lo])
    frac = pos - lo
    return float(xs[lo] * (1.0 - frac) + xs[hi] * frac)


def _std(vals: Sequence[float]) -> float:
    if not vals:
        return float("nan")
    mu = _mean(vals)
    return float(math.sqrt(sum((x - mu) ** 2 for x in vals) / len(vals)))


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return float("nan")
    mx = _mean(xs)
    my = _mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0.0 or vy <= 0.0:
        return float("nan")
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return float(cov / math.sqrt(vx * vy))


def _rankdata(vals: Sequence[float]) -> List[float]:
    order = sorted(enumerate(vals), key=lambda kv: kv[1])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and order[j][1] == order[i][1]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k][0]] = rank
        i = j
    return ranks


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return float("nan")
    return _pearson(_rankdata(xs), _rankdata(ys))


def _trim_rows(rows: Sequence[Dict[str, object]], sort_key: str, trim_fraction: float = TRIM_FRACTION) -> List[Dict[str, object]]:
    if not rows:
        return []
    sorted_rows = sorted(rows, key=lambda r: _finite_float(r.get(sort_key)))
    k = int(math.floor(len(sorted_rows) * float(trim_fraction)))
    if k <= 0:
        return list(sorted_rows)
    if len(sorted_rows) - 2 * k <= 0:
        return list(sorted_rows)
    return list(sorted_rows[k:-k])


def _metric_summary(rows: Sequence[Dict[str, object]], param: str, prefix: str = "") -> Dict[str, object]:
    count = len(rows)
    gt_vals = [_finite_float(r.get(f"{param}_gt_eval")) for r in rows]
    pred_vals = [_finite_float(r.get(f"{param}_pred_eval")) for r in rows]
    abs_errs = [_finite_float(r.get(f"{param}_abs_err")) for r in rows]
    sq_errs = [_finite_float(r.get(f"{param}_sq_err")) for r in rows]
    signed_errs = [p - g for g, p in zip(gt_vals, pred_vals)]
    mapes = [
        _finite_float(r.get(f"{param}_mape_percent"))
        for r in rows
        if math.isfinite(_finite_float(r.get(f"{param}_mape_percent")))
    ]
    gt_mean = _mean(gt_vals)
    pred_mean = _mean(pred_vals)
    gt_std = _std(gt_vals)
    pred_std = _std(pred_vals)
    ss_res = sum((p - g) ** 2 for g, p in zip(gt_vals, pred_vals))
    ss_tot = sum((g - gt_mean) ** 2 for g in gt_vals)
    pearson_corr = _pearson(gt_vals, pred_vals)
    slope = float("nan")
    intercept = float("nan")
    if count >= 2 and math.isfinite(gt_std) and gt_std > 0.0:
        cov = sum((g - gt_mean) * (p - pred_mean) for g, p in zip(gt_vals, pred_vals)) / count
        slope = float(cov / (gt_std**2))
        intercept = float(pred_mean - slope * gt_mean)
    out: Dict[str, object] = {
        f"{prefix}count": count,
        f"{prefix}mae": _mean(abs_errs),
        f"{prefix}rmse": math.sqrt(_mean(sq_errs)) if sq_errs else float("nan"),
        f"{prefix}mape_percent": _mean(mapes),
        f"{prefix}median_ae": _median(abs_errs),
        f"{prefix}p90_ae": _percentile(abs_errs, 0.90),
        f"{prefix}bias": _mean(signed_errs),
        f"{prefix}gt_mean": gt_mean,
        f"{prefix}pred_mean": pred_mean,
        f"{prefix}gt_std": gt_std,
        f"{prefix}pred_std": pred_std,
        f"{prefix}pred_gt_std_ratio": (pred_std / gt_std) if math.isfinite(gt_std) and gt_std > 0.0 else float("nan"),
        f"{prefix}pearson_corr": pearson_corr,
        f"{prefix}spearman_corr": _spearman(gt_vals, pred_vals),
        f"{prefix}r2": (1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan"),
        f"{prefix}calibration_slope": slope,
        f"{prefix}calibration_intercept": intercept,
    }
    return out


def _param_weight_rows(records: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for param in PARAM_KEYS:
        valid_rows = [r for r in records if _is_valid_param_row(r, param)]
        gt_trimmed = _trim_rows(valid_rows, f"{param}_gt_eval")
        gt_vals = [_finite_float(r.get(f"{param}_gt_eval")) for r in gt_trimmed]
        lo = min(gt_vals) if gt_vals else float("nan")
        hi = max(gt_vals) if gt_vals else float("nan")
        span = hi - lo if math.isfinite(lo) and math.isfinite(hi) else float("nan")
        if not math.isfinite(span) or span <= 1e-12:
            span = 1.0
        rows.append(
            {
                "param": param,
                "metric_space": "log" if param in LOG_PARAM_KEYS else "raw",
                "valid_count": len(valid_rows),
                "trim_fraction": float(TRIM_FRACTION),
                "trimmed_count": len(gt_trimmed),
                "trimmed_gt_min": lo,
                "trimmed_gt_max": hi,
                "trimmed_gt_range": span,
                "range_norm_weight": 1.0 / span,
            }
        )
    return rows


def build_param_error_weights(records: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    return _param_weight_rows(list(records))


def _weight_map(weight_rows: Sequence[Mapping[str, object]]) -> Dict[str, float]:
    return {str(r["param"]): float(r["range_norm_weight"]) for r in weight_rows}


def enrich_param_records_with_composite(
    records: Iterable[Dict[str, object]],
    *,
    weight_rows: Sequence[Mapping[str, object]] | None = None,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    enriched = [dict(r) for r in records]
    weights = list(weight_rows) if weight_rows is not None else build_param_error_weights(enriched)
    wmap = _weight_map(weights)
    composite_rows: List[Dict[str, object]] = []
    composite_vals: List[float] = []
    for row in enriched:
        vals: List[float] = []
        out_row: Dict[str, object] = {
            "sample_id": row.get("sample_id", ""),
            "action": row.get("action", ""),
            "material": row.get("material", ""),
            "object_name": row.get("object_name", ""),
        }
        for param in PARAM_KEYS:
            norm = float("nan")
            if _is_valid_param_row(row, param):
                norm = _finite_float(row.get(f"{param}_abs_err")) * float(wmap.get(param, 1.0))
                if math.isfinite(norm):
                    vals.append(norm)
            row[f"{param}_range_norm_abs_err"] = norm
            out_row[f"{param}_range_norm_abs_err"] = norm
        comp = _mean(vals)
        row["param_composite_error"] = comp
        row["param_composite_error_valid_count"] = len(vals)
        out_row["param_composite_error"] = comp
        out_row["param_composite_error_valid_count"] = len(vals)
        composite_rows.append(out_row)
        if math.isfinite(comp):
            composite_vals.append(comp)
    summary = {
        "count": len(composite_vals),
        "mean": _mean(composite_vals),
        "median": _median(composite_vals),
        "p90": _percentile(composite_vals, 0.90),
        "rmse": math.sqrt(_mean([x * x for x in composite_vals])) if composite_vals else float("nan"),
        "min": min(composite_vals) if composite_vals else float("nan"),
        "max": max(composite_vals) if composite_vals else float("nan"),
    }
    return enriched, weights, composite_rows, summary


def aggregate_param_records(
    records: Iterable[Dict[str, object]],
    *,
    group_fields: Sequence[str],
) -> List[Dict[str, object]]:
    rec_list = list(records)
    grouped: Dict[Tuple[object, ...], List[Dict[str, object]]] = {}
    for row in rec_list:
        grouped.setdefault(_group_key(row, group_fields), []).append(row)

    out: List[Dict[str, object]] = []
    for key_vals, rows in sorted(grouped.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        for param in PARAM_KEYS:
            valid_rows = [r for r in rows if _is_valid_param_row(r, param)]
            gt_trimmed_rows = _trim_rows(valid_rows, f"{param}_gt_eval")
            err_trimmed_rows = _trim_rows(valid_rows, f"{param}_abs_err")
            gt_trim_vals = [_finite_float(r.get(f"{param}_gt_eval")) for r in gt_trimmed_rows]
            summary = _metric_summary(valid_rows, param)
            gt_trim_summary = _metric_summary(gt_trimmed_rows, param, prefix="gt_trim_")
            err_trim_summary = _metric_summary(err_trimmed_rows, param, prefix="err_trim_")
            out_row: Dict[str, object] = {
                "count": summary["count"],
                "param": param,
                "metric_space": "log" if param in LOG_PARAM_KEYS else "raw",
                "mae": summary["mae"],
                "rmse": summary["rmse"],
                "mape_percent": summary["mape_percent"],
                "trim_fraction": float(TRIM_FRACTION),
                "gt_trim_lower": min(gt_trim_vals) if gt_trim_vals else float("nan"),
                "gt_trim_upper": max(gt_trim_vals) if gt_trim_vals else float("nan"),
                "gt_trim_range": (
                    max(gt_trim_vals) - min(gt_trim_vals) if len(gt_trim_vals) >= 2 else float("nan")
                ),
            }
            out_row.update({k: v for k, v in summary.items() if k not in out_row})
            out_row.update(gt_trim_summary)
            out_row.update(err_trim_summary)
            for field, val in zip(group_fields, key_vals):
                out_row[field] = val
            out.append(out_row)
    return out


def summarize_param_composite_rows(rows: Iterable[Mapping[str, object]]) -> Dict[str, object]:
    vals = [
        _finite_float(r.get("param_composite_error"))
        for r in rows
        if math.isfinite(_finite_float(r.get("param_composite_error")))
    ]
    return {
        "count": len(vals),
        "mean": _mean(vals),
        "median": _median(vals),
        "p90": _percentile(vals, 0.90),
        "rmse": math.sqrt(_mean([x * x for x in vals])) if vals else float("nan"),
        "min": min(vals) if vals else float("nan"),
        "max": max(vals) if vals else float("nan"),
    }
