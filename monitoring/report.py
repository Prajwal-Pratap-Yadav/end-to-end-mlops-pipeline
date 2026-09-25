"""Render monitoring results as a self-contained HTML report (no external assets)."""

from __future__ import annotations

import html
import math
from typing import Any

_STYLE = """
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:2rem;color:#1f2933;max-width:1100px}
h1{font-size:1.5rem;margin-bottom:.25rem} h2{font-size:1.15rem;margin-top:2rem}
.meta{color:#52606d;margin-top:0}
.badge{display:inline-block;padding:.15rem .6rem;border-radius:999px;font-weight:600;font-size:.85rem}
.ok{background:#e3f9e5;color:#1f7a2e} .warn{background:#fff3c4;color:#8d5a00} .bad{background:#ffe3e3;color:#a61b1b}
table{border-collapse:collapse;width:100%;font-size:.9rem;margin-top:.5rem}
th,td{border-bottom:1px solid #e4e7eb;padding:.45rem .6rem;text-align:left;vertical-align:top}
th{background:#f5f7fa} td.num{text-align:right;font-variant-numeric:tabular-nums}
code{background:#f5f7fa;padding:.1rem .3rem;border-radius:4px}
"""


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return "-" if math.isnan(value) else f"{value:.{digits}f}"
    return html.escape(str(value))


def _fmt_p(value: Any) -> str:
    if isinstance(value, float) and not math.isnan(value) and value < 1e-4:
        return "&lt;0.0001"
    return _fmt(value)


def _summary(summary: dict[str, Any]) -> str:
    if "distribution" in summary:
        return ", ".join(f"{html.escape(k)}: {v:.0%}" for k, v in summary["distribution"].items())
    parts = [f"mean {_fmt(summary.get('mean'), 2)}", f"std {_fmt(summary.get('std'), 2)}"]
    if summary.get("missing_fraction"):
        parts.append(f"missing {summary['missing_fraction']:.1%}")
    return ", ".join(parts)


def _status_badge(status: str) -> str:
    css = {"ok": "ok", "insufficient_data": "warn", "no_model": "warn"}.get(status, "bad")
    return f'<span class="badge {css}">{html.escape(status.replace("_", " ").upper())}</span>'


def render_html(result: dict[str, Any]) -> str:
    """Render a monitoring result dictionary (see ``MonitoringResult.to_dict``) as HTML."""
    rows: list[str] = []
    drift = result.get("drift") or {}
    for feature in drift.get("features", []):
        flag = (
            '<span class="badge bad">DRIFT</span>'
            if feature["drift_detected"]
            else '<span class="badge ok">stable</span>'
        )
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(feature['feature'])}</code></td>"
            f"<td>{html.escape(feature['kind'])}</td>"
            f'<td class="num">{_fmt(feature["psi"])}</td>'
            f"<td>{html.escape(feature['test'])}</td>"
            f'<td class="num">{_fmt_p(feature["p_value"])}</td>'
            f"<td>{_summary(feature['reference'])}</td>"
            f"<td>{_summary(feature['current'])}</td>"
            f"<td>{flag}</td>"
            "</tr>"
        )

    prediction = drift.get("prediction_drift")
    prediction_html = (
        f"<p>Predicted churn probability PSI: <b>{_fmt(prediction['psi'])}</b> "
        f"(KS p-value {_fmt_p(prediction['p_value'])}); reference {_summary(prediction['reference'])} "
        f"&rarr; current {_summary(prediction['current'])}.</p>"
        if prediction
        else "<p>No prediction drift computed.</p>"
    )

    performance = result.get("performance") or {}
    perf_metrics = performance.get("metrics") or {}
    perf_rows = "".join(
        f'<tr><td>{html.escape(name)}</td><td class="num">'
        f'{int(value) if name == "n_samples" else _fmt(value)}</td></tr>'
        for name, value in sorted(perf_metrics.items())
    )
    perf_html = (
        f"<table><tr><th>Metric</th><th>Live value</th></tr>{perf_rows}</table>"
        if perf_rows
        else f"<p>Not enough labeled predictions yet ({performance.get('n_labeled', 0)} available).</p>"
    )

    retraining = result.get("retraining")
    retrain_html = (
        "<table>"
        + "".join(
            f"<tr><th>{html.escape(str(k))}</th><td>{_fmt(v)}</td></tr>"
            for k, v in retraining.items()
        )
        + "</table>"
        if retraining
        else "<p>Retraining was not triggered in this cycle.</p>"
    )
    reasons = result.get("retrain_reasons") or []
    reasons_html = (
        "<ul>" + "".join(f"<li>{html.escape(r)}</li>" for r in reasons) + "</ul>"
        if reasons
        else "<p>No retraining triggers fired.</p>"
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Model monitoring report</title><style>{_STYLE}</style></head>
<body>
<h1>Model monitoring report {_status_badge(result.get("status", "unknown"))}</h1>
<p class="meta">{html.escape(str(result.get("timestamp")))} &middot; model
<code>{html.escape(str(result.get("model_name")))} v{html.escape(str(result.get("model_version")))}</code>
&middot; window of {result.get("window_size", 0)} predictions vs {drift.get("n_reference", 0)} reference rows</p>

<h2>Data drift</h2>
<p>Drift share <b>{_fmt(drift.get("drift_share"), 2)}</b>
(threshold {_fmt((drift.get("thresholds") or {}).get("drift_share_threshold"), 2)}) &middot;
dataset drift: <b>{drift.get("dataset_drift", False)}</b>.
A feature drifts when PSI &ge; {_fmt((drift.get("thresholds") or {}).get("psi_threshold"), 2)}.</p>
<table><tr><th>Feature</th><th>Type</th><th>PSI</th><th>Test</th><th>p-value</th>
<th>Reference</th><th>Current</th><th>Status</th></tr>{"".join(rows)}</table>

<h2>Prediction drift</h2>{prediction_html}
<h2>Live performance (labeled feedback)</h2>{perf_html}
<h2>Retraining triggers</h2>{reasons_html}
<h2>Retraining outcome</h2>{retrain_html}
</body></html>
"""
