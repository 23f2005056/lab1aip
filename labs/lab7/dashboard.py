#!/usr/bin/env python3
"""Lab 7 — Observability Dashboard.

Run:

    streamlit run labs/lab7/dashboard.py

Reads JSONL traces from .aip_traces/.

This dashboard intentionally avoids Streamlit dataframe/table/chart APIs
because the current Windows environment blocks the PyArrow DLL.
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# PROJECT SETUP
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip.config import settings  # noqa: E402


st.set_page_config(
    page_title="Aurora Assistant — Ops",
    layout="wide",
)

st.title("Aurora Policy Assistant — operations")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def html_table(headers, rows):
    """Render a simple HTML table without Streamlit dataframe APIs."""

    header_html = "".join(
        f"<th>{html.escape(str(h))}</th>"
        for h in headers
    )

    body_html = ""

    for row in rows:
        body_html += "<tr>"

        for value in row:
            body_html += (
                f"<td>{html.escape(str(value))}</td>"
            )

        body_html += "</tr>"

    return f"""
    <table style="
        width:100%;
        border-collapse:collapse;
        margin-top:10px;
        margin-bottom:20px;
    ">
        <thead>
            <tr>
                {header_html}
            </tr>
        </thead>
        <tbody>
            {body_html}
        </tbody>
    </table>
    """


def percentile(values, q):
    """Calculate percentile safely."""

    if not values:
        return None

    return float(
        pd.Series(values).quantile(q)
    )


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ============================================================
# LOAD TRACE FILES
# ============================================================

runs = sorted(
    settings.trace_dir.glob("*.jsonl"),
    reverse=True,
)

if not runs:

    st.info(
        f"No traces yet in {settings.trace_dir}. "
        "Run some queries first."
    )

    st.stop()


# ============================================================
# RUN SELECTION
# ============================================================

chosen = st.sidebar.multiselect(
    "runs",
    [p.stem for p in runs],
    default=[runs[0].stem],
)


# ============================================================
# READ JSONL
# ============================================================

rows = []

for path in runs:

    if path.stem not in chosen:
        continue

    with path.open(
        encoding="utf-8"
    ) as file:

        for line in file:

            if not line.strip():
                continue

            try:
                rows.append(
                    json.loads(line)
                )
            except json.JSONDecodeError:
                continue


if not rows:

    st.warning(
        "No trace spans found for the selected runs."
    )

    st.stop()


# ============================================================
# DATAFRAME
#
# Pandas itself is fine.
# The problem is Streamlit's rendering of DataFrames,
# not pandas.
# ============================================================

df = pd.DataFrame(rows)


# ============================================================
# NORMALIZE COLUMNS
# ============================================================

if "ts" in df.columns:

    df["ts"] = pd.to_datetime(
        df["ts"],
        unit="s",
        errors="coerce",
    )


if "duration_ms" not in df.columns:
    df["duration_ms"] = 0.0


df["duration_ms"] = pd.to_numeric(
    df["duration_ms"],
    errors="coerce",
).fillna(0.0)


if "cost_usd" not in df.columns:
    df["cost_usd"] = 0.0


df["cost_usd"] = pd.to_numeric(
    df["cost_usd"],
    errors="coerce",
).fillna(0.0)


if "name" not in df.columns:
    df["name"] = "unknown"


# ============================================================
# OVERVIEW METRICS
# ============================================================

total_cost = float(
    df["cost_usd"].sum()
)


# Current service uses llm.stream.
# Keep llm.call for compatibility with older traces.

llm_names = {
    "llm.call",
    "llm.stream",
}

llm = df[
    df["name"].isin(llm_names)
]


if "cached" in llm.columns and len(llm):

    cache_hit_rate = (
        llm["cached"]
        .fillna(False)
        .astype(bool)
        .mean()
    )

else:

    cache_hit_rate = 0.0


if "status" in df.columns:

    errors_count = int(
        (
            df["status"]
            .astype(str)
            .str.lower()
            == "error"
        ).sum()
    )

else:

    errors_count = 0


# ============================================================
# TOP METRICS
# ============================================================

c = st.columns(5)


c[0].metric(
    "spans",
    len(df),
)


c[1].metric(
    "total cost",
    f"${total_cost:.4f}",
)


c[2].metric(
    "model calls",
    len(llm),
)


c[3].metric(
    "cache hit rate",
    f"{cache_hit_rate:.0%}",
)


c[4].metric(
    "errors",
    errors_count,
)


# ============================================================
# C3 — LATENCY BY STAGE
# ============================================================

st.subheader("Latency by stage")


stage_rows = []


for name, group in df.groupby("name"):

    durations = (
        pd.to_numeric(
            group["duration_ms"],
            errors="coerce",
        )
        .dropna()
        .tolist()
    )

    if not durations:
        continue

    stage_rows.append(
        [
            name,
            len(durations),
            f"{percentile(durations, 0.50):.2f}",
            f"{percentile(durations, 0.95):.2f}",
            f"{sum(durations):.2f}",
        ]
    )


stage_rows.sort(
    key=lambda x: float(x[4]),
    reverse=True,
)


st.markdown(
    html_table(
        [
            "Stage",
            "Count",
            "P50 (ms)",
            "P95 (ms)",
            "Total (ms)",
        ],
        stage_rows,
    ),
    unsafe_allow_html=True,
)


# ============================================================
# C3 — LATENCY BY STAGE OVER TIME
# ============================================================

st.subheader("Latency by stage over time")


st.caption(
    "Trace-stage samples shown below. "
    "This avoids Streamlit's Arrow-based chart APIs."
)


time_rows = []


for _, row in df.sort_values("ts").iterrows():

    timestamp = row.get("ts")

    if pd.isna(timestamp):
        continue

    time_rows.append(
        [
            str(timestamp),
            str(row.get("name", "")),
            f"{safe_float(row.get('duration_ms')):.2f}",
        ]
    )


if time_rows:

    # Show the latest 50 samples to keep the dashboard readable.
    display_rows = time_rows[-50:]

    st.markdown(
        html_table(
            [
                "Timestamp",
                "Stage",
                "Latency (ms)",
            ],
            display_rows,
        ),
        unsafe_allow_html=True,
    )

else:

    st.info(
        "No timestamped latency data available."
    )


# ============================================================
# C3 — COST OVER TIME
# ============================================================

st.subheader("Cumulative cost over time")


cost_rows = []


if "ts" in df.columns:

    cost_df = df.sort_values("ts").copy()

    cumulative = 0.0

    for _, row in cost_df.iterrows():

        timestamp = row.get("ts")

        if pd.isna(timestamp):
            continue

        cumulative += safe_float(
            row.get("cost_usd")
        )

        cost_rows.append(
            [
                str(timestamp),
                f"${cumulative:.6f}",
            ]
        )


if cost_rows:

    st.markdown(
        html_table(
            [
                "Timestamp",
                "Cumulative Cost",
            ],
            cost_rows[-50:],
        ),
        unsafe_allow_html=True,
    )

else:

    st.info(
        "No timestamped cost data available."
    )


# ============================================================
# CACHE PERFORMANCE
# ============================================================

st.subheader("Cache performance")


exact_hits = 0
semantic_hits = 0
misses = 0


# First try to use HTTP request spans,
# because these represent actual requests.

request_names = {
    "http.ask",
    "http.ask.stream",
}


request_spans = df[
    df["name"].isin(request_names)
]


if len(request_spans):

    for _, row in request_spans.iterrows():

        cached = row.get(
            "cached",
            False,
        )

        cache_type = str(
            row.get(
                "cache_type",
                "",
            )
        ).lower()

        if cached:

            if "semantic" in cache_type:
                semantic_hits += 1

            elif "exact" in cache_type:
                exact_hits += 1

            else:

                # Try the trace name / metadata if available.
                if "semantic" in str(row).lower():
                    semantic_hits += 1
                else:
                    exact_hits += 1

        else:

            misses += 1


cache_total = (
    exact_hits
    + semantic_hits
    + misses
)


cc = st.columns(4)


cc[0].metric(
    "exact cache hits",
    exact_hits,
)


cc[1].metric(
    "semantic cache hits",
    semantic_hits,
)


cc[2].metric(
    "cache misses",
    misses,
)


if cache_total:

    hit_rate = (
        (exact_hits + semantic_hits)
        / cache_total
    )

else:

    hit_rate = 0.0


cc[3].metric(
    "cache hit rate",
    f"{hit_rate:.0%}",
)


# ============================================================
# STREAMING PERFORMANCE
# ============================================================

st.subheader("Streaming performance")


stream_spans = df[
    df["name"] == "llm.stream"
]


if len(stream_spans):

    stream_values = (
        pd.to_numeric(
            stream_spans["duration_ms"],
            errors="coerce",
        )
        .dropna()
        .tolist()
    )


    sc = st.columns(4)


    sc[0].metric(
        "stream calls",
        len(stream_values),
    )


    sc[1].metric(
        "stream P50",
        f"{percentile(stream_values, 0.50):.0f} ms",
    )


    sc[2].metric(
        "stream P95",
        f"{percentile(stream_values, 0.95):.0f} ms",
    )


    sc[3].metric(
        "stream max",
        f"{max(stream_values):.0f} ms",
    )


else:

    st.info(
        "No llm.stream spans found."
    )


# ============================================================
# ERRORS
# ============================================================

st.subheader("Errors")


if "status" in df.columns:

    errs = df[
        df["status"]
        .astype(str)
        .str.lower()
        == "error"
    ]

else:

    errs = pd.DataFrame()


if len(errs):

    error_rows = []

    for _, row in errs.iterrows():

        error_rows.append(
            [
                str(row.get("ts", "")),
                str(row.get("name", "")),
                str(row.get("error", "")),
            ]
        )


    st.markdown(
        html_table(
            [
                "Timestamp",
                "Stage",
                "Error",
            ],
            error_rows,
        ),
        unsafe_allow_html=True,
    )

else:

    st.success(
        "No errors recorded in the selected traces."
    )


# ============================================================
# ERROR RATE BY TYPE
# ============================================================

st.subheader("Error rate by type")


if len(errs):

    error_counts = (
        errs["name"]
        .value_counts()
    )


    error_rows = []


    for error_type, count in error_counts.items():

        rate = (
            count / len(df)
            if len(df)
            else 0
        )


        error_rows.append(
            [
                error_type,
                int(count),
                f"{rate:.2%}",
            ]
        )


    st.markdown(
        html_table(
            [
                "Error type",
                "Count",
                "Rate",
            ],
            error_rows,
        ),
        unsafe_allow_html=True,
    )

else:

    st.write(
        "No errors recorded."
    )


# ============================================================
# TOOL CALLS
# ============================================================

st.subheader("Tool calls")


tool_spans = df[
    df["name"]
    .astype(str)
    .str.contains(
        "tool",
        case=False,
        na=False,
    )
]


if len(tool_spans):

    total_tools = len(tool_spans)


    if "status" in tool_spans.columns:

        denied = int(
            (
                tool_spans["status"]
                .astype(str)
                .str.lower()
                .isin(
                    [
                        "denied",
                        "blocked",
                    ]
                )
            ).sum()
        )

    else:

        denied = 0


    allowed = total_tools - denied


    tc = st.columns(4)


    tc[0].metric(
        "tool calls",
        total_tools,
    )


    tc[1].metric(
        "allowed",
        allowed,
    )


    tc[2].metric(
        "denied",
        denied,
    )


    tc[3].metric(
        "denial rate",
        f"{denied / total_tools:.0%}",
    )


else:

    st.info(
        "No tool-call spans found in the selected traces."
    )


# ============================================================
# C4 — ALERT
# ============================================================

st.subheader("Alert")


# Lab 7 streaming SLO:
# TTFT <= 1,500 ms

TTFT_SLO_MS = 1500.0


ttft_values = []


if len(stream_spans):

    if "ttft_ms" in stream_spans.columns:

        ttft_values = (
            pd.to_numeric(
                stream_spans["ttft_ms"],
                errors="coerce",
            )
            .dropna()
            .tolist()
        )


    # If explicit TTFT is not present in traces,
    # use stream duration as a conservative fallback.
    if not ttft_values:

        ttft_values = (
            pd.to_numeric(
                stream_spans["duration_ms"],
                errors="coerce",
            )
            .dropna()
            .tolist()
        )


if ttft_values:

    ttft_p95 = percentile(
        ttft_values,
        0.95,
    )


    if ttft_p95 > TTFT_SLO_MS:

        st.error(
            f"ALERT: streaming TTFT P95 is "
            f"{ttft_p95:.0f} ms, above the "
            f"{TTFT_SLO_MS:.0f} ms SLO."
        )


        st.write(
            "Action: investigate model-generation "
            "latency and the slowest streaming traces."
        )

    else:

        st.success(
            f"OK: streaming TTFT P95 is "
            f"{ttft_p95:.0f} ms, within the "
            f"{TTFT_SLO_MS:.0f} ms SLO."
        )

else:

    st.info(
        "No streaming latency data available "
        "for the alert."
    )


# ============================================================
# TRACE DETAILS
# ============================================================

st.subheader("Trace details")


st.write(
    f"Selected runs: {len(chosen)}"
)


st.write(
    f"Trace directory: `{settings.trace_dir}`"
)


st.write(
    f"Total trace records loaded: {len(df)}"
)