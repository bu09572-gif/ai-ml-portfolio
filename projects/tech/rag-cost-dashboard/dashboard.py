"""Cost dashboard for the filing Q&A system.

    pip install streamlit && streamlit run dashboard.py

Reads results/summary.json (measured in CI). The prices are inputs: set your own machine rate and,
to compare with a hosted API, your provider's price per million tokens.
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

S = json.loads((Path(__file__).parent / "results" / "summary.json").read_text())
t = pd.DataFrame(S["configs"])

st.set_page_config(page_title="RAG cost dashboard", layout="wide")
st.title("Filing Q&A: cost, speed and quality")
st.caption(f"{S['questions']} FinanceBench questions, measured on a 4 vCPU GitHub runner with free local models.")

c1, c2, c3, c4 = st.columns(4)
rate = c1.number_input("Machine $ per hour", 0.0, 10.0, 0.10, 0.01)
pin = c2.number_input("API $ per 1M input tokens", 0.0, 50.0, 0.10, 0.05)
pout = c3.number_input("API $ per 1M output tokens", 0.0, 100.0, 0.40, 0.05)
volume = c4.number_input("Questions per month", 100, 10_000_000, 100_000, 1000)

t["self_hosted_$_per_1k"] = t.cpu_seconds_mean * rate / 3600 * 1000
t["api_$_per_1k"] = (t.input_tokens_mean * pin + t.output_tokens_mean * pout) / 1e6 * 1000
t["self_hosted_$_per_month"] = t["self_hosted_$_per_1k"] * volume / 1000
t["machines_for_1_per_second"] = t.cpu_seconds_mean            # one question a second needs this many runners
models = st.multiselect("Models", sorted(t.model.unique()), default=sorted(t.model.unique()))
v = t[t.model.isin(models)]

st.subheader("Accuracy against cost")
st.scatter_chart(v, x="self_hosted_$_per_1k", y="numeric_accuracy", color="model", size="k")
best = v[v.numeric_accuracy >= st.slider("Minimum numeric accuracy", 0.0, 1.0, 0.0, 0.01)]
if len(best):
    b = best.sort_values("self_hosted_$_per_1k").iloc[0]
    st.success(f"Cheapest config at that accuracy: {b.model}, k={b.k}, compress={b.compress}: "
               f"{b.numeric_accuracy:.1%} accurate, ${b['self_hosted_$_per_1k']:.3f} per 1,000 questions, "
               f"p95 latency {b.latency_p95_s:.0f} s")
st.subheader("All configs")
st.dataframe(v.sort_values("numeric_accuracy", ascending=False), use_container_width=True)
