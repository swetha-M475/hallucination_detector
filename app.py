import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import config
from utils.device import device_info
from llm.generator import get_generator
from llm.features import extract_features, aggregate_sentence_features, aggregate_response_features
from llm.evaluator import evaluate_consistency, jaccard_similarity
from llm.classifier import predict_risk, retrain_if_missing

# ── Auto-train classifier if no model exists yet ──────────────────────────────
retrain_if_missing()

st.set_page_config(page_title="Hallucination Detection", layout="wide", page_icon="🔍")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔍 Universal Source-Free Hallucination Detection")
st.markdown(
    "Detects hallucinations using **both** White-Box Internal Signals *(Entropy, SIVR, HARP)* "
    "**and** Black-Box Self-Consistency — combined into a single cumulative risk score."
)

# ── System Info Panel ─────────────────────────────────────────────────────────
with st.expander("⚙️ System Configuration & How It Works", expanded=True):
    col_a, col_b, col_c, col_d = st.columns(4)

    with col_a:
        st.markdown("#### 🤖 Model")
        st.markdown(f"**Model:** `{config.MODEL_NAME}`")
        st.markdown(f"**Device:** `{device_info()}`")
        st.markdown(f"**Main Tokens:** `{config.MAX_NEW_TOKENS}`")
        st.markdown(f"**Sample Tokens:** `{config.SAMPLE_MAX_TOKENS}`")

    with col_b:
        st.markdown("#### 🧠 White-Box Signals")
        st.markdown("**Entropy:** Token prediction uncertainty")
        st.markdown("**SIVR:** Cross-layer hidden state variance")
        st.markdown("**HARP:** SVD projection of hidden state")
        st.markdown(f"**SIVR Layers:** `{config.SIVR_LAYER_PAIRS}`")

    with col_c:
        st.markdown("#### 🔁 Self-Consistency")
        st.markdown(f"**Samples:** `{config.NUM_SAMPLES}`")
        st.markdown(f"**Main Temp:** `{config.TEMPERATURE_MAIN}` *(greedy)*")
        st.markdown(f"**Sample Temp:** `{config.TEMPERATURE_SAMPLES}` *(creative)*")
        st.markdown("**Metric:** Jaccard Word-Overlap")

    with col_d:
        st.markdown("#### 🎯 Cumulative Risk")
        st.markdown("**Formula:** `0.5 × (White-Box Risk) + 0.5 × (SC Risk)`")
        st.markdown(f"**Low Threshold:** `{config.RISK_PROB_LOW}`")
        st.markdown(f"**High Threshold:** `{config.RISK_PROB_HIGH}`")
        st.markdown("**Scorer:** No external KB needed")

    st.markdown("---")
    st.markdown("""
    **Pipeline:**
    1. 🧠 **White-Box Pass** — Generate with `output_hidden_states=True` to capture Entropy, SIVR, HARP per token.
    2. 🔁 **Self-Consistency Pass** — Generate stochastic samples and compare via Jaccard similarity.
    3. 🎯 **Cumulative Risk** — Average both scores into a single final hallucination risk score.
    """)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Generation Settings")
max_tokens = st.sidebar.slider("Max New Tokens (main)", 10, 150, 50)
num_samples = st.sidebar.slider("Number of SC Samples", 1, 5, config.NUM_SAMPLES)
sample_tokens = st.sidebar.slider("Max Tokens per Sample", 10, 100, config.SAMPLE_MAX_TOKENS)
wb_weight = st.sidebar.slider("White-Box Weight", 0.0, 1.0, 0.5, 0.05,
                               help="Weight for the white-box signal in the cumulative score. "
                                    "1 - this value = self-consistency weight.")

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Model:** `{config.MODEL_NAME}`")
st.sidebar.markdown(f"**Device:** `{device_info()}`")
st.sidebar.markdown(f"**SIVR Layer Pairs:** `{config.SIVR_LAYER_PAIRS}`")

# ── Input ─────────────────────────────────────────────────────────────────────
question = st.text_area("💬 Enter your question:", "What is the capital of France?", height=100)

if st.button("🚀 Generate & Analyze", type="primary"):

    with st.spinner("⏳ Step 1/2 — White-box pass (capturing Entropy, SIVR, HARP)…"):
        generator = get_generator()
        unembedding = generator.get_unembedding_matrix()
        full_output = generator.generate_full(question, max_new_tokens=max_tokens)
        token_features = extract_features(full_output, unembedding)
        sentences = aggregate_sentence_features(token_features)
        agg = aggregate_response_features(token_features)

    with st.spinner(f"⏳ Step 2/2 — Self-consistency pass ({num_samples} samples)…"):
        main_answer, samples = generator.generate_with_samples(
            question,
            max_new_tokens=max_tokens,
            num_samples=num_samples,
            sample_max_tokens=sample_tokens,
        )
        sc_risk = evaluate_consistency(question, main_answer, samples)

    # ── Compute cumulative risk ────────────────────────────────────────────
    # Normalised white-box sub-signals (always computed for display)
    norm_entropy  = min(agg["entropy"] / 4.0, 1.0)
    norm_sivr     = min(agg["sivr"] / 0.5, 1.0)
    norm_harp     = min(agg["harp"] / 50.0, 1.0)   # wider divisor — HARP is an unbounded norm
    norm_uncert   = 1.0 - agg["confidence"]

    # White-box risk: prefer the trained classifier; fall back to the manual formula
    clf_prob = predict_risk(agg)   # returns float | None
    if clf_prob is not None:
        wb_risk = clf_prob
        wb_source = "Classifier (LogReg)"
    else:
        wb_risk = float(np.mean([norm_entropy, norm_sivr, norm_harp, norm_uncert]))
        wb_source = "Manual formula (no model)"

    sc_weight = 1.0 - wb_weight
    cumulative_risk = wb_weight * wb_risk + sc_weight * sc_risk

    # ── Results ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Analysis Results")

    # ── Top metrics row ────────────────────────────────────────────────────
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("🎯 Cumulative Risk", f"{cumulative_risk:.0%}")
    m2.metric("🧠 White-Box Risk", f"{wb_risk:.0%}", help=f"Source: {wb_source}")
    m3.metric("🔁 Self-Consistency Risk", f"{sc_risk:.0%}")
    m4.metric("📈 Avg Entropy", f"{agg['entropy']:.3f} bits")
    m5.metric("🔀 Avg SIVR", f"{agg['sivr']:.4f}")
    m6.metric("📐 Avg HARP", f"{agg['harp']:.2f}")
    st.caption(f"🤖 White-Box source: **{wb_source}**")

    st.markdown("---")

    col1, col2 = st.columns([3, 2])

    with col1:
        # ── Sentence-level highlighted answer ─────────────────────────────
        st.markdown("### ✍️ Generated Answer *(sentence-level entropy highlight)*")
        html = "<div style='line-height:1.8; font-size:17px; padding:15px; border-radius:8px; background:#1a1a2e; color:#e0e0e0;'>"
        for s in sentences:
            intensity = min(1.0, max(0.0, (s["entropy"] - 0.5) / 3.0))
            r = int(255 * intensity)
            g = int(200 * (1 - intensity))
            tooltip = (f"Entropy: {s['entropy']:.3f} | Confidence: {s['confidence']:.3f} | "
                       f"SIVR: {s['sivr']:.4f} | HARP: {s['harp']:.2f}")
            html += (f"<span title='{tooltip}' style='background:rgba({r},{g},30,0.35); "
                     f"padding:2px 5px; border-radius:4px; margin:1px;'>{s['text']}</span> ")
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)
        st.caption("💡 Hover over sentences to see per-sentence signal values.")

        # ── Self-Consistency samples ───────────────────────────────────────
        st.markdown(f"### 🎲 Self-Consistency Samples *(temp = {config.TEMPERATURE_SAMPLES})*")
        for i, sample in enumerate(samples):
            sim = jaccard_similarity(main_answer, sample)
            if sim >= 0.4:
                color, label, icon = "#2d6a4f", "Consistent", "✅"
            elif sim >= 0.2:
                color, label, icon = "#b5640a", "Uncertain", "⚠️"
            else:
                color, label, icon = "#9b2226", "Contradicts", "❌"
            st.markdown(
                f"**Sample {i+1}** &nbsp; {icon} "
                f"<span style='color:{color}; font-weight:bold'>{label} ({sim:.0%} overlap)</span>"
                f"<br><span style='opacity:0.85'>{sample}</span>",
                unsafe_allow_html=True,
            )
            st.markdown("")

    with col2:
        # ── Cumulative gauge ───────────────────────────────────────────────
        st.markdown("### 🎯 Cumulative Hallucination Risk")
        fig_risk = go.Figure(go.Indicator(
            mode="gauge+number",
            value=cumulative_risk * 100,
            title={"text": "Combined Risk (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkred" if cumulative_risk > config.RISK_PROB_HIGH
                        else "darkorange" if cumulative_risk > config.RISK_PROB_LOW else "green"},
                "steps": [
                    {"range": [0, 33], "color": "#d8f3dc"},
                    {"range": [33, 66], "color": "#fff3cd"},
                    {"range": [66, 100], "color": "#fde8e8"},
                ],
            },
        ))
        fig_risk.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_risk, use_container_width=True)

        if cumulative_risk > config.RISK_PROB_HIGH:
            st.error("🚨 **HIGH RISK** — Likely hallucinating.")
        elif cumulative_risk > config.RISK_PROB_LOW:
            st.warning("⚠️ **MEDIUM RISK** — Treat with caution.")
        else:
            st.success("✅ **LOW RISK** — Model appears consistent and confident.")

        # ── Radar chart ────────────────────────────────────────────────────
        st.markdown("#### 📡 White-Box Signal Breakdown")
        fig_radar = go.Figure(go.Scatterpolar(
            r=[norm_entropy, norm_sivr, norm_harp, norm_uncert, sc_risk],
            theta=["Entropy", "SIVR", "HARP", "Uncertainty\n(1-Conf)", "Self-Consistency\nRisk"],
            fill="toself",
            marker=dict(color="rgba(255,80,80,0.8)"),
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=False,
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # ── Full cumulative score table ────────────────────────────────────────
    st.markdown("---")
    st.subheader("📋 Cumulative Score Summary")
    summary_df = pd.DataFrame([
        {"Signal": "🔴 Entropy (normalised)",        "Raw Value": f"{agg['entropy']:.4f} bits",  "Normalised": f"{norm_entropy:.4f}", "Contributes To": "White-Box features"},
        {"Signal": "🟠 SIVR (cross-layer var)",       "Raw Value": f"{agg['sivr']:.4f}",          "Normalised": f"{norm_sivr:.4f}",    "Contributes To": "White-Box features"},
        {"Signal": "🟡 HARP (SVD projection)",        "Raw Value": f"{agg['harp']:.4f}",          "Normalised": f"{norm_harp:.4f}",    "Contributes To": "White-Box features"},
        {"Signal": "🔵 Uncertainty (1-Conf)",         "Raw Value": f"{1-agg['confidence']:.4f}",  "Normalised": f"{norm_uncert:.4f}",  "Contributes To": "White-Box features"},
        {"Signal": "🟣 Self-Consistency (Jaccard)",   "Raw Value": f"1 - {1-sc_risk:.4f}",        "Normalised": f"{sc_risk:.4f}",      "Contributes To": "SC Risk"},
        {"Signal": f"⚫ White-Box Risk ({wb_source})","Raw Value": "—",                           "Normalised": f"{wb_risk:.4f}",      "Contributes To": "Cumulative"},
        {"Signal": "⚫ SC Risk",                       "Raw Value": "—",                           "Normalised": f"{sc_risk:.4f}",      "Contributes To": "Cumulative"},
        {"Signal": "🎯 CUMULATIVE RISK",              "Raw Value": f"{wb_weight:.0%}×WB + {sc_weight:.0%}×SC", "Normalised": f"{cumulative_risk:.4f}", "Contributes To": "Final Score"},
    ])
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # ── Token-level detail ─────────────────────────────────────────────────
    with st.expander("🔬 Token-Level Signals"):
        token_rows = [{
            "Token": t["token"],
            "Entropy": round(t["entropy"], 4),
            "Confidence": round(t["confidence"], 4),
            "SIVR": round(t["sivr"], 4),
            "HARP": round(t["harp"], 2),
        } for t in token_features]
        st.dataframe(pd.DataFrame(token_rows), use_container_width=True)

    with st.expander("📝 Sentence-Level Signals"):
        sent_rows = [{
            "Sentence": s["text"],
            "Entropy": round(s["entropy"], 4),
            "Confidence": round(s["confidence"], 4),
            "SIVR": round(s["sivr"], 4),
            "HARP": round(s["harp"], 2),
        } for s in sentences]
        st.dataframe(pd.DataFrame(sent_rows), use_container_width=True)
