"""
app.py
-------
Streamlit web app — AI Voice Quality Evaluation System
Deploy free at share.streamlit.io
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import time
from evaluator import evaluate_transcript, grade, VoiceEvalResult
from report import generate_report, export_csv
import os, tempfile

st.set_page_config(
    page_title="AI Voice Quality Evaluator",
    page_icon="🎙️",
    layout="wide"
)

st.markdown("""
<style>
.score-box {
    background: #161b22;
    border-radius: 10px;
    padding: 14px;
    text-align: center;
    border: 1px solid #21262d;
}
.score-num  { font-size: 32px; font-weight: bold; }
.score-lbl  { font-size: 12px; color: #8b949e; margin-top: 4px; }
.grade-excellent { color: #3fb950; }
.grade-good      { color: #58a6ff; }
.grade-fair      { color: #d29922; }
.grade-poor      { color: #f85149; }
.grade-critical  { color: #ff0000; }
.issue-box  { background: #2d1a1a; border-left: 3px solid #f85149; padding: 8px 12px; border-radius: 4px; margin: 4px 0; font-size: 13px; }
.suggest-box{ background: #1a2d1a; border-left: 3px solid #3fb950; padding: 8px 12px; border-radius: 4px; margin: 4px 0; font-size: 13px; }
</style>
""", unsafe_allow_html=True)

STYLE = {
    "bg":      "#0d1117",
    "surface": "#161b22",
    "accent":  "#58a6ff",
    "green":   "#3fb950",
    "red":     "#f85149",
    "yellow":  "#d29922",
    "purple":  "#bc8cff",
    "text":    "#e6edf3",
    "muted":   "#8b949e",
    "grid":    "#21262d",
}

DEMO_SAMPLES = {
    "Good Sample — Professional AI Assistant": {
        "transcript": "Hello and welcome! I'm your AI assistant, here to help you today. Our platform enables you to create personalized, high-quality video messages at scale. We support multiple languages and voice styles to suit your communication needs. Thank you for choosing our service, and please let me know how I can assist you further.",
        "reference": "Hello and welcome. I am your AI assistant here to help you today. Our platform enables you to create personalized high quality video messages at scale. We support multiple languages and voice styles. Thank you for choosing our service.",
        "duration": 22.0,
        "confidence": 0.95
    },
    "Poor Sample — Hallucination & Filler Words": {
        "transcript": "Um, so basically, uh, our company was founded in, like, 1847 by Dr. James Wellington III in Edinburgh, Scotland. You know, we have over 50,000 employees worldwide. I mean, basically we're the largest AI company in, uh, the entire universe. So yeah, like, we can help you with, um, everything.",
        "reference": "Our company was founded in 2020. We have 200 employees. We are an AI video generation company.",
        "duration": 20.0,
        "confidence": 0.78
    },
    "Medium Sample — Fast Speech": {
        "transcript": "WelcometoourAIplatformwherewehelp businesses create amazing video content using cutting edge artificial intelligence technology our models are trained on millions of hours of human speech to produce natural sounding voice outputs that engage your audience effectively.",
        "reference": "Welcome to our AI platform where we help businesses create video content using artificial intelligence.",
        "duration": 8.0,
        "confidence": 0.88
    },
    "Negative Tone Sample": {
        "transcript": "Unfortunately I cannot process your request at this time. There is a critical error in the system and we are unable to help you. I'm sorry but this feature is broken and we don't know when it will be fixed. This is a problem we cannot resolve.",
        "reference": "We are experiencing technical difficulties. Please try again later.",
        "duration": 16.0,
        "confidence": 0.92
    }
}

def score_color(score):
    if score >= 90: return STYLE["green"]
    if score >= 75: return STYLE["accent"]
    if score >= 60: return STYLE["yellow"]
    return STYLE["red"]

def render_radar(result: VoiceEvalResult):
    categories = ['Transcription', 'Clarity', 'Hallucination\nDetection', 'Tone', 'Fluency']
    values = [
        result.transcription_score,
        result.clarity_score,
        result.hallucination_score,
        result.tone_score,
        result.fluency_score
    ]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    values_plot = values + values[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(STYLE["bg"])
    ax.set_facecolor(STYLE["surface"])
    ax.plot(angles, values_plot, color=STYLE["accent"], lw=2)
    ax.fill(angles, values_plot, color=STYLE["accent"], alpha=0.2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color=STYLE["text"], size=9)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25", "50", "75", "100"], color=STYLE["muted"], size=7)
    ax.grid(color=STYLE["grid"], linewidth=0.5)
    ax.spines['polar'].set_color(STYLE["grid"])
    plt.tight_layout()
    return fig

def render_bar_chart(result: VoiceEvalResult):
    dims = ["Transcription", "Clarity", "Hallucination", "Tone", "Fluency"]
    scores = [
        result.transcription_score,
        result.clarity_score,
        result.hallucination_score,
        result.tone_score,
        result.fluency_score
    ]
    colors = [score_color(s) for s in scores]

    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor(STYLE["bg"])
    ax.set_facecolor(STYLE["surface"])
    bars = ax.barh(dims, scores, color=colors, alpha=0.85, height=0.5)
    ax.set_xlim(0, 100)
    ax.axvline(75, color=STYLE["muted"], lw=0.8, ls="--", alpha=0.5)
    for bar, score in zip(bars, scores):
        ax.text(score + 1, bar.get_y() + bar.get_height()/2,
                f"{score:.0f}", va='center', color=STYLE["text"], fontsize=10)
    ax.tick_params(colors=STYLE["muted"])
    for spine in ax.spines.values(): spine.set_color(STYLE["grid"])
    ax.grid(axis='x', color=STYLE["grid"], linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    return fig

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🎙️ AI Voice Quality Evaluation System")
st.markdown("*Transcription accuracy · Hallucination detection · Clarity · Tone · Fluency*")
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    input_mode = st.radio("Input Mode", ["Use Demo Sample", "Paste Transcript", "Upload Audio"])
    st.markdown("---")
    st.markdown("**Built by [Kshitij Mishra](https://kshitij.info)**")
    st.markdown("📂 [GitHub](https://github.com/KshitijMishra1418)")
    st.markdown("📊 [Crypto Project](https://crypto-analysis-km.streamlit.app)")

# ── Input Section ─────────────────────────────────────────────────────────────
transcript = ""
reference  = ""
duration   = 0.0
confidence = 0.90
filename   = "sample"

if input_mode == "Use Demo Sample":
    sample_name = st.selectbox("Choose a demo sample:", list(DEMO_SAMPLES.keys()))
    sample = DEMO_SAMPLES[sample_name]
    transcript = sample["transcript"]
    reference  = sample["reference"]
    duration   = sample["duration"]
    confidence = sample["confidence"]
    filename   = sample_name
    with st.expander("View sample transcript"):
        st.text(transcript)
    with st.expander("View reference text"):
        st.text(reference)

elif input_mode == "Paste Transcript":
    col1, col2 = st.columns(2)
    with col1:
        transcript = st.text_area("AI Voice Transcript", height=150,
                                   placeholder="Paste the text output from your AI voice model...")
        duration   = st.number_input("Audio duration (seconds)", min_value=0.0, value=15.0)
        confidence = st.slider("Transcription confidence", 0.0, 1.0, 0.90)
    with col2:
        reference  = st.text_area("Reference / Expected Text (optional)", height=150,
                                   placeholder="Paste the original script or expected output...")
    filename = "manual_input"

elif input_mode == "Upload Audio":
    st.info("Upload an audio file — Whisper will transcribe it automatically.")
    audio_file = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a", "ogg", "flac"])
    reference  = st.text_area("Reference / Expected Text (optional)", height=100)

    if audio_file:
        st.audio(audio_file)
        filename = audio_file.name
        if st.button("Transcribe with Whisper"):
            with st.spinner("Transcribing with Whisper (this may take 30-60 seconds)..."):
                try:
                    from transcriber import transcribe_bytes
                    suffix = "." + audio_file.name.split(".")[-1]
                    result_t = transcribe_bytes(audio_file.read(), suffix=suffix)
                    if result_t.get("error"):
                        st.error(result_t["error"])
                    else:
                        transcript = result_t["text"]
                        duration   = result_t["duration"]
                        confidence = result_t["confidence"]
                        st.success(f"Transcribed! Confidence: {confidence:.0%}, Duration: {duration:.1f}s")
                        st.text_area("Transcript", transcript, height=120)
                except Exception as e:
                    st.error(f"Transcription error: {e}")

st.markdown("---")

# ── Evaluate Button ───────────────────────────────────────────────────────────
run = st.button("🔍 Evaluate Voice Quality", use_container_width=True)

if run:
    if not transcript.strip():
        st.warning("Please provide a transcript first.")
        st.stop()

    with st.spinner("Evaluating..."):
        time.sleep(0.5)
        result = evaluate_transcript(
            transcript=transcript,
            filename=filename,
            reference_text=reference if reference.strip() else None,
            duration_seconds=duration,
            transcription_confidence=confidence
        )

    st.markdown("---")

    # ── Overall Score Banner ──────────────────────────────────────────────
    g = grade(result.overall_score)
    color = score_color(result.overall_score)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div style="text-align:center; padding: 20px; background: #161b22; border-radius: 12px; border: 2px solid {color};">
            <div style="font-size: 14px; color: #8b949e;">Overall Quality Score</div>
            <div style="font-size: 56px; font-weight: bold; color: {color};">{result.overall_score}</div>
            <div style="font-size: 20px; color: {color};">{g}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Dimension Scores ──────────────────────────────────────────────────
    st.subheader("Dimension Breakdown")
    c1, c2, c3, c4, c5 = st.columns(5)
    for col, label, score in zip(
        [c1, c2, c3, c4, c5],
        ["Transcription", "Clarity", "Hallucination", "Tone", "Fluency"],
        [result.transcription_score, result.clarity_score,
         result.hallucination_score, result.tone_score, result.fluency_score]
    ):
        col.metric(label, f"{score}/100", grade(score))

    st.markdown("---")

    # ── Charts ────────────────────────────────────────────────────────────
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Radar Chart")
        st.pyplot(render_radar(result))
    with col2:
        st.subheader("Score Breakdown")
        st.pyplot(render_bar_chart(result))

    st.markdown("---")

    # ── Details ───────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Voice Metrics")
        st.markdown(f"**Speech Rate:** {result.speech_rate_wpm:.0f} WPM {'✅' if 120 <= result.speech_rate_wpm <= 170 else '⚠️'}")
        st.markdown(f"**Word Count:** {result.word_count}")
        st.markdown(f"**Duration:** {result.duration_seconds:.1f}s")
        st.markdown(f"**Tone:** {result.tone_label}")
        st.markdown(f"**Hallucination:** {'⚠️ Detected' if result.has_hallucination else '✅ Clean'}")

        if result.filler_words_found:
            st.markdown(f"**Filler words:** {', '.join(result.filler_words_found[:5])}")
        if result.repeated_phrases:
            st.markdown(f"**Repeated phrases:** {', '.join(result.repeated_phrases[:3])}")

    with col2:
        if result.issues:
            st.subheader("Issues Found")
            for issue in result.issues:
                st.markdown(f'<div class="issue-box">⚠ {issue}</div>', unsafe_allow_html=True)

        if result.suggestions:
            st.subheader("Suggestions")
            for s in result.suggestions:
                st.markdown(f'<div class="suggest-box">→ {s}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Transcript Preview ────────────────────────────────────────────────
    with st.expander("View Full Transcript"):
        st.text(result.transcript)

    # ── Export ────────────────────────────────────────────────────────────
    st.subheader("Export Report")
    col1, col2 = st.columns(2)
    with col1:
        report_text = f"""AI VOICE QUALITY EVALUATION REPORT
File: {result.filename}
Overall: {result.overall_score}/100 ({grade(result.overall_score)})
Transcription: {result.transcription_score}/100
Clarity: {result.clarity_score}/100 ({result.speech_rate_wpm:.0f} WPM)
Hallucination: {result.hallucination_score}/100 ({'Detected' if result.has_hallucination else 'Clean'})
Tone: {result.tone_score}/100 ({result.tone_label})
Fluency: {result.fluency_score}/100
Issues: {chr(10).join(result.issues) if result.issues else 'None'}
Suggestions: {chr(10).join(result.suggestions) if result.suggestions else 'None'}
"""
        st.download_button("Download Report (.txt)", report_text,
                           file_name=f"voice_eval_{result.filename}.txt",
                           use_container_width=True)
    with col2:
        csv_text = "filename,overall,transcription,clarity,hallucination,tone,fluency,wpm,tone_label\n"
        csv_text += f"{result.filename},{result.overall_score},{result.transcription_score},"
        csv_text += f"{result.clarity_score},{result.hallucination_score},{result.tone_score},"
        csv_text += f"{result.fluency_score},{result.speech_rate_wpm:.0f},{result.tone_label}\n"
        st.download_button("Download CSV", csv_text,
                           file_name=f"voice_eval_{result.filename}.csv",
                           mime="text/csv",
                           use_container_width=True)
