# AI Voice Quality Evaluation System

A Python-based evaluation framework for AI-generated voice outputs.
Scores audio across 5 quality dimensions with hallucination detection.

**Built by [Kshitij Mishra](https://kshitij.info)**

---

## What It Does

Upload or paste an AI voice transcript and get instant quality scores across:

| Dimension | What it measures |
|-----------|-----------------|
| Transcription Accuracy | Whisper confidence score |
| Clarity | Speech rate (WPM), sentence structure |
| Hallucination Detection | Semantic comparison against reference text |
| Tone | Professional warmth vs negative/apologetic language |
| Fluency | Filler words, phrase repetition, sentence variety |

---

## Quick Start

```bash
pip install -r requirements.txt

# Run the web app
streamlit run app.py

# Or run the evaluator directly
python evaluator.py
```

---

## Project Structure

```
voice_eval/
  evaluator.py      # Core scoring engine (5 dimensions)
  transcriber.py    # Whisper audio transcription
  report.py         # Text + CSV report generator
  app.py            # Streamlit web app
  requirements.txt
```

---

## Tech Stack

- **Python** — core language
- **OpenAI Whisper** — local audio transcription (free, no API key)
- **Streamlit** — web app framework
- **Pandas / NumPy** — data processing
- **Matplotlib** — radar + bar charts

---

## Live Demo

[voice-eval-km.streamlit.app](https://voice-eval-km.streamlit.app)
