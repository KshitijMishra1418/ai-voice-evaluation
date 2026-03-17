"""
evaluator.py
-------------
Core AI Voice Quality Evaluation Engine.

Scores audio on 5 dimensions:
  1. Transcription Accuracy   — via Whisper
  2. Clarity Score            — word confidence + speech rate
  3. Hallucination Detection  — checks if output matches reference text
  4. Emotional Tone           — keyword-based sentiment analysis
  5. Fluency Score            — pause detection, filler words, repetition
"""

import re
import math
from dataclasses import dataclass, field
from typing import Optional


# ── Data Models ───────────────────────────────────────────────────────────────

@dataclass
class VoiceEvalResult:
    filename: str
    transcript: str
    reference_text: Optional[str]

    # Scores (0-100)
    transcription_score: float = 0.0
    clarity_score:       float = 0.0
    hallucination_score: float = 0.0   # 100 = no hallucination
    tone_score:          float = 0.0
    fluency_score:       float = 0.0
    overall_score:       float = 0.0

    # Flags
    has_hallucination:   bool  = False
    filler_words_found:  list  = field(default_factory=list)
    repeated_phrases:    list  = field(default_factory=list)
    tone_label:          str   = "Neutral"
    word_count:          int   = 0
    duration_seconds:    float = 0.0
    speech_rate_wpm:     float = 0.0

    # Raw detail
    issues:              list  = field(default_factory=list)
    suggestions:         list  = field(default_factory=list)


# ── Scoring Functions ─────────────────────────────────────────────────────────

FILLER_WORDS = {
    "um", "uh", "er", "ah", "like", "you know", "basically",
    "literally", "actually", "so", "right", "okay so", "i mean"
}

POSITIVE_WORDS = {
    "good", "great", "excellent", "happy", "wonderful", "thank",
    "pleasure", "glad", "appreciate", "fantastic", "perfect",
    "helpful", "assist", "support", "welcome", "sure"
}

NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "sorry", "wrong", "error", "fail",
    "issue", "problem", "unfortunately", "cannot", "unable", "broken"
}

FORMAL_WORDS = {
    "therefore", "however", "furthermore", "additionally", "consequently",
    "regarding", "pursuant", "accordingly", "hereby", "therein"
}


def score_clarity(transcript: str, duration: float) -> tuple[float, float, str]:
    """
    Clarity score based on:
    - Speech rate (ideal: 130-160 WPM)
    - Sentence length variance
    - Punctuation density
    Returns (score, wpm, feedback)
    """
    words = transcript.split()
    word_count = len(words)

    if duration <= 0:
        return 70.0, 0.0, "Duration unknown — clarity estimated."

    wpm = (word_count / duration) * 60

    # Score based on WPM
    if 120 <= wpm <= 170:
        rate_score = 100
        rate_feedback = f"Ideal speech rate ({wpm:.0f} WPM)"
    elif 90 <= wpm < 120 or 170 < wpm <= 200:
        rate_score = 80
        rate_feedback = f"Slightly {'slow' if wpm < 120 else 'fast'} ({wpm:.0f} WPM)"
    elif wpm < 90:
        rate_score = 60
        rate_feedback = f"Too slow ({wpm:.0f} WPM) — may sound unnatural"
    else:
        rate_score = 55
        rate_feedback = f"Too fast ({wpm:.0f} WPM) — hard to follow"

    # Sentence length score
    sentences = re.split(r'[.!?]+', transcript)
    sentences = [s.strip() for s in sentences if s.strip()]
    if sentences:
        avg_len = sum(len(s.split()) for s in sentences) / len(sentences)
        len_score = 100 if 8 <= avg_len <= 20 else 75
    else:
        len_score = 70

    clarity = (rate_score * 0.6 + len_score * 0.4)
    return round(clarity, 1), round(wpm, 1), rate_feedback


def score_fluency(transcript: str) -> tuple[float, list, list, str]:
    """
    Fluency score based on:
    - Filler word frequency
    - Phrase repetition
    - Sentence variety
    Returns (score, fillers_found, repeated_phrases, feedback)
    """
    text_lower = transcript.lower()
    words = text_lower.split()
    total_words = len(words) or 1

    # Filler word detection
    found_fillers = []
    for filler in FILLER_WORDS:
        count = text_lower.count(filler)
        if count > 0:
            found_fillers.append(f'"{filler}" ×{count}')

    filler_ratio = len(found_fillers) / total_words
    filler_score = max(0, 100 - filler_ratio * 500)

    # Repetition detection (3+ word phrases)
    repeated = []
    trigrams = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
    seen = {}
    for tg in trigrams:
        seen[tg] = seen.get(tg, 0) + 1
    repeated = [f'"{k}" ×{v}' for k, v in seen.items() if v > 1]

    rep_score = max(0, 100 - len(repeated) * 15)

    fluency = (filler_score * 0.6 + rep_score * 0.4)
    feedback = "Natural and fluent" if fluency > 80 else \
               "Some disfluencies detected" if fluency > 60 else \
               "Multiple fluency issues found"

    return round(fluency, 1), found_fillers, repeated[:5], feedback


def score_tone(transcript: str) -> tuple[float, str]:
    """
    Tone score: how professional and warm the voice output sounds.
    Returns (score, label)
    """
    text_lower = transcript.lower()
    words = set(text_lower.split())

    pos = len(words & POSITIVE_WORDS)
    neg = len(words & NEGATIVE_WORDS)
    formal = len(words & FORMAL_WORDS)

    total = pos + neg + formal or 1
    pos_ratio    = pos / total
    formal_ratio = formal / len(words) if words else 0

    if pos_ratio > 0.4:
        label = "Warm & Positive"
        score = 90 + min(10, pos * 2)
    elif formal_ratio > 0.05:
        label = "Formal & Professional"
        score = 85
    elif neg > pos:
        label = "Negative / Apologetic"
        score = 55
    else:
        label = "Neutral"
        score = 75

    return min(100, round(score, 1)), label


def score_hallucination(transcript: str, reference: Optional[str]) -> tuple[float, bool, list]:
    """
    Hallucination score:
    - If no reference: checks for factual red flags (made-up names, dates, numbers)
    - If reference given: compares semantic overlap
    Returns (score, has_hallucination, issues)
    """
    issues = []

    if not reference:
        # Heuristic checks
        suspicious = []

        # Check for very specific numbers that may be fabricated
        numbers = re.findall(r'\b\d{4,}\b', transcript)
        if len(numbers) > 3:
            suspicious.append(f"Many specific numbers: {numbers[:5]}")

        # Check for all-caps words (may be acronyms or emphasis)
        caps_words = re.findall(r'\b[A-Z]{3,}\b', transcript)
        if len(caps_words) > 5:
            suspicious.append(f"Many capitalised terms: {caps_words[:5]}")

        if suspicious:
            issues = suspicious
            return 70.0, True, issues

        return 95.0, False, []

    # Reference-based comparison
    ref_words   = set(reference.lower().split())
    trans_words = set(transcript.lower().split())

    # Remove stopwords
    stopwords = {"the","a","an","is","are","was","were","i","you","we","it","in","on","at","to","for","of","and","or"}
    ref_words   -= stopwords
    trans_words -= stopwords

    if not ref_words:
        return 85.0, False, []

    overlap = len(ref_words & trans_words) / len(ref_words)
    extra   = trans_words - ref_words
    extra   = [w for w in extra if len(w) > 4]

    if overlap < 0.3:
        issues.append(f"Low overlap with reference ({overlap:.0%}) — possible hallucination")
        score = 40.0
        has_h = True
    elif overlap < 0.6:
        issues.append(f"Moderate overlap with reference ({overlap:.0%})")
        score = 70.0
        has_h = len(extra) > 10
    else:
        score = 90 + overlap * 10
        has_h = False

    if len(extra) > 15:
        issues.append(f"{len(extra)} words not in reference text")

    return min(100, round(score, 1)), has_h, issues


def compute_overall(scores: dict, weights: Optional[dict] = None) -> float:
    """Weighted average of all dimension scores."""
    if weights is None:
        weights = {
            "transcription": 0.25,
            "clarity":       0.20,
            "hallucination": 0.25,
            "fluency":       0.15,
            "tone":          0.15,
        }
    total = sum(scores[k] * weights[k] for k in weights if k in scores)
    return round(total, 1)


# ── Main Evaluator ────────────────────────────────────────────────────────────

def evaluate_transcript(
    transcript: str,
    filename: str = "audio",
    reference_text: Optional[str] = None,
    duration_seconds: float = 0.0,
    transcription_confidence: float = 0.85,
) -> VoiceEvalResult:
    """
    Full evaluation pipeline for a transcript.
    
    Args:
        transcript:               The text output from the voice AI
        filename:                 Name of the source audio file
        reference_text:           Expected/correct text (optional)
        duration_seconds:         Length of audio clip in seconds
        transcription_confidence: Confidence from Whisper (0-1)
    
    Returns:
        VoiceEvalResult with all scores and flags
    """
    result = VoiceEvalResult(
        filename=filename,
        transcript=transcript,
        reference_text=reference_text,
        duration_seconds=duration_seconds,
        word_count=len(transcript.split()),
    )

    # 1. Transcription score
    result.transcription_score = round(transcription_confidence * 100, 1)

    # 2. Clarity
    result.clarity_score, result.speech_rate_wpm, clarity_note = score_clarity(transcript, duration_seconds)

    # 3. Hallucination
    result.hallucination_score, result.has_hallucination, hall_issues = score_hallucination(transcript, reference_text)

    # 4. Tone
    result.tone_score, result.tone_label = score_tone(transcript)

    # 5. Fluency
    result.fluency_score, result.filler_words_found, result.repeated_phrases, fluency_note = score_fluency(transcript)

    # Overall
    result.overall_score = compute_overall({
        "transcription": result.transcription_score,
        "clarity":       result.clarity_score,
        "hallucination": result.hallucination_score,
        "fluency":       result.fluency_score,
        "tone":          result.tone_score,
    })

    # Issues & suggestions
    if result.has_hallucination:
        result.issues.extend(hall_issues)
        result.suggestions.append("Review transcript against source script for fabricated content.")
    if result.filler_words_found:
        result.issues.append(f"Filler words detected: {', '.join(result.filler_words_found[:5])}")
        result.suggestions.append("Retrain or fine-tune voice model to reduce filler words.")
    if result.repeated_phrases:
        result.issues.append(f"Repeated phrases: {', '.join(result.repeated_phrases[:3])}")
        result.suggestions.append("Check for repetition in the TTS prompt or model output.")
    if result.speech_rate_wpm > 0 and not (120 <= result.speech_rate_wpm <= 170):
        result.issues.append(clarity_note)
        result.suggestions.append(f"Adjust speech rate. Current: {result.speech_rate_wpm:.0f} WPM. Target: 130-160 WPM.")
    if result.tone_label in ["Negative / Apologetic"]:
        result.issues.append("Negative tone detected in output.")
        result.suggestions.append("Review prompt or script for overly apologetic language.")

    return result


def grade(score: float) -> str:
    """Convert numeric score to grade label."""
    if score >= 90: return "Excellent"
    if score >= 75: return "Good"
    if score >= 60: return "Fair"
    if score >= 45: return "Poor"
    return "Critical"


if __name__ == "__main__":
    # Quick test
    sample = """
    Hello and welcome to Tavus AI. I'm your AI assistant and I'm here to help you today.
    Um, so basically what we can do is, uh, help you create personalized video messages.
    You know, our platform is really quite good at generating realistic voice and video outputs.
    I mean, it's, like, one of the best platforms available right now.
    """
    result = evaluate_transcript(
        transcript=sample,
        filename="test_sample.wav",
        duration_seconds=18.0,
        transcription_confidence=0.91
    )
    print(f"\n=== Evaluation Results ===")
    print(f"Overall Score:    {result.overall_score}/100 ({grade(result.overall_score)})")
    print(f"Transcription:    {result.transcription_score}/100")
    print(f"Clarity:          {result.clarity_score}/100  ({result.speech_rate_wpm:.0f} WPM)")
    print(f"Hallucination:    {result.hallucination_score}/100  ({'⚠ Detected' if result.has_hallucination else '✓ Clean'})")
    print(f"Tone:             {result.tone_score}/100  ({result.tone_label})")
    print(f"Fluency:          {result.fluency_score}/100")
    if result.filler_words_found:
        print(f"Filler words:     {', '.join(result.filler_words_found)}")
    if result.issues:
        print(f"\nIssues:")
        for i in result.issues: print(f"  • {i}")
    if result.suggestions:
        print(f"\nSuggestions:")
        for s in result.suggestions: print(f"  → {s}")
