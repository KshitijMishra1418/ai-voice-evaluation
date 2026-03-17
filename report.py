"""
report.py
----------
Generates structured evaluation reports in text and CSV formats.
"""

import csv
import os
from datetime import datetime
from evaluator import VoiceEvalResult, grade


def generate_report(results: list, output_dir: str = "reports") -> str:
    """Generate a full text report from a list of VoiceEvalResults."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    lines = []

    def line(t=""): lines.append(t)

    line("=" * 65)
    line("       AI VOICE QUALITY EVALUATION REPORT")
    line(f"       Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    line(f"       Samples evaluated: {len(results)}")
    line("=" * 65)

    # Summary stats
    if results:
        avg_overall = sum(r.overall_score for r in results) / len(results)
        flagged     = sum(1 for r in results if r.has_hallucination)
        avg_clarity = sum(r.clarity_score for r in results) / len(results)

        line()
        line("  BATCH SUMMARY")
        line(f"  Avg Overall Score:  {avg_overall:.1f}/100  ({grade(avg_overall)})")
        line(f"  Avg Clarity Score:  {avg_clarity:.1f}/100")
        line(f"  Hallucinations:     {flagged}/{len(results)} samples flagged")
        line()

    for i, r in enumerate(results, 1):
        line("─" * 65)
        line(f"  [{i}] {r.filename}")
        line()
        line(f"  Overall Score:     {r.overall_score}/100  ({grade(r.overall_score)})")
        line(f"  Transcription:     {r.transcription_score}/100")
        line(f"  Clarity:           {r.clarity_score}/100  ({r.speech_rate_wpm:.0f} WPM)")
        line(f"  Hallucination:     {r.hallucination_score}/100  ({'⚠ FLAGGED' if r.has_hallucination else '✓ Clean'})")
        line(f"  Tone:              {r.tone_score}/100  ({r.tone_label})")
        line(f"  Fluency:           {r.fluency_score}/100")
        line()
        line(f"  Word Count:        {r.word_count}")
        line(f"  Duration:          {r.duration_seconds:.1f}s")
        line()

        if r.filler_words_found:
            line(f"  Filler words:      {', '.join(r.filler_words_found[:5])}")
        if r.repeated_phrases:
            line(f"  Repeated phrases:  {', '.join(r.repeated_phrases[:3])}")
        if r.issues:
            line()
            line("  Issues:")
            for issue in r.issues:
                line(f"    • {issue}")
        if r.suggestions:
            line()
            line("  Suggestions:")
            for s in r.suggestions:
                line(f"    → {s}")
        line()

    line("=" * 65)
    line("  End of Report — AI Voice Quality Evaluation System")
    line("  Built by Kshitij Mishra — kshitij.info")
    line("=" * 65)

    report_path = os.path.join(output_dir, f"voice_eval_report_{timestamp}.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print(f"\nReport saved: {report_path}")
    return report_path


def export_csv(results: list, output_dir: str = "reports") -> str:
    """Export results to CSV for further analysis."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    csv_path = os.path.join(output_dir, f"voice_eval_{timestamp}.csv")

    fields = [
        "filename", "overall_score", "grade",
        "transcription_score", "clarity_score",
        "hallucination_score", "tone_score", "fluency_score",
        "has_hallucination", "tone_label",
        "speech_rate_wpm", "word_count", "duration_seconds",
        "filler_count", "issues_count"
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "filename":            r.filename,
                "overall_score":       r.overall_score,
                "grade":               grade(r.overall_score),
                "transcription_score": r.transcription_score,
                "clarity_score":       r.clarity_score,
                "hallucination_score": r.hallucination_score,
                "tone_score":          r.tone_score,
                "fluency_score":       r.fluency_score,
                "has_hallucination":   r.has_hallucination,
                "tone_label":          r.tone_label,
                "speech_rate_wpm":     r.speech_rate_wpm,
                "word_count":          r.word_count,
                "duration_seconds":    r.duration_seconds,
                "filler_count":        len(r.filler_words_found),
                "issues_count":        len(r.issues),
            })

    print(f"CSV saved: {csv_path}")
    return csv_path
