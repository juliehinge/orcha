import re
import sys
from pathlib import Path


def extract_scores(path: Path) -> dict[str, float]:
    text = path.read_text()

    scores = {}
    for name, score in re.findall(r"[•*-]\s+([\w_]+):\s+([0-9.]+)", text):
        scores[name] = float(score)

    return scores


def compare_results(previous: Path, current: Path) -> str:
    previous_scores = extract_scores(previous)
    current_scores = extract_scores(current)

    lines = [
        "## Comparison with previous run",
        "",
        "| Evaluation | Previous | Current | Change |",
        "|---|---:|---:|---:|",
    ]

    for name, current_score in current_scores.items():
        previous_score = previous_scores.get(name)

        if previous_score is None:
            lines.append(
                f"| {name} | — | {current_score:.3f} | 🆕 New metric |"
            )
            continue

        change = current_score - previous_score

        if change > 0:
            indicator = "🟩"
        elif change < 0:
            indicator = "🟥"
        else:
            indicator = "⬜"

        lines.append(
            f"| {name} | {previous_score:.3f} | "
            f"{current_score:.3f} | {indicator} {change:+.3f} |"
        )

    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(
            "Usage: python compare_results.py PREVIOUS_REPORT CURRENT_REPORT"
        )

    previous = Path(sys.argv[1])
    current = Path(sys.argv[2])

    print(compare_results(previous, current))