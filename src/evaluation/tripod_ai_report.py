"""Extracted from notebook Sec. 10 ("TRIPOD+AI Reporting Checklist").
The checklist table is kept as a Python list of dicts and rendered to
Markdown here, rather than hand-copied, so reports/TRIPOD_AI_checklist.md
cannot silently drift from the code that generates it.
"""
from pathlib import Path
from typing import Dict, List

TRIPOD_AI_CHECKLIST: List[Dict[str, str]] = [
    {"item": "Source of data, eligibility criteria",
     "where_addressed": "Sec. 2-3 (PhysioNet Apnea-ECG, Kaggle mirror access)"},
    {"item": "Outcome definition",
     "where_addressed": "Per-minute apnea/normal from expert `.apn` annotations (Sec. 3)"},
    {"item": "Grouping to prevent leakage",
     "where_addressed": "Sec. 3, `GroupShuffleSplit` by `recording_id`, asserted disjoint"},
    {"item": "Feature engineering",
     "where_addressed": "Sec. 4 (time+frequency HRV, established QRS detector, not hand-rolled)"},
    {"item": "Model development (3 tiers)",
     "where_addressed": "Sec. 5-6"},
    {"item": "Discrimination + uncertainty",
     "where_addressed": "Sec. 7.2 recording-clustered bootstrap AUROC CIs"},
    {"item": "Calibration",
     "where_addressed": "Sec. 7.3, Sec. 8 reliability plot"},
    {"item": "Comparison between models",
     "where_addressed": "Sec. 8, DeLong pairwise tests"},
    {"item": "Physiological plausibility check",
     "where_addressed": "Sec. 5, Tier A coefficient inspection against Sec. 2's CVHR mechanism"},
    {"item": "Limitations",
     "where_addressed": "Sec. 11"},
]


def render_tripod_ai_markdown(checklist: List[Dict[str, str]] = TRIPOD_AI_CHECKLIST) -> str:
    lines = [
        "# TRIPOD+AI Reporting Checklist",
        "",
        "| TRIPOD+AI item | Where addressed |",
        "|---|---|",
    ]
    for row in checklist:
        lines.append(f"| {row['item']} | {row['where_addressed']} |")
    return "\n".join(lines) + "\n"


def write_tripod_ai_report(path: str = "reports/TRIPOD_AI_checklist.md") -> None:
    Path(path).write_text(render_tripod_ai_markdown(), encoding="utf-8")


if __name__ == "__main__":
    write_tripod_ai_report()
