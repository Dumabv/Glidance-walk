"""GitHub Actions / local CLI quality gate for Glidance classroom evaluation traces.

Runs TWO GPU-free, deterministic evaluation layers in < 0.3 seconds:
  1. 20 real labeled classroom/hallway photos (`real_photos_trace.npz`): Naive vs Glidance F1,
     Directional Accuracy, Clear-Floor False Alarms, and exact list of any missed scenes.
  2. 71-second continuous classroom walk (`classroom_walk_trace.npz`): Lean 4 runtime safety
     invariants on all 357 frames, 7 Golden Timestamp Scenarios, and Self-Diagnostics Advisor.

Outputs a rich Markdown summary to stdout and `$GITHUB_STEP_SUMMARY` (GitHub PR Checks UI).

Usage:
    python scripts/ci_video_replay.py
    python scripts/ci_video_replay.py --github-summary --min-score 60 --min-photo-f1 0.75
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.evaluation.trace_replay import (
    format_markdown_summary,
    replay_real_photos_trace,
    replay_trace,
)

FIXTURES_DIR = REPO_ROOT / "backend" / "tests" / "fixtures"
DEFAULT_WALK_NPZ = FIXTURES_DIR / "classroom_walk_trace.npz"
DEFAULT_WALK_META = FIXTURES_DIR / "classroom_walk_meta.json"
DEFAULT_PHOTOS_NPZ = FIXTURES_DIR / "real_photos_trace.npz"
DEFAULT_PHOTOS_META = FIXTURES_DIR / "real_photos_meta.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay pre-baked classroom photos + 71s walk trace as a fast CI quality gate."
    )
    parser.add_argument("--npz", type=Path, default=DEFAULT_WALK_NPZ, help="Path to baked walk .npz trace")
    parser.add_argument("--meta", type=Path, default=DEFAULT_WALK_META, help="Path to baked walk _meta.json")
    parser.add_argument("--photos-npz", type=Path, default=DEFAULT_PHOTOS_NPZ, help="Path to baked real photos .npz")
    parser.add_argument("--photos-meta", type=Path, default=DEFAULT_PHOTOS_META, help="Path to baked real photos _meta.json")
    parser.add_argument("--min-score", type=int, default=60, help="Minimum self-diagnostic health score (default: 60)")
    parser.add_argument("--min-photo-f1", type=float, default=0.75, help="Minimum Glidance F1 on real photos (default: 0.75)")
    parser.add_argument("--github-summary", action="store_true", help="Append Markdown report to $GITHUB_STEP_SUMMARY")
    args = parser.parse_args()

    photo_report = None
    if args.photos_npz.exists() and args.photos_meta.exists():
        photo_report = replay_real_photos_trace(args.photos_npz, args.photos_meta)

    report = replay_trace(args.npz, args.meta)
    score = int(report["diagnostics"]["health_score"])
    if score < args.min_score:
        report["passed"] = False
    if photo_report is not None and float(photo_report["smart"]["f1"]) < args.min_photo_f1:
        report["passed"] = False

    md = format_markdown_summary(report, photo_report=photo_report)
    print(md)

    summary_file = os.getenv("GITHUB_STEP_SUMMARY")
    if (args.github_summary or summary_file) and summary_file:
        with open(summary_file, "a", encoding="utf-8") as fh:
            fh.write(md)

    if not report["passed"]:
        photo_f1 = photo_report["smart"]["f1"] if photo_report else None
        print(
            f"[CI Replay FAILED] score={score} (min={args.min_score}), "
            f"photo_f1={photo_f1} (min={args.min_photo_f1}), "
            f"failed_scenarios={sum(1 for s in report['scenarios'] if not s['passed'])}, "
            f"invariant_violations={len(report['invariant_violations'])}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
