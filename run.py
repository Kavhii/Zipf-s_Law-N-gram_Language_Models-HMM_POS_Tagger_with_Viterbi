"""Run the full NLU assignment pipeline from a single entry point."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPTS = [
    ROOT / "nlu_assignment1" / "part1_zipf.py",
    ROOT / "nlu_assignment1" / "part2_lm.py",
    ROOT / "sentences" / "part3_hmm.py",
]


def main() -> None:
    print("Running the full NLU assignment pipeline...\n")

    for script in SCRIPTS:
        print(f"==> Running {script.relative_to(ROOT)}")
        subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)
        print()

    print("All assignment steps completed successfully.")


if __name__ == "__main__":
    main()
