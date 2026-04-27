"""Sweep GBT blend weight in 5-fold LORO with R6 included.

Usage: python sweep_gbt_blend.py [val1,val2,...]
"""

import sys
import os
import numpy as np
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))

from sweep_l7_class import evaluate_loro, BASE_L7

def main():
    if len(sys.argv) > 1:
        values = [float(v) for v in sys.argv[1].split(",")]
    else:
        values = [0.20, 0.25, 0.30, 0.35, 0.40]

    print(f"Sweeping GBT blend weight over {values}")
    print(f"L7 strengths: {BASE_L7.tolist()}")
    print()

    best_val = None
    best_avg = -1

    for blend in values:
        avg, per_round = evaluate_loro(BASE_L7, blend)
        rounds_str = " ".join(f"R{r}={s:.2f}" for r, s in sorted(per_round.items()))
        print(f"  blend={blend:.2f}: avg={avg:.2f}  {rounds_str}")
        if avg > best_avg:
            best_avg = avg
            best_val = blend

    print(f"\nBEST: blend={best_val:.2f} -> avg={best_avg:.2f}")


if __name__ == "__main__":
    main()
