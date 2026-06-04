"""Run the whole thing in one go.

By default it skips the data download (the snapshot is already in data/) and
just runs the analysis, the energy yield, the wind shear and the forecasting
model. pass --fetch if you want to pull a fresh copy from Open Meteo first.

Run:  python run_all.py
      python run_all.py --fetch
"""

import argparse
import os
import sys

SRC = os.path.join(os.path.dirname(__file__), "src")
sys.path.insert(0, os.path.abspath(SRC))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true",
                    help="download fresh data before running")
    args = ap.parse_args()

    if args.fetch:
        import fetch_data
        print("\n### fetch_data ###")
        fetch_data.main()

    import analysis
    import energy_yield
    import wind_shear
    import model

    for name, mod in [("analysis", analysis), ("energy_yield", energy_yield),
                      ("wind_shear", wind_shear), ("model", model)]:
        print(f"\n### {name} ###")
        mod.main()

    print("\nall done. figures are in results/")


if __name__ == "__main__":
    main()
