# -*- coding: utf-8 -*-
"""Optional compatibility wrapper.

Use only if you already have a harmonized CSV and want to rebuild enriched features.
The main pipeline now does this automatically.
"""

import argparse
from pathlib import Path
import pandas as pd

if __package__ is None or __package__ == "":
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from meta_analysis.features import prepare_dataframe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("harmonized_csv")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()
    input_csv = Path(args.harmonized_csv)
    output_dir = Path(args.output_dir) if args.output_dir else input_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(input_csv).to_dict(orient="records")
    df = prepare_dataframe(rows)
    df.to_csv(output_dir / "ca1_metadata_enriched_features.csv", index=False)
    print("Done:", output_dir / "ca1_metadata_enriched_features.csv")


if __name__ == "__main__":
    main()
