"""Local prediction with the trained pipeline; informational, not authorship proof.

Example: python -m src.predict --text "Some example text"
"""
import argparse
from pathlib import Path
import joblib
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="outputs/latest/selected_model.joblib")
    p.add_argument("--text", required=True)
    args = p.parse_args()
    if not Path(args.model).exists():
        raise FileNotFoundError("Run python -m src.train first; no trained model found.")
    model = joblib.load(args.model)
    cls = int(model.predict(pd.DataFrame({"Text": [args.text]}))[0])
    print(f"Predicted label: {cls} ({'Human' if cls == 1 else 'AI'})")
    print("Note: prediction is not reliable proof of authorship.")


if __name__ == "__main__":
    main()
