from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from ibm_client import IBMScoringClient


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = ROOT / "data" / "sample_features.csv"

api_key = os.environ["IBM_API_KEY"]
scoring_url = os.environ["IBM_SCORING_URL"]

sample = pd.read_csv(SAMPLE_PATH)
client = IBMScoringClient(api_key=api_key, scoring_url=scoring_url)

predictions = client.score(
    fields=sample.columns.tolist(),
    values=sample.astype(float).values.tolist(),
)

result = pd.DataFrame({"prediction": predictions})
print(result.to_string(index=False))
