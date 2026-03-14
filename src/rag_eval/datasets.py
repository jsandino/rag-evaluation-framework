from typing import List, Dict
import json
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def load_documents() -> List[Dict]:
    """Load evaluation documents."""
    path = DATA_DIR / "documents.json"

    with open(path, "r") as f:
        return json.load(f)


def load_queries() -> List[Dict]:
    """Load evaluation queries."""
    path = DATA_DIR / "queries.json"

    with open(path, "r") as f:
        return json.load(f)
