"""pytest: the published nbe.json satisfies the player schema; the validator catches breakage."""
import json
import os
from pathlib import Path

from validate_nbe import validate

ROOT = Path(__file__).resolve().parents[1]
NBE = Path(os.environ.get("NBE_JSON", ROOT / "output" / "nbe.json"))


def test_published_file_is_valid():
    doc = json.loads(NBE.read_text(encoding="utf-8"))
    errors, _ = validate(doc)
    assert errors == []
    assert 2 <= len(doc["sessions"]) <= 4
    for s in doc["sessions"]:
        assert 10 <= len(s["lines"]) <= 40


def test_validator_rejects_bad_events_and_long_lines():
    doc = {"generated": "x", "date": "2026-10-12",
           "graph": {"nodes": [{"id": "A", "label": "A", "views": 1}], "links": [], "clusters": []},
           "sessions": [{"title": "t", "lines": [["out", "x" * 60], ["hi", "y", {"type": "pulse", "id": "B"}],
                                                 ["nope", "z"]]}]}
    errors, _ = validate(doc)
    assert any("chars" in e for e in errors)
    assert any("unknown node" in e for e in errors)
    assert any("unknown kind" in e for e in errors)
