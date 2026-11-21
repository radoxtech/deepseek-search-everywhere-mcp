"""--skip-labelled keeps packs to pages that have no verdict yet."""

from __future__ import annotations

import json

import pytest
import query_lab


@pytest.fixture
def lab(tmp_path, monkeypatch):
    monkeypatch.setattr(query_lab, "HERE", tmp_path)
    monkeypatch.setattr(query_lab, "judge", type("J", (), {"_rubric_text": staticmethod(lambda _t: "kp")}))
    monkeypatch.setattr(query_lab, "_rubric", lambda _t: {"topic": "T", "key_points": []})

    def topic(name: str, urls: list[str]) -> None:
        d = tmp_path / "results" / name / "t1"
        d.mkdir(parents=True)
        (d / "aspects.json").write_text(
            json.dumps({"results": [{"query": "q", "url": u, "title": "t", "content": "c"} for u in urls]})
        )

    topic("old", ["https://a.test/1", "https://b.test/2"])
    topic("new", ["https://a.test/1", "https://c.test/3"])

    v = tmp_path / "results" / "old" / "t1" / "verdicts"
    v.mkdir(parents=True)
    (v / "p00.json").write_text(json.dumps({"pages": {query_lab._page_id("https://a.test/1"): ["k1"]}}))
    return tmp_path


def _pack_body(tmp_path, name: str) -> str:
    return "\n".join(p.read_text() for p in (tmp_path / "results" / name / "t1" / "packs").glob("*.md"))


def test_skips_pages_already_labelled_elsewhere(lab) -> None:
    query_lab.packs(type("A", (), {"name": "new", "skip_labelled": "old"})())
    body = _pack_body(lab, "new")
    assert "https://c.test/3" in body
    assert "https://a.test/1" not in body


def test_without_the_flag_every_page_is_packed(lab) -> None:
    query_lab.packs(type("A", (), {"name": "new", "skip_labelled": None})())
    body = _pack_body(lab, "new")
    assert "https://a.test/1" in body and "https://c.test/3" in body
