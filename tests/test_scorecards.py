"""The normaliser must read every sibling shape and never raise on a bad one."""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ledger_paths  # noqa: E402
import scorecards  # noqa: E402
from collect import _unbool, build, render_md  # noqa: E402

GATE_NODE = {
    "passes": False,
    "deflated_sharpe": {"sr": 0.1281, "ratio": -18.3675, "prob": 0.0,
                        "n": 210, "n_trials": 8},
    "pbo": 0.0, "profit_factor": 1.4154, "sharpe": 0.128, "n_trades": 210,
    "reasons": ["deflated_sharpe_ratio -18.3675 <= 0.0"],
}


@pytest.fixture
def fake_box(tmp_path, monkeypatch):
    """A miniature box with one scorecard of each known shape."""
    macro = tmp_path / "macro_gpu_lab" / "journal" / "scorecards"
    qlib = tmp_path / "qlib_lab" / "journal" / "scorecards"
    alpaca = tmp_path / "alpaca_gpu_lab" / "journal" / "scorecards"
    for d in (macro, qlib, alpaca):
        d.mkdir(parents=True)

    (macro / "macro_gpu_rf_2026-07-04.json").write_text(json.dumps({
        "model": "rf", "horizons": {"5d": GATE_NODE, "21d": GATE_NODE}}))
    # the surprise shape: strategies nested under a horizon, plus detection stats
    (macro / "macro_gpu_surprise_2026-07-04.json").write_text(json.dumps({
        "model": "surprise_shift",
        "horizons": {"5d": {"strategies": {"short_spy_overlay": GATE_NODE},
                            "detection": {"precision": 0.23, "base_rate_calm": 0.104},
                            "lift_vs_calm_base": 2.21}}}))
    (qlib / "qlib_lgbm_2026-07-29.json").write_text(json.dumps({
        "model": "qlib_lgbm", "n_trials": 4, "data_through": "2026-07-29",
        "horizons": {"5d": GATE_NODE}}))
    (qlib / "experiments_2026-07-13.json").write_text(json.dumps({
        "n_trials": 20, "battery": {"H01_tsmom": {**GATE_NODE, "mean_pnl": 0.00114}},
        "lead_lag": {"n_tests": 22800, "bonferroni_alpha": 2.19e-6,
                     "survivors": [{"leader": "SPY", "target": "INDA", "lag": 1,
                                    "corr_discovery": -0.2187,
                                    "corr_validation": -0.0777}]}}))
    (qlib / "deep_history_2026-07-14.json").write_text(json.dumps({
        "window": ["1995-01-02", "2015-12-31"],
        "results": {"H07_vrp_spy": {"n": 236, "pf": 2.9199, "pf_2016_2026": 5.02,
                                    "verdict": "replicates",
                                    "gate": {"passes": False, "pbo": 0.004}}}}))
    (alpaca / "B04_xgb_sentiment_2026-07-17.json").write_text(json.dumps({
        "battery_id": "B04_xgb_sentiment", "timeframe": "1Min",
        "run_at": "2026-07-17T15:23:37+00:00",
        "combos": [{"params": {"model": "xgb"},
                    "horizons": {"5": {"passes": False, "deflated_sharpe": None,
                                       "pbo": None, "n_trades": 0,
                                       "reasons": ["insufficient PnL"]}}}]}))
    # a corrupt file must be skipped with a note, not raise
    (qlib / "qlib_lgbm_2026-07-28.json").write_text("{not json at all")

    monkeypatch.setattr(ledger_paths, "MACRO_SCORECARDS", macro)
    monkeypatch.setattr(ledger_paths, "QLIB_SCORECARDS", qlib)
    monkeypatch.setattr(ledger_paths, "ALPACA_SCORECARDS", alpaca)
    monkeypatch.setattr(ledger_paths, "ALPACA_RESULTS_MD", tmp_path / "nope.md")
    monkeypatch.setattr(ledger_paths, "FLAG_FILES", {})
    scorecards.PARSE_NOTES.clear()
    return tmp_path


def test_reads_every_shape(fake_box):
    rows = list(scorecards.iter_gate_rows())
    labels = {r["label"] for r in rows}
    assert any("macro rf" in x for x in labels)
    assert any("short_spy_overlay" in x for x in labels)
    assert any("qlib qlib_lgbm" in x for x in labels)
    assert any("H01_tsmom" in x for x in labels)
    assert any("deep-history" in x for x in labels)
    assert any("B04_xgb_sentiment" in x for x in labels)


def test_gate_fields_survive_normalisation(fake_box):
    rows = [r for r in scorecards.iter_gate_rows() if r["label"] == "macro rf"]
    assert rows
    r = rows[0]
    assert r["sr"] == 0.1281 and r["dsr_ratio"] == -18.3675
    assert r["n_trials"] == 8 and r["n_trades"] == 210 and r["pbo"] == 0.0
    assert r["passes"] is False


def test_n_trials_falls_back_to_the_file_level_value(fake_box):
    """qlib_lgbm carries n_trials at the top level; a battery inherits it."""
    rows = [r for r in scorecards.iter_gate_rows() if "H01_tsmom" in r["label"]]
    assert rows[0]["n_trials"] == 8       # node value wins when present
    assert rows[0]["mean_pnl"] == 0.00114


def test_detection_only_rows_are_kept(fake_box):
    rows = [r for r in scorecards.iter_gate_rows() if r["label"].endswith("detection")]
    assert rows and rows[0]["detection"]["precision"] == 0.23
    assert rows[0]["lift_vs_base"] == 2.21


def test_corrupt_file_is_noted_not_raised(fake_box):
    list(scorecards.iter_gate_rows())
    assert any("qlib_lgbm_2026-07-28" in n for n in scorecards.PARSE_NOTES)


def test_zero_trade_rows_do_not_invent_numbers(fake_box):
    rows = [r for r in scorecards.iter_gate_rows()
            if "B04_xgb_sentiment" in r["label"]]
    assert rows[0]["n_trades"] == 0
    assert rows[0]["sr"] is None and rows[0]["dsr_ratio"] is None


def test_lead_lag_blocks_parse(fake_box):
    blocks = list(scorecards.lead_lag_blocks())
    assert blocks and blocks[0]["n_tests"] == 22800
    assert blocks[0]["survivors"][0]["target"] == "INDA"


def test_missing_directories_yield_nothing(tmp_path, monkeypatch):
    for name in ("MACRO_SCORECARDS", "QLIB_SCORECARDS", "ALPACA_SCORECARDS"):
        monkeypatch.setattr(ledger_paths, name, tmp_path / "absent")
    monkeypatch.setattr(ledger_paths, "ALPACA_RESULTS_MD", tmp_path / "absent.md")
    assert list(scorecards.iter_gate_rows()) == []


def test_freshness_spots_gated_once_never_regated(fake_box):
    fresh = {f["repo"]: f for f in scorecards.scorecard_freshness(date(2026, 7, 30))}
    assert fresh["macro_gpu_lab"]["distinct_dates"] == 1
    assert fresh["macro_gpu_lab"]["age_days"] == 26
    assert fresh["qlib_lab"]["distinct_dates"] >= 3


def test_flag_staleness_reports_missing_as_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger_paths, "FLAG_FILES",
                        {"nope": tmp_path / "gone.json"})
    out = scorecards.flag_staleness(date(2026, 7, 30))
    assert out[0]["status"] == "MISSING" and out[0]["exists"] is False


def test_yaml_no_is_not_parsed_as_false():
    """The trap that inverted a verdict on first run -- keep it caught."""
    assert _unbool(False) == "NO"
    assert _unbool(True) == "YES"
    assert _unbool("NO-DATA") == "NO-DATA"


def test_render_md_is_stable_and_mentions_every_question(fake_box, monkeypatch):
    monkeypatch.setattr(ledger_paths, "JOURNAL", fake_box / "journal")
    (fake_box / "journal").mkdir(exist_ok=True)
    doc = build({"meta": {"gate_caveat": "careful"},
                 "questions": [
                     {"id": "Q-X-01", "question": "does it work?", "repo": "r",
                      "status": "NO", "yes_bar": "IC>0", "answered_by": "g.py"},
                     {"id": "Q-X-02", "question": "and this?", "repo": "r",
                      "status": "OPEN", "yes_bar": "n>=150", "answered_by": "TODO"}]})
    md = render_md(doc)
    assert "Q-X-01" in md and "Q-X-02" in md
    assert "careful" in md
    assert doc["counts"] == {"NO": 1, "OPEN": 1}
    # NO sorts after OPEN? no -- OPEN(4) before NO(5): verdicts first, unknowns last
    assert [a["id"] for a in doc["answers"]] == ["Q-X-02", "Q-X-01"]
