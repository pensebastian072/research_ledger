"""Every sibling path in one place, overridable by env.

Nothing else in this repo hardcodes a sibling location. Each accessor returns a
Path that may not exist -- callers treat a missing path as NO-DATA, never as an
error (hard rule 3).
"""
from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
BOX = Path(os.environ.get("QUANT_BOX_DIR", str(HERE.parent)))

MACRO_GPU_LAB = Path(os.environ.get("MACRO_GPU_LAB_DIR", str(BOX / "macro_gpu_lab")))
QLIB_LAB = Path(os.environ.get("QLIB_LAB_DIR", str(BOX / "qlib_lab")))
ALPACA_LAB = Path(os.environ.get("ALPACA_GPU_LAB_DIR", str(BOX / "alpaca_gpu_lab")))
HQ = Path(os.environ.get("HQ_TRADING_DIR", str(BOX / "hq-trading-system")))

# ── outputs of THIS repo ────────────────────────────────────────────
QUESTIONS = HERE / "questions.yaml"
ANSWERS_JSON = HERE / "answers.json"
ANSWERS_MD = HERE / "ANSWERS.md"
JOURNAL = HERE / "journal"
PROPOSALS = HERE / "proposals"

# ── sibling evidence ────────────────────────────────────────────────
MACRO_SCORECARDS = MACRO_GPU_LAB / "journal" / "scorecards"
MACRO_FLAGS = MACRO_GPU_LAB / "journal" / "flags"
MACRO_RUNS = MACRO_GPU_LAB / "journal" / "runs"
MACRO_PANELS = MACRO_GPU_LAB / "data"

QLIB_SCORECARDS = QLIB_LAB / "journal" / "scorecards"
QLIB_FLAGS = QLIB_LAB / "journal" / "flags"
QLIB_SCORE_HISTORY = QLIB_LAB / "journal" / "exports" / "qlib_score_history.parquet"
QLIB_PRICES = QLIB_LAB / "csv"

ALPACA_RESULTS_MD = ALPACA_LAB / "journal" / "RESULTS.md"
ALPACA_SCORECARDS = ALPACA_LAB / "journal" / "scorecards"
ALPACA_FLAGS = ALPACA_LAB / "journal" / "flags"
ALPACA_NEWS_SCORES = ALPACA_LAB / "market_data" / "scores" / "news_scores.parquet"
ALPACA_BARS_1DAY = ALPACA_LAB / "market_data" / "raw" / "bars_1Day"

HQ_SENTIMENT_RECON = HQ / "journal" / "sentiment" / "reconciliation"
HQ_CONSENSUS = HQ / "journal" / "consensus"
HQ_MODEL_VS_NEWS = HQ / "journal" / "macro" / "model_vs_news"
HQ_VETO_FLAG = HQ / "journal" / "sentiment" / "veto_flag.json"

FLAG_FILES = {
    "macro_gpu_lab/tv_signal": MACRO_FLAGS / "tv_signal.json",
    "macro_gpu_lab/macro_gpu_state": MACRO_FLAGS / "macro_gpu_state.json",
    "qlib_lab/qlib_state": QLIB_FLAGS / "qlib_state.json",
    "qlib_lab/vol_desk_state": QLIB_FLAGS / "vol_desk_state.json",
    "hq/veto_flag": HQ_VETO_FLAG,
}
