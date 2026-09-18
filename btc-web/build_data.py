import csv
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "btc-trend-research"
summary = json.loads((SOURCE / "summary.json").read_text())
v2 = json.loads((SOURCE / "v2" / "summary.json").read_text())
v3 = json.loads((SOURCE / "v3" / "web_model.json").read_text())
manifest = json.loads((SOURCE / "data_manifest.json").read_text())
if hashlib.sha256((SOURCE / "btc_daily.csv").read_bytes()).hexdigest() != manifest["sha256"]:
    raise ValueError("Historical data checksum mismatch")
if v3["dataSha256"] != manifest["sha256"]:
    raise ValueError("V3 model and historical data snapshot differ")
if v3["protocolSha256"] != hashlib.sha256((SOURCE / "v3" / "protocol.json").read_bytes()).hexdigest():
    raise ValueError("V3 protocol changed after model export")
with (SOURCE / "btc_daily.csv").open() as handle:
    candles = [[row["date"], *[float(row[key]) for key in ["open", "high", "low", "close", "volume"]]]
               for row in csv.DictReader(handle)]
curves = {}
with (SOURCE / "holdout_daily.csv").open() as handle:
    for row in csv.DictReader(handle):
        series = curves.setdefault(row["strategy"], [[row["execution_date"], 1.0, 0.0]])
        valuation = (date.fromisoformat(row["execution_date"]) + timedelta(days=1)).isoformat()
        equity = float(row["equity"])
        series.append([valuation, equity, float(row["weight"])])
payload = {
    "candles": candles,
    "manifest": {key: manifest[key] for key in ["source", "symbol", "timezone", "first_closed_candle", "last_closed_candle", "rows", "sha256"]},
    "forecast": summary["holdout_forecast"],
    "metrics": summary["metrics"]["holdout"],
    "acceptance": summary["acceptance"],
    "curves": curves,
    "v2": {"candidateCount": v2["candidate_count"], "selected": v2["selected_candidate"]["id"],
           "results": v2["reused_history_metrics"], "goalAchieved": v2["primary_goal_achieved"]},
    "v3": v3,
}
serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
(ROOT / "data.js").write_text("window.BTC_DATA = " + serialized + ";\n")
print(f"Bundled {len(candles)} candles and {len(curves)} backtest curves into data.js")
