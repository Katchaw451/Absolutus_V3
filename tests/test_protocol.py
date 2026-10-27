"""Verify that the runner and analyser work end-to-end on one tiny dev cell."""
import json
import subprocess
import sys
from pathlib import Path


def test_runner_produces_valid_json(tmp_path):
    out = tmp_path / "smoke.json"
    cmd = [
        sys.executable, "scripts/run_cell.py",
        "--family", "planted", "--n", "12", "--k", "3", "--seed", "0",
        "--solver", "greedy", "--out", str(out),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    rec = json.loads(out.read_text().strip().splitlines()[-1])
    for field in ["family", "n", "k", "seed", "solver", "wall_time_s", "peak_mb",
                  "objective", "feasible", "n_evals", "energy", "env"]:
        assert field in rec, f"missing {field}"
    assert rec["family"] == "planted" and rec["n"] == 12
    assert rec["feasible"] is True
    assert rec["env"]["git_head"] != ""


def test_analyze_produces_csv(tmp_path):
    raw = tmp_path / "raw.jsonl"
    # fake 20 seeds of one cell for the analyzer (this is a test, not a real run)
    lines = []
    for s in range(20):
        lines.append(json.dumps({
            "family": "planted", "n": 12, "k": 3, "seed": 100 + s,
            "solver": "greedy", "engine": "statevector", "p": 1,
            "lambda_coh": 1.0, "lambda_red": 0.1,
            "time_limit": 30.0, "n_evals_budget": 10000,
            "success": True, "error": None,
            "wall_time_s": 0.01, "peak_mb": 10.0,
            "energy": -1.0, "objective": 3.0 + 0.01 * s,
            "feasible": True, "feasible_fraction": 1.0, "n_evals": 10,
            "extra": {}, "env": {"git_head": "test"},
        }))
    raw.write_text("\n".join(lines) + "\n")
    out = tmp_path / "summary.csv"
    r = subprocess.run(
        [sys.executable, "scripts/analyze.py", "--in", str(raw), "--out", str(out)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    text = out.read_text()
    assert "obj_ci_lo" in text and "obj_ci_hi" in text
    assert "greedy" in text
