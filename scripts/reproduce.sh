#!/usr/bin/env bash
# Regenerate every table and figure from scratch under results/test/.
# Reads tuning from results/tuning/ (regenerated on dev seeds if missing).
set -u
cd "$(dirname "$0")/.."

echo "=== reproduce: venv ==="
[ -d .venv ] || python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -e ".[dev]" -q

echo "=== reproduce: generate frozen test instances ==="
mkdir -p results/instances
python - <<'PY'
import hashlib, json
from pathlib import Path
from modbench.generators import random_sparse, planted_scale_free, planted_decoy_hubs
from modbench.instances import save

GEN = {"random": random_sparse, "planted": planted_scale_free, "decoy": planted_decoy_hubs}
root = Path("results/instances"); root.mkdir(parents=True, exist_ok=True)
manifest = {}
for fam, fn in GEN.items():
    for n in (12, 20, 30, 40, 60):
        k = max(3, round(n/5))
        for seed in range(100, 120):
            inst = fn(n=n, k=k, seed=seed)
            base = root / f"{fam}_n{n}_k{k}_s{seed}"
            save(inst, base)
            h = hashlib.sha256(Path(str(base) + ".npz").read_bytes()).hexdigest()
            manifest[str(base.relative_to(root))] = h
Path(root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
print(f"wrote {len(manifest)} instance files + manifest.json")
PY

echo "=== reproduce: tune QAOA on dev (if missing) ==="
for fam in random planted decoy; do
  for n in 12 20; do
    for p in 1 2; do
      if [ ! -f "results/tuning/${fam}_n${n}_p${p}.json" ]; then
        python scripts/tune_qaoa_dev.py --family "$fam" --n "$n" --p "$p"
      fi
    done
  done
done

echo "=== reproduce: run test matrix ==="
python scripts/run_matrix.py --kind classical --seeds test
python scripts/run_matrix.py --kind exact     --seeds test --max-n 20
python scripts/run_matrix.py --kind qaoa-sv   --seeds test
python scripts/run_matrix.py --kind qaoa-mps  --seeds test --sizes 12 20 --ps 1 --chi 2 4 8 16 32

echo "=== reproduce: aggregate + figures + report ==="
python scripts/analyze.py \
  --in $(find results/test -name '*.jsonl' ! -empty) \
  --out results/test/summary_all.csv
python scripts/make_figures.py --summaries results/test/summary_all.csv \
  --outdir results/test/figures
if [ -x scripts/build_report.py ]; then
  python scripts/build_report.py --summary results/test/summary_all.csv \
    --out docs/REPORT.md || echo "build_report.py not ready yet"
fi
echo "=== reproduce: DONE ==="
