.PHONY: test smoke pilot tune reproduce clean

test:
	pytest -q

smoke:
	python scripts/run_cell.py --family planted --n 12 --k 3 --seed 0 \
	    --solver greedy --out results/dev/smoke.jsonl

# Small pilot: proves the pipeline works end-to-end. NOT reported (dev).
pilot:
	python scripts/run_matrix.py --kind classical --seeds dev --sizes 12 --families planted
	python scripts/analyze.py --in $$(find results/dev -name '*.jsonl' ! -empty) \
	    --out results/dev/summary_all.csv

# Tune QAOA on dev (required before qaoa-sv/qaoa-mps runs).
tune:
	for fam in random planted decoy; do \
	  for n in 12 20; do \
	    for p in 1 2; do \
	      python scripts/tune_qaoa_dev.py --family $$fam --n $$n --p $$p; \
	    done; \
	  done; \
	done

reproduce:
	bash scripts/reproduce.sh

clean:
	rm -rf results/dev results/test results/tuning
