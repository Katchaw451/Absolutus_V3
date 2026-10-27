.PHONY: test reproduce clean smoke

test:
	pytest -q

smoke:
	python scripts/run_cell.py --family planted --n 12 --k 3 --seed 0 \
	    --solver greedy --out results/dev/smoke.jsonl

reproduce:
	bash scripts/reproduce.sh

clean:
	rm -rf results/test results/dev
