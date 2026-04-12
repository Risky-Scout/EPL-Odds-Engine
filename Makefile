PYTHON ?= python3
CONFIG ?= configs/epl_walkforward.yaml

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e .

sync:
	epl-pmf sync-historical --config "$(CONFIG)"

backtest:
	epl-pmf backtest --config "$(CONFIG)"

report:
	epl-pmf report --config "$(CONFIG)"

live:
	epl-pmf live-snapshot --config "$(CONFIG)"

sample:
	epl-pmf build-sample-run --config "$(CONFIG)"

notebook:
	epl-pmf build-notebook --config "$(CONFIG)"

test:
	$(PYTHON) -m unittest discover -s tests -p "test_*.py"

format:
	@echo "Use your formatter of choice, for example: ruff format . && ruff check --fix ."
