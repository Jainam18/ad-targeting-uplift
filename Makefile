# The ".nosync" suffix keeps iCloud from syncing the venv. Synced venvs get the macOS "hidden"
# flag on .pth files, which Python 3.13 skips, breaking the editable install.
VENV ?= .venv.nosync
KERNEL := criteo-uplift
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

RAW_CSV := data/raw/criteo-uplift-v2.1.csv
PARQUET := data/processed/criteo.parquet
RAW_URL := https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/criteo-research-uplift-v2.1.csv.gz
export PYTHONPATH := $(CURDIR)/src

NOTEBOOKS := $(sort $(wildcard notebooks/0[1-9]_*.ipynb))

.PHONY: all setup download data analysis test lint format clean

all: data analysis

$(PY):
	mkdir -p $(dir $(VENV))
	python3.13 -m venv $(VENV)

setup: $(PY)
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	$(PY) -m ipykernel install --user --name $(KERNEL) --display-name "Python 3.13 ($(KERNEL))"

download: $(RAW_CSV)

$(RAW_CSV):
	mkdir -p data/raw
	curl -L --fail -o $(RAW_CSV).gz $(RAW_URL)
	gunzip -c $(RAW_CSV).gz > $(RAW_CSV)
	rm $(RAW_CSV).gz

data: $(PARQUET)

$(PARQUET): $(RAW_CSV) src/uplift/data.py
	$(PY) -m uplift.data

analysis:
	@if [ -z "$(NOTEBOOKS)" ]; then echo "No analysis notebooks yet."; fi
	@for nb in $(NOTEBOOKS); do \
		echo "Executing $$nb"; \
		$(VENV)/bin/jupyter nbconvert --to notebook --execute --inplace \
			--ExecutePreprocessor.kernel_name=$(KERNEL) "$$nb" || exit 1; \
	done

test:
	$(VENV)/bin/pytest -q

lint:
	$(VENV)/bin/ruff check .
	$(VENV)/bin/ruff format --check .

format:
	$(VENV)/bin/ruff check --fix .
	$(VENV)/bin/ruff format .

clean:
	rm -rf data/processed/*.parquet results/*.json reports/figures/*.png reports/figures/*.svg
