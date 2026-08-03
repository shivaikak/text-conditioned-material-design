.PHONY: validate test baseline dry-run

validate:
	python scripts/validate_setup.py

test:
	pytest -q

baseline:
	python scripts/init_baseline.py

dry-run:
	python scripts/scheduler.py submit baseline --max-parallel 1 --dry-run
