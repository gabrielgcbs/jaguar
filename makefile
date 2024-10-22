.PHONY: run format

install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "Installation complete."

run:
	@echo "Running Jaguar!"
	python run.py

format:
	@echo "Running black formatter"
	black **/*.py


