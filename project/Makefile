# Makefile for Breast Cancer Detection MLOps Project

.PHONY: help setup lint test train docker_build docker_run docker_push format

help:
	@echo "Available commands:"
	@echo "  make setup         - Install Python dependencies"
	@echo "  make lint          - Run code linter (black, flake8)"
	@echo "  make format        - Format code using black"
	@echo "  make test          - Run all tests"
	@echo "  make train         - Run training pipeline"
	@echo "  make docker_build  - Build Docker image"
	@echo "  make docker_run    - Run FastAPI container"
	@echo "  make docker_push   - Push Docker image to AWS ECR"

setup:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pre-commit install

lint:
	black src/ tests/
	flake8 src/ tests/

format:
	black src/ tests/

test:
	pytest tests/

train:
	python src/training/train.py

docker_build:
	docker build -t breast-cancer-api .

docker_run:
	docker run -p 8000:8000 breast-cancer-api

docker_push:
	# Replace with your actual AWS ECR URL
	docker tag breast-cancer-api:latest <your_ecr_repo_url>/breast-cancer-api:latest
	docker push <your_ecr_repo_url>/breast-cancer-api:latest
