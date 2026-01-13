.PHONY: help build up down restart logs clean

help:
	@echo "Harper Docker Commands:"
	@echo "  make build    - Build Docker images"
	@echo "  make up       - Start services"
	@echo "  make down     - Stop services"
	@echo "  make restart  - Restart services"
	@echo "  make logs     - View logs"
	@echo "  make clean    - Remove containers, volumes, and images"

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

restart:
	docker compose restart

logs:
	docker compose logs -f

logs-server:
	docker compose logs -f harper-server

logs-client:
	docker compose logs -f harper-client

clean:
	docker compose down -v
	docker system prune -f

ps:
	docker compose ps

shell-server:
	docker compose exec harper-server bash

shell-client:
	docker compose exec harper-client sh

# --- RunPod ---

create-faster-whisper-pod:
	uv run python scripts/runpod/create_faster_whisper_pod.py

create-orpheus-pod:
	uv run python scripts/runpod/create_orpheus_pod.py

create-harper-pod:
	uv run python scripts/runpod/create_harper_pod.py
