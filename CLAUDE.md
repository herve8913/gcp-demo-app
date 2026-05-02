# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GCP-based ML inference platform with two VMs: a GPU VM running model services (DistilBERT sentiment analysis, ResNet-50 image classification, metrics-agent, Prometheus) and a Webapp VM running a Flask frontend. Services communicate over HTTP, with OpenTelemetry metrics flowing via OTLP gRPC to a unified metrics-agent.

## Build & Deploy Commands

```bash
# Infrastructure
cd terraform && terraform init && terraform plan && terraform apply

# Local development - webapp (needs GPU_VM_IP set for model access)
GPU_VM_IP=<ip> docker compose -f webapp-compose.yml up --build

# GPU VM services (requires NVIDIA GPU + drivers)
docker compose up --build

# GPU VM management (cost control)
./scripts/gpu-start.sh    # Start VM, wait for health, print endpoints
./scripts/gpu-stop.sh     # Stop VM
./scripts/gpu-status.sh   # Check status
```

There is no test suite, linter, or CI/CD pipeline configured.

## Architecture

```
Webapp VM (Flask:5000)
    ↓ HTTP proxies to GPU VM
GPU VM:
    ├─ DistilBERT (FastAPI:8001) ──┐
    ├─ ResNet-50  (FastAPI:8002) ──┤── Send OTLP gRPC metrics to :4317
    ├─ Metrics-Agent (:8080/:4317) ←── Aggregates app + GPU metrics
    └─ Prometheus (:9090)          ←── Scrapes :8080 every 15s
```

**Metrics flow:** Model services → OTLP gRPC → metrics-agent's MetricStore → Prometheus scrapes `/metrics` endpoint. GPU metrics collected via pynvml in a polling thread within metrics-agent.

## Key Code Locations

- **Model services:** `containers/distilbert/` and `containers/resnet50/` — each has `app/main.py` (FastAPI endpoints) and `app/model.py` (model loading/inference)
- **Metrics agent:** `containers/metrics-agent/src/main.py` — unified collector with GPU polling thread, OTLP gRPC receiver, and Prometheus HTTP exporter
- **Webapp:** `containers/webapp/app.py` — Flask app proxying to model services and Prometheus
- **Insights engine:** `containers/webapp/insights.py` — trend/anomaly/correlation detection using Prometheus queries
- **Infrastructure:** `terraform/main.tf` — two GCP Compute Engine VMs, VPC, firewall rules

## Conventions

- All Python services use OpenTelemetry SDK 1.24.0 with custom histogram buckets (0.005s–10s) for latency metrics
- Custom metric names follow `ml.` prefix pattern: `ml.inference.latency`, `ml.inference.request_count`, `ml.inference.error_count`, `ml.model.load_time`
- Model services expose `/health` and `/predict` endpoints
- Webapp API routes are prefixed with `/api/`
- Environment variables: `OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_SERVICE_NAME`, `MODEL_CACHE_DIR`, `GPU_VM_IP`
