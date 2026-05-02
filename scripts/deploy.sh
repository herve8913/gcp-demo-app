#!/usr/bin/env bash
# Deploy gcp-demo-app to a single GCP Compute Engine VM with L4 GPU.
# Usage: ./scripts/deploy.sh [--instance NAME] [--zone ZONE]
#
# Prerequisites on VM: Docker, NVIDIA driver, NVIDIA Container Toolkit
# All are pre-installed on the chamber-agent-test VM.

set -euo pipefail

INSTANCE="${1:-chamber-agent-test}"
ZONE="${2:-us-central1-a}"
REMOTE_DIR="~/gcp-demo-app"
COMPOSE_FILE="docker-compose.unified.yml"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

ssh_cmd() {
    gcloud compute ssh "$INSTANCE" --zone="$ZONE" --command="$1" 2>&1
}

# ── Step 0: Verify VM is reachable ──────────────────────────────────
info "Checking VM '$INSTANCE' in zone '$ZONE'..."
STATUS=$(gcloud compute instances describe "$INSTANCE" --zone="$ZONE" --format="value(status)" 2>/dev/null) || error "VM not found"
if [ "$STATUS" != "RUNNING" ]; then
    error "VM is $STATUS, not RUNNING. Start it first: gcloud compute instances start $INSTANCE --zone=$ZONE"
fi
info "VM is RUNNING"

# ── Step 1: Verify prerequisites ────────────────────────────────────
info "Verifying Docker and NVIDIA setup on VM..."
ssh_cmd "nvidia-smi > /dev/null 2>&1 && docker --version > /dev/null 2>&1" || error "Missing Docker or NVIDIA driver on VM"
info "Prerequisites OK"

# ── Step 2: Add user to docker group if needed ──────────────────────
info "Ensuring user is in docker group..."
ssh_cmd "groups | grep -q docker || sudo usermod -aG docker \$USER" || true

# ── Step 3: Ensure dcgm-exporter is running (for chamber-agent-standalone) ──
info "Checking dcgm-exporter..."
if ssh_cmd "sudo docker ps --format '{{.Names}}' | grep -q dcgm-exporter"; then
    info "dcgm-exporter already running"
else
    info "Starting dcgm-exporter..."
    ssh_cmd "sudo docker rm dcgm-exporter 2>/dev/null || true"
    ssh_cmd "sudo docker run -d --name dcgm-exporter --restart unless-stopped --gpus all --cap-add SYS_ADMIN -p 9400:9400 nvcr.io/nvidia/k8s/dcgm-exporter:3.3.8-3.6.0-ubuntu22.04"
    info "dcgm-exporter started"
fi

# ── Step 4: Transfer project files ──────────────────────────────────
info "Transferring project files to VM..."

# Create a temp tarball excluding unnecessary files
TARBALL=$(mktemp /tmp/gcp-demo-app-XXXX.tar.gz)
tar -czf "$TARBALL" \
    -C "$PROJECT_ROOT" \
    --exclude='.git' \
    --exclude='terraform' \
    --exclude='__pycache__' \
    --exclude='.terraform' \
    --exclude='*.tfstate*' \
    --exclude='.venv' \
    --exclude='node_modules' \
    .

gcloud compute scp "$TARBALL" "$INSTANCE:~/gcp-demo-app.tar.gz" --zone="$ZONE"
rm -f "$TARBALL"

# Extract on VM
ssh_cmd "mkdir -p $REMOTE_DIR && tar -xzf ~/gcp-demo-app.tar.gz -C $REMOTE_DIR && rm -f ~/gcp-demo-app.tar.gz"
info "Files transferred"

# ── Step 5: Build and launch containers ─────────────────────────────
info "Building and launching containers (this may take several minutes on first run)..."
ssh_cmd "cd $REMOTE_DIR && sudo docker compose -f $COMPOSE_FILE down --remove-orphans 2>/dev/null || true"
ssh_cmd "cd $REMOTE_DIR && sudo docker compose -f $COMPOSE_FILE up --build -d"
info "Containers launched"

# ── Step 6: Wait for health checks ─────────────────────────────────
EXTERNAL_IP=$(gcloud compute instances describe "$INSTANCE" --zone="$ZONE" --format="value(networkInterfaces[0].accessConfigs[0].natIP)")

info "Waiting for services to become healthy..."
MAX_ATTEMPTS=40
INTERVAL=15

check_health() {
    local url="$1"
    local name="$2"
    curl -sf --max-time 5 "$url" > /dev/null 2>&1
}

for ((i=1; i<=MAX_ATTEMPTS; i++)); do
    DISTILBERT_OK=false
    RESNET_OK=false
    METRICS_OK=false
    PROM_OK=false
    WEBAPP_OK=false

    check_health "http://$EXTERNAL_IP:8001/health" "distilbert" && DISTILBERT_OK=true
    check_health "http://$EXTERNAL_IP:8002/health" "resnet50" && RESNET_OK=true
    check_health "http://$EXTERNAL_IP:8080/metrics" "metrics-agent" && METRICS_OK=true
    check_health "http://$EXTERNAL_IP:9090/-/healthy" "prometheus" && PROM_OK=true
    check_health "http://$EXTERNAL_IP:5001/" "webapp" && WEBAPP_OK=true

    echo -n "  [$i/$MAX_ATTEMPTS] "
    $DISTILBERT_OK && echo -n "distilbert:OK " || echo -n "distilbert:-- "
    $RESNET_OK && echo -n "resnet50:OK " || echo -n "resnet50:-- "
    $METRICS_OK && echo -n "metrics:OK " || echo -n "metrics:-- "
    $PROM_OK && echo -n "prometheus:OK " || echo -n "prometheus:-- "
    $WEBAPP_OK && echo -n "webapp:OK" || echo -n "webapp:--"
    echo

    if $DISTILBERT_OK && $RESNET_OK && $METRICS_OK && $PROM_OK && $WEBAPP_OK; then
        echo
        info "All services are healthy!"
        break
    fi

    if [ "$i" -eq "$MAX_ATTEMPTS" ]; then
        echo
        warn "Timed out waiting for all services. Check logs with:"
        warn "  gcloud compute ssh $INSTANCE --zone=$ZONE --command='sudo docker compose -f $REMOTE_DIR/$COMPOSE_FILE logs'"
        break
    fi

    sleep "$INTERVAL"
done

# ── Step 7: Print endpoints ─────────────────────────────────────────
echo
info "========================================="
info "  Deployment complete!"
info "========================================="
echo
echo "  Webapp:        http://$EXTERNAL_IP:5001"
echo "  DistilBERT:    http://$EXTERNAL_IP:8001/health"
echo "  ResNet-50:     http://$EXTERNAL_IP:8002/health"
echo "  Metrics Agent: http://$EXTERNAL_IP:8080/metrics"
echo "  Prometheus:    http://$EXTERNAL_IP:9090"
echo
echo "  SSH:           gcloud compute ssh $INSTANCE --zone=$ZONE"
echo "  Logs:          gcloud compute ssh $INSTANCE --zone=$ZONE --command='cd $REMOTE_DIR && sudo docker compose -f $COMPOSE_FILE logs -f'"
echo
