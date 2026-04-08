#!/usr/bin/env bash

# ── OpenEnv Pre-Validation Script (Custom) ─────────────────────────────────────

set -euo pipefail

# ── CONFIG ─────────────────────────────────────────────────────────────────────
PING_URL="${1:-}"
REPO_DIR="${2:-.}"
DOCKER_TIMEOUT=600

# ── COLORS ─────────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

log()  { echo -e "[INFO] $1"; }
pass() { echo -e "${GREEN}[PASS]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

# ── VALIDATE INPUT ─────────────────────────────────────────────────────────────
if [ -z "$PING_URL" ]; then
  echo "Usage: ./prevalidate.sh <HF_SPACE_URL> [repo_dir]"
  exit 1
fi

PING_URL="${PING_URL%/}"
REPO_DIR="$(cd "$REPO_DIR" && pwd)"

echo ""
echo -e "${BOLD}========================================${NC}"
echo -e "${BOLD}   OpenEnv Pre-Validation Script${NC}"
echo -e "${BOLD}========================================${NC}"
echo ""

log "Repo: $REPO_DIR"
log "HF Space: $PING_URL"
echo ""

# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — CHECK HF SPACE
# ──────────────────────────────────────────────────────────────────────────────
log "Step 1: Checking HF Space (/reset endpoint)..."

HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
  -X POST "$PING_URL/reset" \
  -H "Content-Type: application/json" \
  -d '{}' \
  --max-time 20 || echo "000")

if [ "$HTTP_CODE" == "200" ]; then
  pass "HF Space is LIVE"
else
  fail "HF Space check failed (HTTP $HTTP_CODE)"
  warn "Make sure your Space is deployed and running"
  exit 1
fi

# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — DOCKER BUILD
# ──────────────────────────────────────────────────────────────────────────────
log "Step 2: Docker build..."

if ! command -v docker &>/dev/null; then
  fail "Docker not installed"
  warn "Install Docker Desktop"
  exit 1
fi

if [ -f "$REPO_DIR/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR"
elif [ -f "$REPO_DIR/server/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR/server"
else
  fail "No Dockerfile found"
  exit 1
fi

log "Using context: $DOCKER_CONTEXT"

if docker build "$DOCKER_CONTEXT" >/dev/null 2>&1; then
  pass "Docker build successful"
else
  fail "Docker build failed"
  exit 1
fi

# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 — OPENENV VALIDATE
# ──────────────────────────────────────────────────────────────────────────────
log "Step 3: Running openenv validate..."

if ! command -v openenv &>/dev/null; then
  fail "openenv not installed"
  warn "Run: pip install openenv-core"
  exit 1
fi

if (cd "$REPO_DIR" && openenv validate); then
  pass "openenv validation passed"
else
  fail "openenv validation failed"
  exit 1
fi

# ──────────────────────────────────────────────────────────────────────────────
# DONE
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}All checks passed ${NC}"
echo -e "${GREEN}Your project is ready for submission ${NC}"
echo ""