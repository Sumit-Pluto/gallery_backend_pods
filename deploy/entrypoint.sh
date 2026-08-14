#!/usr/bin/env bash
# =============================================================================
# Shared container entrypoint for every crm-ai-backend service.
#
# This is the piece that makes "change one file, redeploy in 15 seconds" work.
#
# The slow part of a rebuild is torch + diffusers + CUDA — never your handler
# code. So dependencies are baked into the image, and Python code is optionally
# refreshed from GitHub at boot:
#
#   deps changed      -> CI builds a new image  (~15 min, one service)
#   only code changed -> restart the pod        (~15 s,  no build at all)
#
# It is FAIL-OPEN by design: if GitHub is unreachable or the ref is bad, the
# baked-in code from the image runs. A network blip must never take a pod down.
#
# Env:
#   SERVICE      gateway | vision | diffusion | cpu_tasks   (set in the Dockerfile)
#   CODE_REPO    https://github.com/Sumit-Pluto/gallery_backend_pods.git
#                private repo: https://x-access-token:<PAT>@github.com/Sumit-Pluto/gallery_backend_pods.git
#   CODE_REF     branch, tag or commit SHA. Unset = use baked code (the default).
#   PORT         defaults to 8000
#   WEB_CONCURRENCY  uvicorn workers. Keep at 1 for GPU services.
# =============================================================================
set -uo pipefail

SERVICE="${SERVICE:?SERVICE must be set in the Dockerfile}"
PORT="${PORT:-8000}"
WEB_CONCURRENCY="${WEB_CONCURRENCY:-1}"

log() { echo "{\"level\":\"INFO\",\"service\":\"$SERVICE\",\"message\":\"[boot] $*\"}"; }
warn() { echo "{\"level\":\"WARNING\",\"service\":\"$SERVICE\",\"message\":\"[boot] $*\"}"; }

if [ -n "${CODE_REF:-}" ] && [ -n "${CODE_REPO:-}" ]; then
  log "refreshing code from ${CODE_REF}"
  rm -rf /tmp/src
  if git clone --filter=blob:none --no-checkout --quiet "$CODE_REPO" /tmp/src \
     && git -C /tmp/src checkout --quiet "$CODE_REF"; then
    SHA="$(git -C /tmp/src rev-parse --short HEAD)"
    if [ -d "/tmp/src/services/${SERVICE}/app" ]; then
      cp -a "/tmp/src/services/${SERVICE}/app/." /app/app/
      # crm_common is pip-installed; overwrite it in place so a shared-lib fix
      # ships the same way a service fix does.
      COMMON_DIR="$(python3 -c 'import crm_common, os; print(os.path.dirname(crm_common.__file__))' 2>/dev/null || true)"
      if [ -n "$COMMON_DIR" ] && [ -d /tmp/src/libs/common/crm_common ]; then
        cp -a /tmp/src/libs/common/crm_common/. "$COMMON_DIR/"
      fi
      export BUILD_SHA="$SHA"
      log "code refreshed to ${SHA}"
    else
      warn "services/${SERVICE}/app not found in ${CODE_REF} — using baked code"
    fi
  else
    warn "git fetch failed — using baked code (this is not fatal)"
  fi
  rm -rf /tmp/src
else
  log "CODE_REF unset — running baked image code (build ${BUILD_SHA:-dev})"
fi

log "starting uvicorn on :${PORT} (workers=${WEB_CONCURRENCY})"
exec uvicorn app.main:app \
  --host 0.0.0.0 \
  --port "$PORT" \
  --workers "$WEB_CONCURRENCY" \
  --timeout-keep-alive 75 \
  --no-access-log
