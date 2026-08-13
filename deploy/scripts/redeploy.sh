#!/usr/bin/env bash
# =============================================================================
# Restart or re-image a single pod through the RunPod REST API.
#
#   export RUNPOD_API_KEY=rpa_...
#   ./deploy/scripts/redeploy.sh restart <POD_ID>
#   ./deploy/scripts/redeploy.sh image   <POD_ID> ghcr.io/org/repo/crm-vision:<sha>
#   ./deploy/scripts/redeploy.sh status  <POD_ID>
#
# `restart` is the one you want after pushing code to a pod that has CODE_REF set
# — the entrypoint re-pulls Python on boot, so this is a full deploy in ~15 s.
#
# `image` is for dependency changes: point the pod at a new SHA tag. It restarts
# as part of applying the change.
#
# NOTE: RunPod's API surface changes from time to time. If a call 404s, check the
# current REST docs and adjust the paths below — the shape of this script (one
# pod at a time, explicit, no fan-out) is the part worth keeping.
# =============================================================================
set -euo pipefail

API="${RUNPOD_API_BASE:-https://rest.runpod.io/v1}"
KEY="${RUNPOD_API_KEY:?export RUNPOD_API_KEY first}"
ACTION="${1:-}"
POD_ID="${2:-}"

die() { echo "error: $*" >&2; exit 1; }
call() {
  local method="$1" path="$2" body="${3:-}"
  if [ -n "$body" ]; then
    curl -sS -X "$method" "${API}${path}" \
      -H "Authorization: Bearer ${KEY}" -H "Content-Type: application/json" -d "$body"
  else
    curl -sS -X "$method" "${API}${path}" -H "Authorization: Bearer ${KEY}"
  fi
}

[ -n "$ACTION" ] || die "usage: $0 {restart|image|status} <POD_ID> [image-tag]"
[ -n "$POD_ID" ] || die "POD_ID required"

case "$ACTION" in
  status)
    call GET "/pods/${POD_ID}"
    echo
    ;;

  restart)
    echo "restarting ${POD_ID}..."
    call POST "/pods/${POD_ID}/restart"
    echo
    echo "poll the pod's /healthz until it answers, then /readyz until it 200s."
    ;;

  image)
    NEW_IMAGE="${3:-}"
    [ -n "$NEW_IMAGE" ] || die "image tag required"
    case "$NEW_IMAGE" in
      *:latest|*:prod)
        die "refusing a moving tag ('${NEW_IMAGE##*:}') — pin a SHA so you can roll back" ;;
    esac
    echo "pointing ${POD_ID} at ${NEW_IMAGE}..."
    call PATCH "/pods/${POD_ID}" "{\"imageName\":\"${NEW_IMAGE}\"}"
    echo
    echo "restarting to pull it..."
    call POST "/pods/${POD_ID}/restart"
    echo
    ;;

  *)
    die "unknown action '${ACTION}'"
    ;;
esac
