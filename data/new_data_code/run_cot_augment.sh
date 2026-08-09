#!/bin/sh
# Deprecated compatibility wrapper. Configure LLM_* variables and pass CLI args.

set -eu

REPO_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
echo "warning: use scripts/augment_ldm_data.py directly" >&2
exec python "$REPO_ROOT/scripts/augment_ldm_data.py" "$@"
