#!/usr/bin/env bash
# Idempotent: download the IC-LoRA Outpaint weights from HuggingFace and
# register under id "ic-lora-outpaint" in loras/registry.json.
# Safe to re-run — skips the download if the file is already present and the
# registry entry exists.
#
# Usage: bash scripts/register_outpaint_lora.sh
set -euo pipefail

REPO="oumoumad/LTX-2.3-22b-IC-LoRA-Outpaint"
WEIGHT_FILE="ltx-2.3-22b-ic-lora-outpaint.safetensors"
LORA_ID="ic-lora-outpaint"
BACKEND_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LORAS_DIR="$BACKEND_ROOT/loras"
REGISTRY="$LORAS_DIR/registry.json"

mkdir -p "$LORAS_DIR"

# 1. Download via hf CLI (skips if the file already matches by hash)
if [ ! -f "$LORAS_DIR/$WEIGHT_FILE" ]; then
    echo "Downloading $REPO ..."
    hf download "$REPO" --include "*.safetensors" --local-dir "$LORAS_DIR"
else
    echo "Weights already present: $LORAS_DIR/$WEIGHT_FILE"
fi

# 2. Canonical symlink — registry resolves path via {id}.safetensors
CANONICAL="$LORAS_DIR/$LORA_ID.safetensors"
if [ ! -e "$CANONICAL" ]; then
    ln -s "$WEIGHT_FILE" "$CANONICAL"
    echo "Symlinked $LORA_ID.safetensors → $WEIGHT_FILE"
fi

# 3. Registry entry
export REGISTRY WEIGHT_FILE LORAS_DIR LORA_ID
if python3 -c "
import json, sys, os
from pathlib import Path
reg = json.loads(Path(os.environ['REGISTRY']).read_text())
ids = {l['id'] for l in reg.get('loras', [])}
sys.exit(0 if os.environ['LORA_ID'] in ids else 1)
" ; then
    echo "Registry entry '$LORA_ID' already present — skipping."
else
    python3 - <<'PY'
import json, os
from datetime import datetime, timezone
from pathlib import Path

registry_path = Path(os.environ["REGISTRY"])
weight_file = os.environ["WEIGHT_FILE"]
weight_path = Path(os.environ["LORAS_DIR"]) / weight_file
size = weight_path.stat().st_size

data = json.loads(registry_path.read_text()) if registry_path.exists() else {"loras": []}
data["loras"].append({
    "id": os.environ["LORA_ID"],
    "name": "IC-LoRA Outpaint",
    "filename": weight_file,
    "base_model": "ltx-2.3",
    "size_bytes": size,
    "uploaded_at": datetime.now(timezone.utc).isoformat(),
    "description": (
        "Video outpaint IC-LoRA by oumoumad. Fills pure-black regions in a "
        "letterboxed source video with temporally consistent content. "
        "Used by /v2/video-outpaint."
    ),
    "trigger_word": None,
    "strategy": "ic_lora_outpaint",
})
registry_path.write_text(json.dumps(data, indent=2))
print(f"Registered {weight_file} as {os.environ['LORA_ID']} ({size/1e9:.2f} GB).")
PY
fi

echo ""
echo "Next step: restart taco-backend so the in-memory registry re-loads:"
echo "  systemctl --user restart taco-backend"
echo ""
echo "Verify:"
echo "  curl -s http://localhost:8090/v1/loras -H 'Authorization: Bearer \$API_KEY' | jq '.loras[] | select(.id==\"ic-lora-outpaint\")'"
