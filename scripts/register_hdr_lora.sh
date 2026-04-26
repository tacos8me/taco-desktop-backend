#!/usr/bin/env bash
# Idempotent: download the IC-LoRA HDR weights from HuggingFace and register
# under id "ic-lora-hdr" in loras/registry.json. Mirror of
# register_outpaint_lora.sh — same conditioning architecture, different LoRA.
# Safe to re-run.
#
# Usage: bash scripts/register_hdr_lora.sh
set -euo pipefail

REPO="Lightricks/LTX-2.3-22b-IC-LoRA-HDR"
WEIGHT_FILE="ltx-2.3-22b-ic-lora-hdr.safetensors"
LORA_ID="ic-lora-hdr"
BACKEND_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LORAS_DIR="$BACKEND_ROOT/loras"
REGISTRY="$LORAS_DIR/registry.json"

mkdir -p "$LORAS_DIR"

# 1. Download via hf CLI (skips if the file already matches by hash). The repo
#    publishes a single .safetensors at the root; we accept whatever filename
#    the upstream uses and rename to our canonical WEIGHT_FILE if needed.
if [ ! -f "$LORAS_DIR/$WEIGHT_FILE" ]; then
    echo "Downloading $REPO ..."
    hf download "$REPO" --include "*.safetensors" --local-dir "$LORAS_DIR"
    # If upstream filename differs, rename to our canonical name. The
    # downloaded file lands in $LORAS_DIR — find any newly-arrived
    # *.safetensors that doesn't match an already-known LoRA filename.
    if [ ! -f "$LORAS_DIR/$WEIGHT_FILE" ]; then
        # Best-effort rename: the upstream is "Lightricks" so likely the file
        # is named like ltx-2.3-22b-ic-lora-hdr.safetensors already. If not,
        # operator should rename manually — we don't want to globbingly move
        # the wrong file.
        echo "WARNING: expected $WEIGHT_FILE not found after download." >&2
        echo "Files present in $LORAS_DIR:" >&2
        ls -lh "$LORAS_DIR"/*.safetensors >&2 || true
        echo "Rename the new file to $WEIGHT_FILE and re-run this script." >&2
        exit 1
    fi
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
    "name": "IC-LoRA HDR",
    "filename": weight_file,
    "base_model": "ltx-2.3",
    "size_bytes": size,
    "uploaded_at": datetime.now(timezone.utc).isoformat(),
    "description": (
        "Video HDR-expansion IC-LoRA by Lightricks. Promotes LDR clips to "
        "expanded dynamic range while preserving temporal coherence. Same "
        "video-conditioning architecture as ic-lora-outpaint (source video "
        "VAE-encoded as reference latent). Used by /v2/video-hdr."
    ),
    "trigger_word": None,
    "strategy": "ic_lora_hdr",
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
echo "  curl -s http://localhost:8090/v1/loras -H 'Authorization: Bearer \$API_KEY' | jq '.loras[] | select(.id==\"ic-lora-hdr\")'"
