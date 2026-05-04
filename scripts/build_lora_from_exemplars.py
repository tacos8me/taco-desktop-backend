#!/usr/bin/env python3
"""v1.19.0 / Phase 1 — L2.5 exemplar LoRA build orchestrator.

Operator workflow (the LoRA-build loop closure):

  1. Operator stars 20-30 generated clips as exemplars in their
     dashboard. Each star is an INSERT into ``exemplar_set_members``.
  2. Operator hits "build LoRA" — backend ``POST
     /v1/exemplar-sets/{set_id}/build-lora`` shells out to *this*
     script with ``--execute`` once admin gating + ETA quote are
     accepted.
  3. This script resolves clip_ids → mp4 paths from history.db +
     uploads/, captions list from generations.prompt, materializes a
     temporary ltx-trainer dataset (process_videos.py →
     process_dataset.py), generates a per-build YAML from
     ``configs/exemplar_lora.yaml``, and spawns
     ``/mnt/nvme-1/repos/LTX-2/packages/ltx-trainer/scripts/train.py``.
  4. On completion: register the resulting safetensors in
     ``lora_registry`` with ``strategy="exemplar_fine_tune"``,
     ``training_runs.status='completed'``, and link
     ``exemplar_sets.last_built_lora_id``.

Defense-in-depth safety: defaults to ``dry_run=True``. ``--execute`` is
required to consume GPU. A real run is ~10-15 GPU-hours and locks
cuda:0; the operator should know exactly what they're starting.

Cancellation contract: when ``POST
/v1/exemplar-sets/{set_id}/build/{run_id}/cancel`` is called, the
backend SIGTERMs this process. The script's signal handler flips
``training_runs.status='cancelled'``, cleans the temp dataset dir, and
exits non-zero. The training_runs row itself is preserved as audit.

Mirrors the safety pattern of ``scripts/train_dpo_sft.py``.

Usage::

    # Dry-run (default). No GPU, no DB writes beyond the stub
    # training_runs row that records the intent.
    uv run python scripts/build_lora_from_exemplars.py --set-id <id>

    # Real training run.
    uv run python scripts/build_lora_from_exemplars.py --set-id <id> --execute
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config  # noqa: E402
from history_store import HistoryStore  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("build_lora_from_exemplars")


TRAINING_RUNS_DIR = _ROOT / "training_runs"
LORAS_DIR = _ROOT / "loras"
LTX_TRAINER_TRAIN_SCRIPT = Path(
    "/mnt/nvme-1/repos/LTX-2/packages/ltx-trainer/scripts/train.py"
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class BuildConfig:
    """Per-build config, derived from configs/exemplar_lora.yaml + body
    overrides on the kick-off endpoint."""

    run_id: str
    set_id: str
    base_model_path: str
    text_encoder_path: str
    rank: int
    alpha: int
    dropout: float
    target_modules: list[str]
    steps: int
    learning_rate: float
    batch_size: int
    gradient_accumulation_steps: int
    optimizer_type: str
    scheduler_type: str
    max_grad_norm: float
    enable_gradient_checkpointing: bool
    mixed_precision_mode: str
    seed: int
    first_frame_conditioning_p: float
    with_audio: bool
    validation_interval: int
    validation_prompts: list[str]
    validation_negative_prompt: str
    validation_video_dims: list[int]
    validation_frame_rate: float
    validation_seed: int
    validation_inference_steps: int
    checkpoint_interval: int
    checkpoint_keep_last_n: int
    checkpoint_precision: str
    hyperparams: dict[str, Any]


def load_config(
    path: Path,
    *,
    set_id: str,
    run_id: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> BuildConfig:
    import yaml
    raw = yaml.safe_load(path.read_text())
    if overrides:
        # Only the operator-tunable hyperparams listed in the plan
        # (rank, learning_rate, steps, base_model) flow through here.
        # Anything else is hardcoded by the YAML template.
        for key in ("rank", "learning_rate", "steps", "base_model_path"):
            if key in overrides and overrides[key] is not None:
                raw[key] = overrides[key]
    rid = run_id or f"{raw.get('run_id_prefix', 'exemplar')}-{set_id}-{int(time.time())}"
    return BuildConfig(
        run_id=rid,
        set_id=set_id,
        base_model_path=str(raw["base_model_path"]),
        text_encoder_path=str(raw["text_encoder_path"]),
        rank=int(raw["rank"]),
        alpha=int(raw.get("alpha", raw["rank"])),
        dropout=float(raw.get("dropout", 0.0)),
        target_modules=list(raw["target_modules"]),
        steps=int(raw["steps"]),
        learning_rate=float(raw["learning_rate"]),
        batch_size=int(raw.get("batch_size", 1)),
        gradient_accumulation_steps=int(raw.get("gradient_accumulation_steps", 1)),
        optimizer_type=str(raw.get("optimizer_type", "adamw")),
        scheduler_type=str(raw.get("scheduler_type", "linear")),
        max_grad_norm=float(raw.get("max_grad_norm", 1.0)),
        enable_gradient_checkpointing=bool(raw.get("enable_gradient_checkpointing", True)),
        mixed_precision_mode=str(raw.get("mixed_precision_mode", "bf16")),
        seed=int(raw.get("seed", 42)),
        first_frame_conditioning_p=float(raw.get("first_frame_conditioning_p", 0.5)),
        with_audio=bool(raw.get("with_audio", False)),
        validation_interval=int(raw.get("validation_interval", 250)),
        validation_prompts=list(raw.get("validation_prompts", [])),
        validation_negative_prompt=str(raw.get("validation_negative_prompt", "")),
        validation_video_dims=list(raw.get("validation_video_dims", [576, 576, 49])),
        validation_frame_rate=float(raw.get("validation_frame_rate", 24.0)),
        validation_seed=int(raw.get("validation_seed", 42)),
        validation_inference_steps=int(raw.get("validation_inference_steps", 30)),
        checkpoint_interval=int(raw.get("checkpoint_interval", 250)),
        checkpoint_keep_last_n=int(raw.get("checkpoint_keep_last_n", 3)),
        checkpoint_precision=str(raw.get("checkpoint_precision", "bfloat16")),
        hyperparams=dict(raw.get("hyperparams", {})),
    )


# ---------------------------------------------------------------------------
# Exemplar resolution — clip_id → mp4 path + caption
# ---------------------------------------------------------------------------


def resolve_exemplars(
    history: HistoryStore, *, set_id: str
) -> list[dict[str, Any]]:
    """Return a list of {clip_id, mp4_path, caption} for the set.

    Each row's ``result_uri`` is resolved via the upload store; missing
    files are dropped with a warning so a few corrupted entries don't
    take down a 25-clip build. The caller decides whether the surviving
    count is above the build threshold.
    """
    rows = history._conn.execute(
        """SELECT g.id AS clip_id, g.result_uri, g.prompt
           FROM exemplar_set_members m
           JOIN generations g ON g.id = m.clip_id
           WHERE m.set_id = ?
             AND g.result_uri IS NOT NULL
             AND g.status = 'completed'
           ORDER BY m.added_at""",
        (set_id,),
    ).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        result_uri = row["result_uri"]
        if not result_uri or not result_uri.startswith("storage://"):
            logger.warning("clip %s has non-storage result_uri %r — skipped", row["clip_id"], result_uri)
            continue
        upload_id = result_uri[len("storage://"):]
        mp4_path = config.UPLOAD_DIR / upload_id
        if not mp4_path.exists():
            logger.warning("clip %s mp4 missing on disk at %s — skipped", row["clip_id"], mp4_path)
            continue
        out.append({
            "clip_id": row["clip_id"],
            "mp4_path": str(mp4_path),
            "caption": row["prompt"] or "",
        })
    return out


# ---------------------------------------------------------------------------
# Dataset materialization
# ---------------------------------------------------------------------------


def materialize_dataset(
    exemplars: list[dict[str, Any]], *, run_id: str, root: Path | None = None
) -> Path:
    """Stage exemplar clips + captions into an ltx-trainer-shaped dir.

    Layout (matches process_videos.py / process_dataset.py expectations):

        <root>/<run_id>/dataset.jsonl   # captions list
        <root>/<run_id>/videos/         # mp4 hardlinks (no copy)

    Returns the dataset root directory. The hardlinks avoid copying
    multi-GB exemplar sets while still giving ltx-trainer a clean
    self-contained tree it can iterate.
    """
    if root is None:
        root = TRAINING_RUNS_DIR
    run_dir = root / run_id
    videos_dir = run_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = run_dir / "dataset.jsonl"
    with jsonl_path.open("w") as f:
        for ex in exemplars:
            src = Path(ex["mp4_path"])
            dst = videos_dir / f"{ex['clip_id']}.mp4"
            if not dst.exists():
                try:
                    os.link(src, dst)
                except OSError:
                    # Cross-filesystem link or permission issue — fall
                    # back to a copy. Rare; uploads/ + training_runs/
                    # are normally on the same nvme.
                    shutil.copy2(src, dst)
            f.write(json.dumps({
                "video": str(dst),
                "caption": ex["caption"],
            }) + "\n")
    return run_dir


# ---------------------------------------------------------------------------
# YAML rendering — convert BuildConfig → ltx-trainer's expected schema.
# ---------------------------------------------------------------------------


def render_trainer_yaml(
    cfg: BuildConfig, *, dataset_root: Path, output_dir: Path
) -> Path:
    """Render an ltx-trainer YAML next to the dataset and return its path.

    Mirrors the structure of
    ``/mnt/nvme-1/repos/LTX-2/packages/ltx-trainer/configs/ltx2_av_lora.yaml``
    so ltx-trainer's ``LtxTrainerConfig`` validator accepts it
    one-to-one.
    """
    import yaml
    payload: dict[str, Any] = {
        "model": {
            "model_path": cfg.base_model_path,
            "text_encoder_path": cfg.text_encoder_path,
            "training_mode": "lora",
            "load_checkpoint": None,
        },
        "lora": {
            "rank": cfg.rank,
            "alpha": cfg.alpha,
            "dropout": cfg.dropout,
            "target_modules": list(cfg.target_modules),
        },
        "training_strategy": {
            "name": "text_to_video",
            "first_frame_conditioning_p": cfg.first_frame_conditioning_p,
            "with_audio": cfg.with_audio,
            "audio_latents_dir": "audio_latents",
        },
        "optimization": {
            "learning_rate": cfg.learning_rate,
            "steps": cfg.steps,
            "batch_size": cfg.batch_size,
            "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
            "max_grad_norm": cfg.max_grad_norm,
            "optimizer_type": cfg.optimizer_type,
            "scheduler_type": cfg.scheduler_type,
            "scheduler_params": {},
            "enable_gradient_checkpointing": cfg.enable_gradient_checkpointing,
        },
        "acceleration": {
            "mixed_precision_mode": cfg.mixed_precision_mode,
            "quantization": None,
            "load_text_encoder_in_8bit": False,
        },
        "data": {
            "preprocessed_data_root": str(dataset_root),
            "num_dataloader_workers": 2,
        },
        "validation": {
            "prompts": list(cfg.validation_prompts),
            "negative_prompt": cfg.validation_negative_prompt,
            "images": None,
            "video_dims": list(cfg.validation_video_dims),
            "frame_rate": cfg.validation_frame_rate,
            "seed": cfg.validation_seed,
            "inference_steps": cfg.validation_inference_steps,
            "interval": cfg.validation_interval,
            "videos_per_prompt": 1,
            "guidance_scale": 4.0,
            "stg_scale": 1.0,
            "stg_blocks": [29],
            "stg_mode": "stg_av",
            "generate_audio": cfg.with_audio,
            "skip_initial_validation": True,
        },
        "checkpoints": {
            "interval": cfg.checkpoint_interval,
            "keep_last_n": cfg.checkpoint_keep_last_n,
            "precision": cfg.checkpoint_precision,
        },
        "flow_matching": {
            "timestep_sampling_mode": "shifted_logit_normal",
            "timestep_sampling_params": {},
        },
        "hub": {"push_to_hub": False, "hub_model_id": None},
        "wandb": {
            "enabled": False, "project": "exemplar-lora",
            "entity": None, "tags": ["exemplar", cfg.set_id],
            "log_validation_videos": False,
        },
        "seed": cfg.seed,
        "output_dir": str(output_dir),
    }
    yaml_path = output_dir / "trainer_config.yaml"
    output_dir.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return yaml_path


# ---------------------------------------------------------------------------
# ETA estimation
# ---------------------------------------------------------------------------


def estimate_eta_hours(*, num_clips: int, rank: int, steps: int) -> float:
    """Rough wall-clock estimate for an exemplar build.

    Empirically a rank-32 LoRA over 20-30 clips at 1500 steps lands in
    ~10-15 hours on a single Blackwell. We linearize off the dominant
    factor (steps) with a small clip-count fudge for dataloader I/O.
    """
    base_hours_per_1k_steps = 7.0  # ~7h per 1000 steps at rank 32
    rank_factor = max(1.0, rank / 32.0)
    clip_factor = 1.0 + max(0, num_clips - 30) * 0.01
    return round((steps / 1000.0) * base_hours_per_1k_steps * rank_factor * clip_factor, 1)


# ---------------------------------------------------------------------------
# Reproducibility helpers
# ---------------------------------------------------------------------------


def get_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


# ---------------------------------------------------------------------------
# DB helpers — training_runs + exemplar_sets state.
# ---------------------------------------------------------------------------


def insert_training_run(
    history: HistoryStore,
    *,
    cfg: BuildConfig,
    num_clips: int,
    dataset_snapshot_path: Path,
    code_sha: str,
    validator_version: str,
    status: str = "running",
) -> None:
    """Insert a training_runs row for this build.

    Persists the full reproducibility metadata captured by the rc1 v4
    schema (training_seed / hyperparams_json / dataset_snapshot_path /
    code_sha / validator_version_at_train) plus the v7 ``status``
    column for the cancellation state machine.
    """
    history._conn.execute(
        """INSERT OR REPLACE INTO training_runs
           (run_id, base_model, base_model_sha, lora_output_path,
            num_pairs, val_loss, eval_metrics_json, trained_at,
            training_seed, hyperparams_json, dataset_snapshot_path,
            code_sha, validator_version_at_train, status)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            cfg.run_id, "ltx-2-3", "unknown",
            str(LORAS_DIR / f"{cfg.run_id}.safetensors"),
            num_clips, None, None, time.time(),
            cfg.seed, json.dumps(cfg.hyperparams),
            str(dataset_snapshot_path), code_sha, validator_version,
            status,
        ),
    )
    history._conn.commit()


def update_run_status(
    history: HistoryStore, *, run_id: str, status: str
) -> None:
    history._conn.execute(
        "UPDATE training_runs SET status = ? WHERE run_id = ?",
        (status, run_id),
    )
    history._conn.commit()


def link_set_to_lora(
    history: HistoryStore, *, set_id: str, lora_id: str
) -> None:
    history._conn.execute(
        """UPDATE exemplar_sets
           SET last_built_lora_id = ?, last_built_at = ?, updated_at = ?
           WHERE set_id = ?""",
        (lora_id, time.time(), time.time(), set_id),
    )
    history._conn.commit()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_build(
    *,
    set_id: str,
    cfg: BuildConfig,
    dry_run: bool,
    db_path: Path | None = None,
    history: HistoryStore | None = None,
    min_members: int = 20,
) -> dict[str, Any]:
    """Run the build (or report what would run in dry-run).

    Returns a summary dict for caller telemetry. The kick-off endpoint
    serializes this dict back to the operator. ``history`` may be
    passed in by the server endpoint to share its existing
    ``server_mod.history`` connection (avoids opening a second
    connection against the same DB and keeps tests using
    ``monkeypatch.setattr(server_mod, "history", ...)`` honest).
    """
    if history is None:
        history = HistoryStore(db_path=db_path) if db_path else HistoryStore()
    validator_version = config.VALIDATOR_VERSION

    exemplars = resolve_exemplars(history, set_id=set_id)
    num_clips = len(exemplars)

    summary: dict[str, Any] = {
        "set_id": set_id,
        "training_run_id": cfg.run_id,
        "validator_version_at_train": validator_version,
        "num_exemplar_clips": num_clips,
        "expected_eta_hours": estimate_eta_hours(
            num_clips=num_clips, rank=cfg.rank, steps=cfg.steps
        ),
        "lora_artifact_path_hint": str(LORAS_DIR / f"{cfg.run_id}.safetensors"),
    }

    if num_clips < min_members:
        summary["error"] = "insufficient_exemplars"
        summary["min_members_threshold"] = min_members
        logger.error(
            "set %s has %d exemplars; build threshold is %d",
            set_id, num_clips, min_members,
        )
        return summary

    # Materialize dataset so dry-run can validate the shape (no GPU).
    run_dir = materialize_dataset(exemplars, run_id=cfg.run_id)
    summary["dataset_snapshot_path"] = str(run_dir)

    # Render YAML so dry-run callers can inspect what would be sent.
    yaml_path = render_trainer_yaml(
        cfg, dataset_root=run_dir, output_dir=run_dir
    )
    summary["trainer_config_path"] = str(yaml_path)

    if dry_run:
        summary["dry_run"] = True
        logger.info(
            "DRY RUN: would train set=%s with %d clips, rank=%d, steps=%d, eta~%.1fh",
            set_id, num_clips, cfg.rank, cfg.steps, summary["expected_eta_hours"],
        )
        return summary

    # ----- REAL TRAINING PATH ----------------------------------------------
    code_sha = get_git_sha()
    insert_training_run(
        history, cfg=cfg, num_clips=num_clips,
        dataset_snapshot_path=run_dir, code_sha=code_sha,
        validator_version=validator_version, status="running",
    )

    # Cancellation handler: SIGTERM from POST .../cancel flips status
    # and cleans the dataset dir. The trainer subprocess gets
    # terminated implicitly when this process exits.
    def _on_sigterm(signum: int, frame: Any) -> None:  # pragma: no cover
        logger.warning(
            "received SIGTERM — flipping run %s to cancelled and cleaning %s",
            cfg.run_id, run_dir,
        )
        try:
            update_run_status(history, run_id=cfg.run_id, status="cancelled")
        except Exception:
            logger.exception("failed to flip status to cancelled")
        try:
            shutil.rmtree(run_dir, ignore_errors=True)
        except Exception:
            logger.exception("failed to clean run dir")
        sys.exit(143)
    signal.signal(signal.SIGTERM, _on_sigterm)

    if not LTX_TRAINER_TRAIN_SCRIPT.exists():
        update_run_status(history, run_id=cfg.run_id, status="failed")
        summary["error"] = "ltx_trainer_not_installed"
        summary["expected_path"] = str(LTX_TRAINER_TRAIN_SCRIPT)
        return summary

    try:
        proc = subprocess.run(
            [sys.executable, str(LTX_TRAINER_TRAIN_SCRIPT), str(yaml_path),
             "--disable-progress-bars"],
            check=False,
        )
        rc = proc.returncode
    except KeyboardInterrupt:
        update_run_status(history, run_id=cfg.run_id, status="cancelled")
        raise
    except Exception:
        logger.exception("ltx-trainer subprocess raised")
        update_run_status(history, run_id=cfg.run_id, status="failed")
        summary["error"] = "trainer_subprocess_error"
        return summary

    if rc != 0:
        update_run_status(history, run_id=cfg.run_id, status="failed")
        summary["error"] = "trainer_nonzero_exit"
        summary["trainer_returncode"] = rc
        return summary

    # Trainer's output_dir is run_dir; the resulting LoRA file we
    # surface is at LORAS_DIR/<run_id>.safetensors. ltx-trainer writes
    # the final adapter into output_dir; copy/rename into the registry
    # location so the existing flat-dir LoRA registry picks it up.
    final_lora_path = LORAS_DIR / f"{cfg.run_id}.safetensors"
    candidate = _find_final_safetensors(run_dir)
    if candidate is not None:
        LORAS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate, final_lora_path)
        try:
            from lora_registry import LoRARegistry
            reg = LoRARegistry(LORAS_DIR)
            reg.add(
                name=f"exemplar-{set_id}",
                filename=f"{cfg.run_id}.safetensors",
                data=final_lora_path.read_bytes(),
                description=f"Exemplar fine-tune from set {set_id}",
                base_model="ltx-2.3",
                strategy="exemplar_fine_tune",
            )
            link_set_to_lora(history, set_id=set_id, lora_id=cfg.run_id)
        except Exception:
            logger.warning("lora_registry registration failed", exc_info=True)
        summary["lora_path"] = str(final_lora_path)
    else:
        logger.warning(
            "trainer succeeded but no .safetensors found under %s — operator must register manually",
            run_dir,
        )
        summary["warning"] = "no_safetensors_after_train"

    update_run_status(history, run_id=cfg.run_id, status="completed")
    summary["dry_run"] = False
    summary["status"] = "completed"
    return summary


def _find_final_safetensors(run_dir: Path) -> Path | None:
    """Return the most recent ``.safetensors`` under run_dir, if any."""
    candidates = sorted(
        run_dir.rglob("*.safetensors"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set-id", required=True)
    parser.add_argument(
        "--config", default=str(_ROOT / "configs" / "exemplar_lora.yaml"),
        help="path to YAML template",
    )
    parser.add_argument("--run-id", default=None, help="explicit run_id (else auto)")
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None,
                        help="learning rate override")
    parser.add_argument("--base-model", default=None,
                        help="override base_model_path")
    parser.add_argument(
        "--execute", action="store_true",
        help="REAL training run (~10-15 GPU-hours). Default is dry-run.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="(default) report dataset shape + ETA, no GPU, no DB writes",
    )
    parser.add_argument(
        "--min-members", type=int, default=20,
        help="minimum exemplar count required to launch (default 20)",
    )
    args = parser.parse_args()

    if args.dry_run and args.execute:
        parser.error("--dry-run and --execute are mutually exclusive")

    overrides = {
        "rank": args.rank,
        "steps": args.steps,
        "learning_rate": args.lr,
        "base_model_path": args.base_model,
    }
    cfg = load_config(
        Path(args.config),
        set_id=args.set_id,
        run_id=args.run_id,
        overrides={k: v for k, v in overrides.items() if v is not None},
    )

    summary = run_build(
        set_id=args.set_id,
        cfg=cfg,
        dry_run=not args.execute,
        min_members=args.min_members,
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
