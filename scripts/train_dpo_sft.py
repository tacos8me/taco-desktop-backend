#!/usr/bin/env python3
"""v1.18.0-rc3 — Phase C SFT-on-chosen LoRA training.

User-locked decision: SFT-on-chosen for v1, Diffusion-DPO deferred to
Phase C.1. This script is INFRASTRUCTURE — the first real training run
waits until the corpus crosses ~1000 pairs (~6-8 weeks from now). Tests
+ runbook ship today; first invocation is operator-driven.

Defense-in-depth safety: the script defaults to ``dry_run=True`` and
requires an explicit ``--execute`` flag to actually consume GPU. ~50-60
GPU-hours per run is a real cost; require explicit confirmation.

Pipeline:

  1. Read ``configs/sft_quality_lora.yaml``
  2. SELECT ``chosen_clip_id`` from ``preference_pairs`` where
     ``signal_strength >= cfg.min_signal_strength`` AND
     ``validator_version = cfg.validator_version`` AND
     ``used_in_training_run_id IS NULL``. Group by chosen so each
     unique winner is one training example.
  3. Snapshot the dataset (frozen JSONL of chosen_ids) under
     ``training_runs/<run_id>/dataset.jsonl`` for reproducibility.
  4. Held-out 10% for eval; train on remaining 90%.
  5. Load ltx-2.3 base via ``LtxV23Loader`` (ltx-trainer dependency);
     wrap with ``LoraConfig`` (rank/alpha from yaml, target modules:
     q_proj/k_proj/v_proj/out_proj).
  6. Train with ``LtxvSFTTrainer`` — bf16, gradient_checkpointing,
     paged_adamw_32bit, LoRA-only (no base weight gradients).
     Checkpoints every epoch (resume-safe).
  7. Save LoRA artifact, persist ``training_runs`` row with FULL
     reproducibility metadata (seed/hyperparams/snapshot/code_sha/
     validator_version_at_train).
  8. UPDATE preference_pairs SET used_in_training_run_id = run_id
     for the consumed chosen_ids.
  9. Register in lora_registry as candidate (NOT auto-deployed —
     A/B harness handles promotion).

Usage::

    # Dry-run (default): no GPU, no DB writes, prints expected count.
    uv run python scripts/train_dpo_sft.py --config configs/sft_quality_lora.yaml

    # Execute (REAL training run, ~50-60 GPU-hours):
    uv run python scripts/train_dpo_sft.py --config configs/sft_quality_lora.yaml --execute

    # Resume from checkpoint:
    uv run python scripts/train_dpo_sft.py --config configs/sft_quality_lora.yaml \\
        --execute --resume-from training_runs/sft-quality-v0.0.1/checkpoint-2/
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
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
logger = logging.getLogger("train_dpo_sft")


TRAINING_RUNS_DIR = _ROOT / "training_runs"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    run_id: str
    base_model_path: str
    validator_version: str | None
    min_signal_strength: float
    rank: int
    alpha: int
    target_modules: list[str]
    epochs: int
    learning_rate: float
    seed: int
    gradient_accumulation_steps: int
    per_device_train_batch_size: int
    hyperparams: dict[str, Any]


def load_config(path: Path) -> TrainConfig:
    import yaml
    raw = yaml.safe_load(path.read_text())
    return TrainConfig(
        run_id=raw["run_id"],
        base_model_path=raw["base_model_path"],
        validator_version=raw.get("validator_version"),  # null → resolve at runtime
        min_signal_strength=float(raw.get("min_signal_strength", 0.5)),
        rank=int(raw["rank"]),
        alpha=int(raw["alpha"]),
        target_modules=list(raw["target_modules"]),
        epochs=int(raw["epochs"]),
        learning_rate=float(raw["learning_rate"]),
        seed=int(raw.get("seed", 42)),
        gradient_accumulation_steps=int(raw.get("gradient_accumulation_steps", 4)),
        per_device_train_batch_size=int(raw.get("per_device_train_batch_size", 1)),
        hyperparams=dict(raw.get("hyperparams", {})),
    )


# ---------------------------------------------------------------------------
# Dataset selection
# ---------------------------------------------------------------------------


def select_chosen_ids(
    history: HistoryStore,
    *,
    validator_version: str,
    min_signal_strength: float,
) -> list[str]:
    """Return de-duplicated chosen_clip_ids meeting the gate.

    Excludes pairs already consumed by a prior run
    (``used_in_training_run_id IS NOT NULL``).
    """
    rows = history._conn.execute(
        """SELECT DISTINCT chosen_clip_id
           FROM preference_pairs
           WHERE signal_strength >= ?
             AND validator_version = ?
             AND used_in_training_run_id IS NULL
             AND chosen_clip_id IS NOT NULL""",
        (min_signal_strength, validator_version),
    ).fetchall()
    return [r["chosen_clip_id"] for r in rows]


def split_held_out(
    chosen_ids: list[str], *, ratio: float, seed: int
) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    shuffled = list(chosen_ids)
    rng.shuffle(shuffled)
    cut = int(len(shuffled) * ratio)
    return shuffled[:cut], shuffled[cut:]


def save_dataset_snapshot(chosen_ids: list[str], run_id: str) -> Path:
    run_dir = TRAINING_RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "dataset.jsonl"
    with path.open("w") as f:
        for cid in chosen_ids:
            f.write(json.dumps({"chosen_clip_id": cid}) + "\n")
    return path


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


def get_base_sha(base_path: str) -> str:
    p = Path(base_path)
    if not p.exists():
        return "unknown"
    h = hashlib.sha256()
    with p.open("rb") as f:
        # First 64MiB is enough to disambiguate distilled vs dev vs hq;
        # full-file hash on a 22B safetensors is ~30s I/O bound.
        h.update(f.read(64 * 1024 * 1024))
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Training-run persistence
# ---------------------------------------------------------------------------


def persist_training_run(
    history: HistoryStore,
    *,
    cfg: TrainConfig,
    base_sha: str,
    lora_path: Path,
    num_pairs: int,
    train_loss: float,
    eval_loss: float,
    eval_metrics: dict,
    dataset_snapshot_path: Path,
    code_sha: str,
    validator_version: str,
) -> None:
    history._conn.execute(
        """INSERT INTO training_runs
           (run_id, base_model, base_model_sha, lora_output_path,
            num_pairs, val_loss, eval_metrics_json, trained_at,
            training_seed, hyperparams_json, dataset_snapshot_path,
            code_sha, validator_version_at_train)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            cfg.run_id, "ltx-2-3-22b", base_sha, str(lora_path),
            num_pairs, eval_loss, json.dumps({**eval_metrics, "train_loss": train_loss}),
            time.time(),
            cfg.seed, json.dumps(cfg.hyperparams), str(dataset_snapshot_path),
            code_sha, validator_version,
        ),
    )
    history._conn.commit()


def mark_pairs_consumed(
    history: HistoryStore, *, run_id: str, chosen_ids: list[str]
) -> None:
    if not chosen_ids:
        return
    placeholders = ",".join("?" for _ in chosen_ids)
    history._conn.execute(
        f"""UPDATE preference_pairs SET used_in_training_run_id = ?
            WHERE chosen_clip_id IN ({placeholders})
              AND used_in_training_run_id IS NULL""",
        (run_id, *chosen_ids),
    )
    history._conn.commit()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_training(
    *,
    cfg: TrainConfig,
    dry_run: bool,
    resume_from: Path | None = None,
    db_path: Path | None = None,
) -> dict[str, Any]:
    """Run training (or report what would run in dry-run mode).

    Returns a summary dict for caller telemetry.
    """
    history = HistoryStore(db_path=db_path) if db_path else HistoryStore()
    validator_version = cfg.validator_version or config.VALIDATOR_VERSION

    chosen_ids = select_chosen_ids(
        history,
        validator_version=validator_version,
        min_signal_strength=cfg.min_signal_strength,
    )
    summary = {
        "run_id": cfg.run_id,
        "validator_version": validator_version,
        "num_chosen_ids": len(chosen_ids),
        "min_signal_strength": cfg.min_signal_strength,
    }

    if dry_run:
        train_ids, eval_ids = split_held_out(chosen_ids, ratio=0.9, seed=cfg.seed)
        summary["dry_run"] = True
        summary["would_train_on"] = len(train_ids)
        summary["would_eval_on"] = len(eval_ids)
        logger.info(
            "DRY RUN: would train on %d chosen clips (held-out: %d) — no GPU consumption, no DB writes",
            len(train_ids), len(eval_ids),
        )
        return summary

    if not chosen_ids:
        logger.error(
            "no eligible chosen_clip_ids — corpus not yet sufficient. min_signal_strength=%.2f, validator_version=%s",
            cfg.min_signal_strength, validator_version,
        )
        summary["error"] = "no_eligible_pairs"
        return summary

    # ---- REAL TRAINING PATH ------------------------------------------------
    # Heavy imports gated behind --execute so dry-run + tests don't pull
    # bitsandbytes / peft / ltx-trainer into the import graph.
    import torch  # noqa: F401
    from peft import LoraConfig, get_peft_model
    from transformers import TrainingArguments

    try:
        from ltx_trainer import LtxV23Loader, LtxvSFTTrainer  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "ltx-trainer not installed; install from /mnt/nvme-1/repos/LTX-2/packages/ltx-trainer "
            "before --execute"
        ) from exc

    train_ids, eval_ids = split_held_out(chosen_ids, ratio=0.9, seed=cfg.seed)
    dataset_snapshot_path = save_dataset_snapshot(chosen_ids, cfg.run_id)

    base_sha = get_base_sha(cfg.base_model_path)
    code_sha = get_git_sha()

    logger.info(
        "loading base model from %s (sha16=%s)",
        cfg.base_model_path, base_sha,
    )
    base = LtxV23Loader.load(cfg.base_model_path)
    lora_cfg = LoraConfig(
        r=cfg.rank, lora_alpha=cfg.alpha, target_modules=cfg.target_modules,
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_cfg)

    output_dir = TRAINING_RUNS_DIR / cfg.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(output_dir),
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        seed=cfg.seed,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        logging_steps=10,
    )
    trainer = LtxvSFTTrainer(
        model=model,
        train_dataset=train_ids,
        eval_dataset=eval_ids,
        args=args,
    )

    if resume_from:
        result = trainer.train(resume_from_checkpoint=str(resume_from))
    else:
        result = trainer.train()
    eval_metrics = trainer.evaluate()
    eval_loss = eval_metrics.get("eval_loss", float("nan"))

    # Save LoRA artifact
    lora_path = _ROOT / "loras" / f"{cfg.run_id}.safetensors"
    lora_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(lora_path.parent / cfg.run_id)

    persist_training_run(
        history,
        cfg=cfg, base_sha=base_sha, lora_path=lora_path,
        num_pairs=len(chosen_ids),
        train_loss=float(getattr(result, "training_loss", 0.0)),
        eval_loss=float(eval_loss),
        eval_metrics=eval_metrics,
        dataset_snapshot_path=dataset_snapshot_path,
        code_sha=code_sha, validator_version=validator_version,
    )
    mark_pairs_consumed(history, run_id=cfg.run_id, chosen_ids=chosen_ids)

    # Best-effort lora_registry candidate registration. The registry's
    # add() takes raw bytes; we already wrote the file, so we read it
    # back. Failure is non-fatal — operator can register manually.
    try:
        from lora_registry import LoRARegistry
        reg = LoRARegistry(_ROOT / "loras")
        reg.add(
            name=f"sft-quality-{cfg.run_id}",
            filename=f"{cfg.run_id}.safetensors",
            data=lora_path.read_bytes(),
            description=f"SFT v{cfg.run_id} candidate; A/B in progress",
            base_model="ltx-2.3",
            strategy="dpo_quality",
        )
    except Exception:
        logger.warning("lora_registry registration failed", exc_info=True)

    summary.update({
        "dry_run": False,
        "trained_on": len(train_ids),
        "evaluated_on": len(eval_ids),
        "train_loss": float(getattr(result, "training_loss", 0.0)),
        "eval_loss": float(eval_loss),
        "lora_path": str(lora_path),
    })
    logger.info("training complete: %s", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to YAML config")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="REAL training run (~50-60 GPU-hours). Default is dry-run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="(default) report dataset shape + counts, no GPU, no DB writes",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="path to checkpoint dir for resume",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    # Defense-in-depth: dry_run is the default. Only --execute flips it off.
    dry_run = not args.execute
    if args.dry_run and args.execute:
        parser.error("--dry-run and --execute are mutually exclusive")

    summary = run_training(
        cfg=cfg,
        dry_run=dry_run,
        resume_from=Path(args.resume_from) if args.resume_from else None,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
