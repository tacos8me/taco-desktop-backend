# Plan: encode_prompts() Helper Integration

## Status: Already Integrated

## Summary

After thorough investigation, **`split_model_manager.py` already uses the upstream `encode_prompts()` helper**. No integration work is needed. This document records the findings for reference.

## Current State

`split_model_manager.py` imports `encode_prompts` directly from the upstream helpers:

```python
from ltx_pipelines.utils.helpers import (
    ...
    encode_prompts,
    ...
)
```

Every generation method already calls it:

| Method | Call | Line |
|--------|------|------|
| `_run_t2v` (fast) | `encode_prompts([prompt], self._encoder_ledger)` | ~323 |
| `_run_t2v` (pro) | `encode_prompts([prompt, DEFAULT_NEGATIVE_PROMPT], self._encoder_ledger)` | ~328 |
| `_run_i2v` (fast) | `encode_prompts([prompt], self._encoder_ledger)` | ~433 |
| `_run_i2v` (pro) | `encode_prompts([prompt, DEFAULT_NEGATIVE_PROMPT], self._encoder_ledger)` | ~438 |
| `_run_a2v` | `encode_prompts([prompt, DEFAULT_NEGATIVE_PROMPT], self._encoder_ledger)` | ~538 |
| `_run_retake` | `encode_prompts([prompt, DEFAULT_NEGATIVE_PROMPT], self._encoder_ledger)` | ~644 |

## How encode_prompts() Works

Defined in `/mnt/nvme-1/repos/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/utils/helpers.py:48-86`:

```python
def encode_prompts(
    prompts: list[str],
    model_ledger: object,
    *,
    enhance_prompt_image: str | None = None,
    enhance_prompt_seed: int = 42,
    enhance_first_prompt: bool = False,
) -> list[EmbeddingsProcessorOutput]:
```

Steps:
1. Loads text encoder via `model_ledger.text_encoder()`
2. Optionally enhances `prompts[0]` via Gemma (t2v or i2v enhancement)
3. Encodes all prompts: `text_encoder.encode(p)` for each prompt
4. Frees text encoder + calls `cleanup_memory()`
5. Loads embeddings processor via `model_ledger.gemma_embeddings_processor()`
6. Processes each raw output: `embeddings_processor.process_hidden_states(hs, mask)`
7. Frees embeddings processor + calls `cleanup_memory()`
8. Returns `list[EmbeddingsProcessorOutput]`

### EmbeddingsProcessorOutput

A `NamedTuple` with three fields:
- `video_encoding: torch.Tensor` -- context vector for video denoising
- `audio_encoding: torch.Tensor | None` -- context vector for audio denoising
- `attention_mask: torch.Tensor` -- attention mask for transformer

### CachingModelLedger Compatibility

The `CachingModelLedger` in `split_model_manager.py` is fully compatible. It implements both methods that `encode_prompts` calls:
- `text_encoder()` -- returns `self._cache["text_encoder"]`
- `gemma_embeddings_processor()` -- returns `self._cache["embeddings_processor"]`

**Important difference from upstream `ModelLedger`:** The caching variant returns persistent references. When `encode_prompts()` does `del text_encoder` and `cleanup_memory()`, it only drops the local reference -- the model stays alive in the cache. This is intentional: we pre-load encoders at startup and keep them resident.

## Cross-Device Transfer

After `encode_prompts()` returns, the embeddings live on GPU:0 (the encoder device). If the acquired worker is on a different GPU, `_contexts_to_device()` moves them:

```python
ctx_p, ctx_n = encode_prompts([prompt, DEFAULT_NEGATIVE_PROMPT], self._encoder_ledger)
ctx_p, ctx_n = self._contexts_to_device([ctx_p, ctx_n], device)
```

## Features Not Yet Used

The upstream `encode_prompts()` supports optional prompt enhancement, but taco-backend does not pass these kwargs:
- `enhance_first_prompt=False` (default, never overridden)
- `enhance_prompt_image=None` (default, never overridden)
- `enhance_prompt_seed=42` (default, never overridden)

This is a deliberate choice -- prompt enhancement is a client-side concern in taco-desktop.

## No Changes Required

The integration is complete. All four generation methods use `encode_prompts()` with the `CachingModelLedger`, matching the upstream pattern used by `TI2VidTwoStagesPipeline`, `DistilledPipeline`, `RetakePipeline`, etc.
