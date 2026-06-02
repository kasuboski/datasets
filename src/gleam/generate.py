"""Production SFT instruction pair generator for Gleam.

Generates ~14K instruction-response pairs from the filtered Gleam corpus
using NeuralWatt Qwen3.6-35B Fast (primary), Cerebras GPT OSS 120B (fast),
or Z.AI GLM-5.1 (fallback).

Features:
  - Crash-resilient: appends JSONL after each request, state file after each file
  - Resumable: reads state file on startup, skips already-processed files
  - Cost guard: stops when account balance drops below threshold (NeuralWatt)
  - Cost estimation for Cerebras ($0.35/M input, $0.75/M output)
  - Structured logging to file + stdout
  - 3 task types per file (code_gen, explanation, completion) → ~14K pairs from ~4.7K files

Usage:
  NEURALWATT_API_KEY=sk-xxx python -m datasets.gleam.generate \
    --provider neuralwatt \
    --output data/sft/generation.jsonl \
    [--max-files 4700] \
    [--balance-limit 0.50]

  # Cerebras (fast: ~2h for full run, ~3K tok/s)
  CEREBRAS_API_KEY=csk-xxx python -m datasets.gleam.generate \
    --provider cerebras --max-files 4700

Resume after crash:
  Just re-run the same command — it reads generation_state.json to find progress.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Import from our local datasets/ package (not the HF datasets pip package)
# Use direct file imports to avoid namespace collision
from src.gleam.pilot.prompts import (  # noqa: E402
    SYSTEM_PROMPT,
    CODE_GEN_INSTRUCTIONS,
    PromptSpec,
)
from src.gleam.pilot.sample import SampledFile  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROVIDERS = {
    "neuralwatt": {
        "base_url": "https://api.neuralwatt.com/v1",
        "model": "qwen3.6-35b-fast",
        "api_key_env": "NEURALWATT_API_KEY",
        "gap_seconds": 0.5,
        "extra_body": {},
        "max_tokens": 4096,
        # Pricing: $0.056/M input, $0.312/M output (imputed from production data)
        # Balance guard: reads allowance_remaining_usd from response
    },
    "cerebras": {
        "base_url": "https://api.cerebras.ai/v1",
        "model": "gpt-oss-120b",
        "api_key_env": "CEREBRAS_API_KEY",
        "gap_seconds": 0.1,
        "extra_body": {"reasoning_effort": "low"},
        "max_tokens": 4096,
        # Pricing: $0.35/M input, $0.75/M output
        # Rate limits (Developer): 1K RPM / 1M input tok/min
        # Speed: ~3000 tok/s output, ~50K tok/s prefill
    },
    "zai": {
        "base_url": "https://api.z.ai/api/coding/paas/v4",
        "model": "glm-5.1",
        "api_key_env": "ZAI_API_KEY",
        "gap_seconds": 3.0,
        "extra_body": {"thinking": {"type": "disabled"}},
        "max_tokens": 4096,
    },
}

TASK_TYPES = ["code_gen", "explanation", "completion"]
# Each file gets all 3 task types → 3 pairs per file
# Matches the pilot: 10 files × 3 types = 30 requests per provider
# with remaining budget going to more code_gen/completion variants

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_path: Path) -> logging.Logger:
    """Set up dual logging: file (DEBUG) + stdout (INFO)."""
    logger = logging.getLogger("generate")
    logger.setLevel(logging.DEBUG)

    # File handler — detailed
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(fh)

    # Stdout handler — progress only
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(sh)

    return logger


# ---------------------------------------------------------------------------
# State management (crash resumability)
# ---------------------------------------------------------------------------

class GenerationState:
    """Tracks which files have been processed. Persists to JSON after each file."""

    def __init__(self, state_path: Path, logger: logging.Logger):
        self.state_path = state_path
        self.log = logger
        self.completed_file_ids: set[str] = set()
        self.total_pairs: int = 0
        self.total_errors: int = 0
        self.total_cost_usd: float = 0.0
        self.total_energy_j: float = 0.0
        self.start_time: float = 0.0
        self.last_balance_usd: float | None = None

        # Load existing state if resuming
        if state_path.exists():
            self._load()

    def _load(self) -> None:
        try:
            with open(self.state_path) as f:
                data = json.load(f)
            self.completed_file_ids = set(data.get("completed_file_ids", []))
            self.total_pairs = data.get("total_pairs", 0)
            self.total_errors = data.get("total_errors", 0)
            self.total_cost_usd = data.get("total_cost_usd", 0.0)
            self.total_energy_j = data.get("total_energy_j", 0.0)
            self.start_time = data.get("start_time", 0.0)
            self.last_balance_usd = data.get("last_balance_usd")
            self.log.info(f"Resumed state: {len(self.completed_file_ids)} files already processed, "
                         f"{self.total_pairs} pairs, ${self.total_cost_usd:.4f} spent")
        except Exception as e:
            self.log.warning(f"Could not load state file: {e}. Starting fresh.")

    def save(self) -> None:
        data = {
            "completed_file_ids": sorted(self.completed_file_ids),
            "total_pairs": self.total_pairs,
            "total_errors": self.total_errors,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_energy_j": round(self.total_energy_j, 2),
            "start_time": self.start_time,
            "last_balance_usd": self.last_balance_usd,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        tmp = self.state_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        tmp.rename(self.state_path)  # atomic on POSIX

    def mark_completed(self, file_id: str) -> None:
        self.completed_file_ids.add(file_id)
        self.save()

    def is_completed(self, file_id: str) -> bool:
        return file_id in self.completed_file_ids


# ---------------------------------------------------------------------------
# Prompt building (single task per file)
# ---------------------------------------------------------------------------

def build_prompt_for_file(
    file_id: str,
    code: str,
    file_path: str,
    repo_name: str,
    task_type: str,
) -> PromptSpec:
    """Build a single prompt for one file with a specific task type."""
    source_label = f"repo {repo_name}"

    if task_type == "code_gen":
        instruction = random.choice(CODE_GEN_INSTRUCTIONS)
        user_msg = f"""Based on the following Gleam code, write a new Gleam module that solves a \
related but different problem. Use idiomatic Gleam: type annotations, \
pattern matching, Result for errors, pipe operators where appropriate.

Reference code (from {source_label}):
```gleam
{code}
```

Write a Gleam module that: {instruction}"""

    elif task_type == "explanation":
        user_msg = f"""Explain the following Gleam code in detail. Describe what each function does, \
how types are used, and any notable patterns or idioms you notice.

```gleam
{code}
```

Provide a thorough explanation suitable for someone learning Gleam."""

    elif task_type == "completion":
        lines = code.split("\n")
        cutoff = max(5, int(len(lines) * 0.6))
        partial = "\n".join(lines[:cutoff])
        partial += "\n// TODO: implement the remaining functions"
        user_msg = f"""Complete the following partial Gleam module. The code has been truncated — \
finish implementing the remaining functions to make a complete, working module.

```gleam
{partial}
```

Complete the implementation. Include type annotations and follow Gleam conventions."""
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    return PromptSpec(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        task_type=task_type,
        approach="single",
        source_file_id=file_id,
    )


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def call_api(
    client,
    provider: str,
    messages: list[dict],
    cfg: dict,
) -> dict[str, Any]:
    """Make a single API call. Returns result dict."""
    t0 = time.time()
    try:
        kwargs = dict(
            model=cfg["model"],
            messages=messages,
            max_tokens=cfg["max_tokens"],
            temperature=0.7,
        )
        if cfg.get("extra_body"):
            kwargs["extra_body"] = cfg["extra_body"]

        response = client.chat.completions.create(**kwargs)
        wall_time = time.time() - t0

        choice = response.choices[0]
        result = {
            "content": choice.message.content,
            "finish_reason": choice.finish_reason,
            "wall_time_ms": round(wall_time * 1000),
            "input_tokens": 0,
            "output_tokens": 0,
        }

        if hasattr(response, "usage") and response.usage:
            result["input_tokens"] = response.usage.prompt_tokens or 0
            result["output_tokens"] = response.usage.completion_tokens or 0

        # Provider-specific fields
        raw = response.model_dump()

        # NeuralWatt energy/cost
        if provider == "neuralwatt":
            if "energy" in raw and isinstance(raw["energy"], dict):
                result["nw_energy"] = raw["energy"]
            if "cost" in raw and isinstance(raw["cost"], dict):
                result["nw_cost"] = raw["cost"]

        # Cerebras: reasoning, timing, cached tokens, cost estimation
        if provider == "cerebras":
            # Capture reasoning if present (included in completion_tokens)
            if hasattr(choice.message, "reasoning") and choice.message.reasoning:
                result["reasoning"] = choice.message.reasoning
            # Timing breakdown
            if "time_info" in raw and isinstance(raw["time_info"], dict):
                result["time_info"] = raw["time_info"]
            # Cached tokens (system prompt reuse)
            cached = 0
            if hasattr(response.usage, "prompt_tokens_details") and response.usage.prompt_tokens_details:
                cached = getattr(response.usage.prompt_tokens_details, "cached_tokens", 0) or 0
            result["cached_input_tokens"] = cached
            # Cost estimation ($0.35/M input, $0.75/M output)
            billable_input = max(0, result["input_tokens"] - cached)
            result["estimated_cost_usd"] = (
                billable_input / 1e6 * 0.35 + result["output_tokens"] / 1e6 * 0.75
            )

        return result

    except Exception as e:
        wall_time = time.time() - t0
        return {
            "content": None,
            "error": str(e),
            "error_type": type(e).__name__,
            "wall_time_ms": round(wall_time * 1000),
            "input_tokens": 0,
            "output_tokens": 0,
        }


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------

def call_with_retry(
    client,
    provider: str,
    messages: list[dict],
    cfg: dict,
    logger: logging.Logger,
    max_retries: int = 3,
    base_delay: float = 5.0,
) -> dict[str, Any]:
    """Call API with exponential backoff retry."""
    for attempt in range(max_retries + 1):
        result = call_api(client, provider, messages, cfg)

        if not result.get("error"):
            return result

        if attempt < max_retries:
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            logger.warning(f"  Attempt {attempt+1}/{max_retries+1} failed: "
                          f"{result['error'][:100]}. Retrying in {delay:.1f}s...")
            time.sleep(delay)
        else:
            logger.error(f"  All {max_retries+1} attempts failed: {result['error'][:200]}")
            return result

    return result  # unreachable, but mypy


# ---------------------------------------------------------------------------
# Balance check (NeuralWatt)
# ---------------------------------------------------------------------------

def check_balance(result: dict, balance_limit: float, logger: logging.Logger) -> float | None:
    """Extract balance from NeuralWatt response. Returns None if OK, balance if below limit."""
    nw_cost = result.get("nw_cost", {})
    balance = nw_cost.get("allowance_remaining_usd")
    if balance is not None and balance < balance_limit:
        logger.warning(f"⚠ Account balance ${balance:.4f} below limit ${balance_limit:.2f}. Stopping.")
        return balance
    return None


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------

def load_corpus_files(
    corpus_path: str | None,
    max_files: int,
    seed: int,
    exclude_repos: list[str] | None = None,
    max_input_tokens: int | None = None,
    logger: logging.Logger | None = None,
) -> list[dict]:
    """Load and shuffle files from corpus.

    If corpus_path is None or 'huggingface', downloads from
    kasuboski/gleam-code-corpus (all split). Otherwise loads local parquet.

    Args:
        max_input_tokens: If set, drop files whose source code would likely
            exceed this many input tokens. Estimated from character count
            using a ~3 chars/token heuristic (empirically calibrated for
            Gleam code on Qwen tokenizers). Only ~1% of files near the
            threshold will be misclassified. Set to 0 or negative to disable.
    """
    import polars as pl

    use_hf = corpus_path is None or corpus_path == "huggingface"

    if use_hf:
        from huggingface_hub import hf_hub_download
        if logger:
            logger.info("Loading corpus from HuggingFace (kasuboski/gleam-code-corpus)...")
        path = hf_hub_download(
            repo_id="kasuboski/gleam-code-corpus",
            filename="all/data.parquet",
            repo_type="dataset",
        )
        df = pl.read_parquet(path)
    else:
        if logger:
            logger.info(f"Loading corpus from {corpus_path}...")
        df = pl.read_parquet(corpus_path)

    # Exclude specified repos
    if exclude_repos:
        before = len(df)
        df = df.filter(~pl.col("repo_name").is_in(exclude_repos))
        if logger:
            logger.info(f"Excluded {before - len(df)} files from {exclude_repos}")

    # Filter out oversized files
    # Large files (>3K input tokens) produce worse output (10.9% truncation,
    # 13.7% no valid Gleam) at 2.3× the cost. We estimate tokens from
    # character count using the median ratio measured from production data.
    #
    # The template overhead is measured by building a dummy prompt with empty
    # code, so it automatically adjusts if the system prompt or templates change
    # (e.g. adding Gleam docs to the system prompt).
    if max_input_tokens and max_input_tokens > 0:
        _CHARS_PER_TOKEN = 2.96  # median ratio from 2,158 production files
        # Measure actual template overhead (code_gen is worst case — full source)
        _dummy_spec = build_prompt_for_file(
            file_id="", code="", file_path="", repo_name="", task_type="code_gen",
        )
        _template_chars = sum(len(m["content"]) for m in _dummy_spec.messages)
        _template_tokens = _template_chars / _CHARS_PER_TOKEN
        max_code_chars = int((max_input_tokens - _template_tokens) * _CHARS_PER_TOKEN)
        before = len(df)
        df = df.filter(pl.col("code").str.len_chars() <= max_code_chars)
        if logger:
            logger.info(
                f"Filtered {before - len(df)} oversized files "
                f"(code > {max_code_chars:,} chars ≈ {max_input_tokens:,} input tokens, "
                f"template overhead {_template_tokens:.0f} tokens), "
                f"{len(df)} remaining"
            )

    # Sample if needed
    total_before = len(df)
    if total_before > max_files:
        df = df.sample(n=max_files, seed=seed)
        if logger:
            logger.info(f"Sampled {max_files} files from {total_before} total (seed={seed})")

    # Shuffle for diversity
    df = df.sample(fraction=1.0, seed=seed + 1)

    files = df.select(["id", "code", "file_path", "repo_name", "repo_url", "stars"]).to_dicts()
    if logger:
        logger.info(f"Loaded {len(files)} files for generation")

    return files

# ---------------------------------------------------------------------------
# Concurrent file processing
# ---------------------------------------------------------------------------

def _process_file(
    file_data: dict,
    client,
    provider: str,
    cfg: dict,
    state: GenerationState,
    output_path: Path,
    logger: logging.Logger,
    lock: threading.Lock,
    stop_event: threading.Event,
    balance_limit: float,
    max_retries: int,
) -> None:
    """Process a single file: generate all 3 task types, write results."""
    file_id = file_data["id"]

    for task_type in TASK_TYPES:
        if stop_event.is_set():
            return

        spec = build_prompt_for_file(
            file_id=file_id,
            code=file_data["code"],
            file_path=file_data["file_path"],
            repo_name=file_data["repo_name"],
            task_type=task_type,
        )

        result = call_with_retry(
            client, provider, spec.messages, cfg, logger, max_retries=max_retries,
        )

        with lock:
            record = {
                "request_id": f"{provider}_{state.total_pairs:06d}",
                "provider": provider,
                "model": cfg["model"],
                "task_type": task_type,
                "source_file_id": file_id,
                "source_file_path": file_data["file_path"],
                "source_repo": file_data["repo_name"],
                "source_stars": file_data.get("stars"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **result,
            }

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "a") as f:
                f.write(json.dumps(record) + "\n")

            if result.get("error"):
                state.total_errors += 1
                logger.info(f"  [{state.total_pairs + state.total_errors + 1:5d}] FAIL {file_id[:50]:50s} | {task_type:12s} | {result['error'][:60]}")
            else:
                state.total_pairs += 1
                wall = result["wall_time_ms"] / 1000
                out_tok = result["output_tokens"]
                tps = out_tok / wall if wall > 0 else 0

                nw_energy = result.get("nw_energy", {})
                nw_cost = result.get("nw_cost", {})
                energy_j = nw_energy.get("energy_joules", 0)
                cost_usd = nw_cost.get("request_cost_usd", 0) or result.get("estimated_cost_usd", 0)
                balance = nw_cost.get("allowance_remaining_usd")
                state.total_energy_j += energy_j
                state.total_cost_usd += cost_usd
                if balance is not None:
                    state.last_balance_usd = balance

                elapsed = time.time() - state.start_time
                pairs_per_sec = state.total_pairs / elapsed if elapsed > 0 else 0
                logger.info(
                    f"  [{state.total_pairs:5d}] OK   {file_id[:50]:50s} | {task_type:12s} "
                    f"| {wall:5.1f}s | {tps:5.0f} tok/s | {out_tok:5d} out "
                    f"| ${state.total_cost_usd:.4f} total "
                    f"| {pairs_per_sec:.1f} pairs/s"
                    + (f" | bal=${balance:.2f}" if balance is not None else "")
                )

                if balance is not None and balance < balance_limit:
                    logger.warning(f"\n⚠ Balance ${balance:.4f} below limit ${balance_limit:.2f}. Stopping.")
                    stop_event.set()
                    state.mark_completed(file_id)
                    return

        # Per-request gap (only meaningful in single-worker mode)
        time.sleep(cfg["gap_seconds"])

    with lock:
        state.mark_completed(file_id)


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def run_generation(
    provider: str,
    files: list[dict],
    output_path: Path,
    state: GenerationState,
    logger: logging.Logger,
    balance_limit: float = 0.50,
    max_retries: int = 3,
    workers: int = 1,
) -> None:
    """Main generation loop. Processes files, writes JSONL, manages state."""

    from concurrent.futures import ThreadPoolExecutor, as_completed
    from openai import OpenAI

    cfg = PROVIDERS[provider]
    client = OpenAI(
        api_key=os.environ[cfg["api_key_env"]],
        base_url=cfg["base_url"],
    )

    # Filter to only unprocessed files
    remaining = [f for f in files if not state.is_completed(f["id"])]
    logger.info(f"\n{'='*60}")
    logger.info(f"GENERATION: {provider} | {cfg['model']} | {workers} worker(s)")
    logger.info(f"  Files remaining: {len(remaining)} / {len(files)} total")
    logger.info(f"  Pairs so far:    {state.total_pairs}")
    logger.info(f"  Cost so far:     ${state.total_cost_usd:.4f}")
    logger.info(f"  Output:          {output_path}")
    logger.info(f"  Balance limit:   ${balance_limit:.2f}")
    logger.info(f"{'='*60}\n")

    if not state.start_time:
        state.start_time = time.time()

    stop_event = threading.Event()
    lock = threading.Lock()

    if workers <= 1:
        # Sequential mode (original behavior)
        consecutive_errors = 0
        max_consecutive_errors = 10

        for i, file_data in enumerate(remaining):
            if stop_event.is_set():
                break

            file_id = file_data["id"]

            if consecutive_errors >= max_consecutive_errors:
                logger.error(f"Too many consecutive errors ({consecutive_errors}). Stopping.")
                break

            for task_type in TASK_TYPES:
                spec = build_prompt_for_file(
                    file_id=file_id,
                    code=file_data["code"],
                    file_path=file_data["file_path"],
                    repo_name=file_data["repo_name"],
                    task_type=task_type,
                )

                result = call_with_retry(
                    client, provider, spec.messages, cfg, logger, max_retries=max_retries,
                )

                record = {
                    "request_id": f"{provider}_{state.total_pairs:06d}",
                    "provider": provider,
                    "model": cfg["model"],
                    "task_type": task_type,
                    "source_file_id": file_id,
                    "source_file_path": file_data["file_path"],
                    "source_repo": file_data["repo_name"],
                    "source_stars": file_data.get("stars"),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    **result,
                }

                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "a") as f:
                    f.write(json.dumps(record) + "\n")

                if result.get("error"):
                    state.total_errors += 1
                    consecutive_errors += 1
                    logger.info(f"  [{state.total_pairs + 1:5d}] FAIL {file_id[:50]:50s} | {task_type:12s} | {result['error'][:60]}")
                else:
                    consecutive_errors = 0
                    state.total_pairs += 1
                    wall = result["wall_time_ms"] / 1000
                    out_tok = result["output_tokens"]
                    tps = out_tok / wall if wall > 0 else 0

                    nw_energy = result.get("nw_energy", {})
                    nw_cost = result.get("nw_cost", {})
                    energy_j = nw_energy.get("energy_joules", 0)
                    cost_usd = nw_cost.get("request_cost_usd", 0) or result.get("estimated_cost_usd", 0)
                    balance = nw_cost.get("allowance_remaining_usd")
                    state.total_energy_j += energy_j
                    state.total_cost_usd += cost_usd
                    if balance is not None:
                        state.last_balance_usd = balance

                    elapsed = time.time() - state.start_time
                    remaining_pairs = len(remaining) * len(TASK_TYPES) - (i * len(TASK_TYPES) + TASK_TYPES.index(task_type)) - 1
                    eta = (elapsed / state.total_pairs) * remaining_pairs if state.total_pairs > 0 else 0
                    logger.info(
                        f"  [{state.total_pairs:5d}] OK   {file_id[:50]:50s} | {task_type:12s} "
                        f"| {wall:5.1f}s | {tps:5.0f} tok/s | {out_tok:5d} out "
                        f"| ${state.total_cost_usd:.4f} total "
                        f"| ETA {eta/3600:.1f}h"
                        + (f" | bal=${balance:.2f}" if balance is not None else "")
                    )

                    if balance is not None and balance < balance_limit:
                        logger.warning(f"\n⚠ Balance ${balance:.4f} below limit ${balance_limit:.2f}. Stopping.")
                        state.mark_completed(file_id)
                        return

                time.sleep(cfg["gap_seconds"])

            state.mark_completed(file_id)
    else:
        # Concurrent mode
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for file_data in remaining:
                if stop_event.is_set():
                    break
                future = executor.submit(
                    _process_file,
                    file_data=file_data,
                    client=client,
                    provider=provider,
                    cfg=cfg,
                    state=state,
                    output_path=output_path,
                    logger=logger,
                    lock=lock,
                    stop_event=stop_event,
                    balance_limit=balance_limit,
                    max_retries=max_retries,
                )
                futures[future] = file_data["id"]

            for future in as_completed(futures):
                file_id = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"  Worker error on {file_id[:50]}: {e}")

                if stop_event.is_set():
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

    # Final summary
    elapsed = time.time() - state.start_time if state.start_time else 0
    logger.info(f"\n{'='*60}")
    logger.info(f"GENERATION COMPLETE")
    logger.info(f"  Pairs generated:  {state.total_pairs}")
    logger.info(f"  Errors:           {state.total_errors}")
    logger.info(f"  Files processed:  {len(state.completed_file_ids)}")
    logger.info(f"  Elapsed:          {elapsed/3600:.1f} hours")
    logger.info(f"  Total cost:       ${state.total_cost_usd:.4f}")
    logger.info(f"  Total energy:     {state.total_energy_j:.0f} J ({state.total_energy_j/3600000:.4f} kWh)")
    if state.last_balance_usd is not None:
        logger.info(f"  Balance remain:   ${state.last_balance_usd:.4f}")
    logger.info(f"{'='*60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Production SFT instruction pair generator for Gleam",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # NeuralWatt, ~4.7K files → ~14K pairs, resume-safe
  NEURALWATT_API_KEY=sk-xxx python src/gleam/generate.py \\
      --provider neuralwatt --max-files 4700

  # Cerebras (fast: ~2h for full run, ~3K tok/s)
  CEREBRAS_API_KEY=csk-xxx python src/gleam/generate.py \\
      --provider cerebras --max-files 4700

  # Resume after crash (just re-run same command)
  NEURALWATT_API_KEY=sk-xxx python src/gleam/generate.py \\
      --provider neuralwatt --max-files 4700

  # Z.AI fallback
  ZAI_API_KEY=xxx python src/gleam/generate.py \\
      --provider zai --max-files 4700
        """,
    )
    parser.add_argument(
        "--provider", choices=["neuralwatt", "cerebras", "zai"], default="neuralwatt",
        help="API provider (default: neuralwatt)",
    )
    parser.add_argument(
        "--max-files", type=int, default=4700,
        help="Max files to process (3 pairs per file, default: 4700 ≈ 14K pairs)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/sft/generation.jsonl"),
        help="Output JSONL path (default: data/sft/generation.jsonl)",
    )
    parser.add_argument(
        "--state-file", type=Path, default=None,
        help="State file path (default: data/sft/generation_state.json)",
    )
    parser.add_argument(
        "--log-file", type=Path, default=None,
        help="Log file path (default: data/sft/generation.log)",
    )
    parser.add_argument(
        "--corpus-path", type=str, default=None,
        help="Path to local corpus Parquet, or 'huggingface' to download from HF (default: huggingface)",
    )
    parser.add_argument(
        "--balance-limit", type=float, default=0.50,
        help="Stop when NeuralWatt balance drops below this (default: 0.50). Ignored for other providers.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--max-retries", type=int, default=3,
        help="Max retries per request (default: 3)",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help=(
            "Number of concurrent workers. Use >1 for Cerebras "
            "(rate limit: 1K RPM). 10 workers ≈ 40min for full run. "
            "(default: 1)"
        ),
    )
    parser.add_argument(
        "--max-input-tokens", type=int, default=3000,
        help=(
            "Drop files whose source code likely exceeds this many input tokens. "
            "Estimated from char count (r=0.956 correlation). Large files produce "
            "worse output at higher cost. Set to 0 to disable. (default: 3000)"
        ),
    )

    args = parser.parse_args()

    # Default paths
    output_dir = Path("data/sft")
    output_dir.mkdir(parents=True, exist_ok=True)

    state_file = args.state_file or output_dir / "generation_state.json"
    log_file = args.log_file or output_dir / "generation.log"

    # Setup logging
    logger = setup_logging(log_file)
    logger.info(f"Generation started at {datetime.now(timezone.utc).isoformat()}")
    logger.info(f"Provider: {args.provider} | Max files: {args.max_files} | Output: {args.output}")

    # Check API key
    cfg = PROVIDERS[args.provider]
    if not os.environ.get(cfg["api_key_env"]):
        logger.error(f"ERROR: {cfg['api_key_env']} environment variable not set")
        sys.exit(1)

    # Load state
    state = GenerationState(state_file, logger)

    # Seed Python random for reproducible CODE_GEN_INSTRUCTIONS picks
    random.seed(args.seed)

    # Load files
    files = load_corpus_files(
        corpus_path=args.corpus_path,
        max_files=args.max_files,
        seed=args.seed,
        exclude_repos=["monks_of_style"],  # semi-generated, exclude
        max_input_tokens=args.max_input_tokens,
        logger=logger,
    )

    if not files:
        logger.error("No files to process")
        sys.exit(1)

    # Run
    try:
        run_generation(
            provider=args.provider,
            files=files,
            output_path=args.output,
            state=state,
            logger=logger,
            balance_limit=args.balance_limit,
            max_retries=args.max_retries,
            workers=args.workers,
        )
    except KeyboardInterrupt:
        logger.info("\n⚠ Interrupted by user. State saved. Re-run to resume.")
        state.save()
    except Exception as e:
        logger.error(f"\n⚠ Unexpected error: {e}", exc_info=True)
        state.save()
        raise


if __name__ == "__main__":
    main()
