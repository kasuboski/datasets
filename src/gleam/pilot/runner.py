"""Pilot runner for comparing Z.AI GLM-5.1 vs NeuralWatt on Gleam SFT generation.

Usage:
  python -m datasets.gleam.pilot.runner zai [--sample-size 20]
  python -m datasets.gleam.pilot.runner neuralwatt [--sample-size 20]
  python -m datasets.gleam.pilot.runner both [--sample-size 20]

Outputs:
  data/pilot/{provider}_approach_a.jsonl   — single-prompt results
  data/pilot/{provider}_approach_b.jsonl   — agent-session results (Z.AI only)
  data/pilot/pilot_summary.json            — aggregate stats
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.gleam.pilot.prompts import (
    PromptSpec,
    build_agent_session,
    build_single_prompts,
)
from src.gleam.pilot.sample import (
    SampledFile,
    print_sample_summary,
    sample_files,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("data/pilot")

PROVIDERS = {
    "zai": {
        "base_url": "https://api.z.ai/api/coding/paas/v4",
        "model": "glm-5.1",
        "api_key_env": "ZAI_API_KEY",
        "approaches": ["a", "b"],  # supports both single + agent session
        "gap_seconds": 3.0,
        "max_concurrent": 1,  # conservative for Z.AI
        "extra_body": {"thinking": {"type": "disabled"}},  # disable reasoning tokens
    },
    "neuralwatt": {
        "base_url": "https://api.neuralwatt.com/v1",  # OpenAI-compatible
        "model": "qwen3.6-35b-fast",  # Fast variant — reasoning model returns content=None
        "api_key_env": "NEURALWATT_API_KEY",
        "approaches": ["a"],  # only single prompts
        "gap_seconds": 0.5,
        "max_concurrent": 2,
        "extra_body": {},
    },
}


# ---------------------------------------------------------------------------
# Live progress display
# ---------------------------------------------------------------------------

class LiveStats:
    """Tracks and displays live progress during pilot execution."""

    def __init__(self, provider: str, approach: str, total: int):
        self.provider = provider
        self.approach = approach
        self.total = total
        self.completed = 0
        self.failed = 0
        self.start_time = time.time()
        self.times: list[float] = []  # per-request wall times
        self.tok_per_sec: list[float] = []
        self.input_tokens: list[int] = []
        self.output_tokens: list[int] = []

    def record(self, wall_time: float, in_tok: int, out_tok: int) -> None:
        self.completed += 1
        self.times.append(wall_time)
        self.input_tokens.append(in_tok)
        self.output_tokens.append(out_tok)
        if wall_time > 0 and out_tok > 0:
            decode_time = wall_time  # approximate; TTFT not available
            self.tok_per_sec.append(out_tok / decode_time)
        self._print_status()

    def record_failure(self) -> None:
        self.failed += 1
        self.completed += 1
        self._print_status()

    def _print_status(self) -> None:
        elapsed = time.time() - self.start_time
        avg_time = sum(self.times) / len(self.times) if self.times else 0
        remaining = self.total - self.completed
        eta = avg_time * remaining if self.times else 0

        avg_tps = (
            sum(self.tok_per_sec) / len(self.tok_per_sec)
            if self.tok_per_sec else 0
        )
        total_in = sum(self.input_tokens)
        total_out = sum(self.output_tokens)

        bar_filled = int(30 * self.completed / self.total)
        bar = "█" * bar_filled + "░" * (30 - bar_filled)

        sys.stdout.write(f"\r  [{bar}] {self.completed}/{self.total} "
                         f"| {elapsed:.0f}s elapsed, ETA {eta:.0f}s "
                         f"| avg {avg_time:.1f}s/req "
                         f"| {avg_tps:.0f} tok/s "
                         f"| {total_in:,} in + {total_out:,} out tok"
                         f"| {self.failed} failed")
        sys.stdout.flush()

    def finish(self) -> dict[str, Any]:
        elapsed = time.time() - self.start_time
        avg_tps = (
            sum(self.tok_per_sec) / len(self.tok_per_sec)
            if self.tok_per_sec else 0
        )
        p50_time = sorted(self.times)[len(self.times) // 2] if self.times else 0
        return {
            "provider": self.provider,
            "approach": self.approach,
            "total": self.total,
            "completed": self.completed,
            "failed": self.failed,
            "elapsed_seconds": round(elapsed, 2),
            "total_input_tokens": sum(self.input_tokens),
            "total_output_tokens": sum(self.output_tokens),
            "avg_tok_per_sec": round(avg_tps, 1),
            "p50_request_seconds": round(p50_time, 2),
            "avg_request_seconds": round(sum(self.times) / len(self.times), 2) if self.times else 0,
        }


# ---------------------------------------------------------------------------
# API clients
# ---------------------------------------------------------------------------

def _get_client(provider: str):
    """Create an OpenAI-compatible client for the provider."""
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai package required. Install with: pip install openai")
        sys.exit(1)

    cfg = PROVIDERS[provider]
    api_key = os.environ.get(cfg["api_key_env"])
    if not api_key:
        print(f"ERROR: {cfg['api_key_env']} environment variable not set")
        sys.exit(1)

    return OpenAI(
        api_key=api_key,
        base_url=cfg["base_url"],
    )


def _call_api(
    client,
    provider: str,
    messages: list[dict],
    max_tokens: int = 4096,
) -> dict[str, Any]:
    """Make a single API call and return timing + response data."""
    cfg = PROVIDERS[provider]
    model = cfg["model"]

    t0 = time.time()
    try:
        # Build kwargs — only include extra_body if non-empty
        kwargs = dict(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
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

        # Extract token counts if available
        if hasattr(response, "usage") and response.usage:
            result["input_tokens"] = response.usage.prompt_tokens or 0
            result["output_tokens"] = response.usage.completion_tokens or 0
            # Capture reasoning tokens if present
            if hasattr(response.usage, "completion_tokens_details") and response.usage.completion_tokens_details:
                result["reasoning_tokens"] = response.usage.completion_tokens_details.reasoning_tokens or 0

        # NeuralWatt-specific: capture energy, cost, duration from response
        if provider == "neuralwatt":
            raw = response.model_dump()
            # NeuralWatt returns rich energy/cost data at top level
            if "energy" in raw and isinstance(raw["energy"], dict):
                result["nw_energy"] = raw["energy"]
            if "cost" in raw and isinstance(raw["cost"], dict):
                result["nw_cost"] = raw["cost"]
            # Store raw keys on first request for debugging
            if not hasattr(_call_api, "_nw_raw_saved"):
                result["_raw_response_keys"] = list(raw.keys())
                _call_api._nw_raw_saved = True

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
# Runners
# ---------------------------------------------------------------------------

def run_approach_a(
    provider: str,
    files: list[SampledFile],
) -> list[dict]:
    """Approach A: Single prompt per task type per file."""
    client = _get_client(provider)
    cfg = PROVIDERS[provider]
    approach = "a"

    # Build all prompts
    all_specs: list[PromptSpec] = []
    for f in files:
        specs = build_single_prompts(
            file_id=f.id,
            code=f.code,
            file_path=f.file_path,
            repo_name=f.repo_name,
        )
        all_specs.extend(specs)

    print(f"\n{'='*60}")
    print(f"APPROACH A: Single prompt | {provider} | {cfg['model']}")
    print(f"  {len(all_specs)} requests ({len(files)} files × 3 task types)")
    print(f"  Gap: {cfg['gap_seconds']}s between requests")
    print(f"{'='*60}")

    stats = LiveStats(provider, approach, len(all_specs))
    results = []

    output_path = OUTPUT_DIR / f"{provider}_approach_a.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for i, spec in enumerate(all_specs):
        result = _call_api(client, provider, spec.messages)

        record = {
            "request_id": f"{provider}_a_{i:04d}",
            "provider": provider,
            "model": cfg["model"],
            "approach": "single",
            "task_type": spec.task_type,
            "source_file_id": spec.source_file_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **result,
        }

        # Write immediately for crash resilience
        with open(output_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        if result.get("error"):
            stats.record_failure()
            print(f"\n  ⚠ Request {i+1} failed: {result['error'][:80]}")
        else:
            stats.record(
                result["wall_time_ms"] / 1000,
                result["input_tokens"],
                result["output_tokens"],
            )

        results.append(record)

        # Rate limiting gap
        if i < len(all_specs) - 1:
            time.sleep(cfg["gap_seconds"])

    print()  # newline after progress bar
    summary = stats.finish()
    _print_summary(summary)
    return results


def run_approach_b(
    provider: str,
    files: list[SampledFile],
) -> list[dict]:
    """Approach B: Agent session — analyze then generate 3 pairs."""
    client = _get_client(provider)
    cfg = PROVIDERS[provider]
    approach = "b"

    # Total requests = files × 2 turns
    total_requests = len(files) * 2

    print(f"\n{'='*60}")
    print(f"APPROACH B: Agent session | {provider} | {cfg['model']}")
    print(f"  {len(files)} sessions × 2 turns = {total_requests} requests")
    print(f"  Gap: {cfg['gap_seconds']}s between requests")
    print(f"{'='*60}")

    stats = LiveStats(provider, approach, total_requests)
    results = []

    output_path = OUTPUT_DIR / f"{provider}_approach_b.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for i, file in enumerate(files):
        turn1_messages, turn2_user = build_agent_session(
            file_id=file.id,
            code=file.code,
            repo_name=file.repo_name,
        )

        # --- Turn 1: Analyze ---
        result1 = _call_api(client, provider, turn1_messages)

        turn1_record = {
            "request_id": f"{provider}_b_{i:04d}_turn1",
            "provider": provider,
            "model": cfg["model"],
            "approach": "agent_session",
            "task_type": "analysis",
            "source_file_id": file.id,
            "turn": 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **result1,
        }

        with open(output_path, "a") as f:
            f.write(json.dumps(turn1_record) + "\n")

        if result1.get("error"):
            stats.record_failure()
            print(f"\n  ⚠ Session {i+1} turn 1 failed: {result1['error'][:80]}")
            stats.record_failure()  # count turn 2 as failed too
            results.append(turn1_record)
            time.sleep(cfg["gap_seconds"])
            continue

        stats.record(
            result1["wall_time_ms"] / 1000,
            result1["input_tokens"],
            result1["output_tokens"],
        )

        time.sleep(cfg["gap_seconds"])

        # --- Turn 2: Generate pairs ---
        # Build context with turn 1 response
        assistant_response = result1["content"]
        turn2_messages = (
            turn1_messages
            + [{"role": "assistant", "content": assistant_response}]
            + [{"role": "user", "content": turn2_user}]
        )

        result2 = _call_api(client, provider, turn2_messages, max_tokens=4096)

        turn2_record = {
            "request_id": f"{provider}_b_{i:04d}_turn2",
            "provider": provider,
            "model": cfg["model"],
            "approach": "agent_session",
            "task_type": "multi_pair_generation",
            "source_file_id": file.id,
            "turn": 2,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **result2,
        }

        with open(output_path, "a") as f:
            f.write(json.dumps(turn2_record) + "\n")

        if result2.get("error"):
            stats.record_failure()
            print(f"\n  ⚠ Session {i+1} turn 2 failed: {result2['error'][:80]}")
        else:
            stats.record(
                result2["wall_time_ms"] / 1000,
                result2["input_tokens"],
                result2["output_tokens"],
            )

        results.append(turn1_record)
        results.append(turn2_record)

        # Gap between sessions
        if i < len(files) - 1:
            time.sleep(cfg["gap_seconds"])

    print()  # newline after progress bar
    summary = stats.finish()
    _print_summary(summary)
    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_summary(summary: dict) -> None:
    print(f"\n{'='*60}")
    print(f"SUMMARY: {summary['provider']} approach {summary['approach']}")
    print(f"{'='*60}")
    print(f"  Requests:     {summary['completed']}/{summary['total']} ({summary['failed']} failed)")
    print(f"  Wall time:    {summary['elapsed_seconds']:.1f}s")
    print(f"  Avg req time: {summary['avg_request_seconds']:.2f}s")
    print(f"  P50 req time: {summary['p50_request_seconds']:.2f}s")
    print(f"  Throughput:   {summary['avg_tok_per_sec']:.0f} tok/s")
    print(f"  Tokens:       {summary['total_input_tokens']:,} in + {summary['total_output_tokens']:,} out")
    print()


def save_pilot_summary(summaries: list[dict]) -> None:
    """Save aggregate pilot summary."""
    output_path = OUTPUT_DIR / "pilot_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summaries": summaries,
    }
    with open(output_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"Pilot summary saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pilot runner for Gleam SFT generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Providers:
  zai         Z.AI Coding Plan (GLM-5.1) — runs approach A + B
  neuralwatt  NeuralWatt (Qwen3-Coder-480B) — runs approach A only
  both        Run both providers sequentially

Examples:
  python -m datasets.gleam.pilot.runner zai
  python -m datasets.gleam.pilot.runner neuralwatt --sample-size 10
  python -m datasets.gleam.pilot.runner both
        """,
    )
    parser.add_argument(
        "provider",
        choices=["zai", "neuralwatt", "both"],
        help="Which provider(s) to test",
    )
    parser.add_argument(
        "--sample-size", type=int, default=20,
        help="Number of files to sample from corpus (default: 20)",
    )
    parser.add_argument(
        "--corpus-path", type=Path,
        default=Path("data/processed/gleam_corpus.parquet"),
        help="Path to filtered corpus Parquet",
    )
    parser.add_argument(
        "--approach", choices=["a", "b", "both"], default="both",
        help="Which approach to run (default: both for zai, a for neuralwatt)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible sampling",
    )

    args = parser.parse_args()

    # Sample files
    print(f"Sampling {args.sample_size} files from corpus (seed={args.seed})...")
    files = sample_files(
        corpus_path=str(args.corpus_path),
        n=args.sample_size,
        seed=args.seed,
    )
    print_sample_summary(files)

    if len(files) < args.sample_size:
        print(f"WARNING: Only sampled {len(files)} files (requested {args.sample_size})")

    # Determine which providers to run
    providers = (
        ["zai", "neuralwatt"] if args.provider == "both"
        else [args.provider]
    )

    all_summaries = []

    for provider in providers:
        cfg = PROVIDERS[provider]

        # Check API key
        api_key = os.environ.get(cfg["api_key_env"])
        if not api_key:
            print(f"\n⚠ Skipping {provider}: {cfg['api_key_env']} not set")
            continue

        # Determine approaches
        if args.approach == "both":
            approaches = cfg["approaches"]
        elif args.approach == "a":
            approaches = ["a"]
        elif args.approach == "b":
            approaches = ["b"]
        else:
            approaches = cfg["approaches"]

        # Check that provider supports requested approach
        if args.approach == "b" and "b" not in cfg["approaches"]:
            print(f"\n⚠ {provider} does not support approach B. Skipping.")
            continue

        if "a" in approaches:
            results_a = run_approach_a(provider, files)
            # Compute and save summary
            if results_a:
                successful = [r for r in results_a if not r.get("error")]
                if successful:
                    times = [r["wall_time_ms"] / 1000 for r in successful]
                    tps_list = [
                        r["output_tokens"] / (r["wall_time_ms"] / 1000)
                        for r in successful
                        if r["wall_time_ms"] > 0 and r["output_tokens"] > 0
                    ]
                    s = {
                        "provider": provider,
                        "approach": "a",
                        "total": len(results_a),
                        "completed": len(successful),
                        "failed": len(results_a) - len(successful),
                        "elapsed_seconds": round(sum(times), 2),
                        "total_input_tokens": sum(r["input_tokens"] for r in successful),
                        "total_output_tokens": sum(r["output_tokens"] for r in successful),
                        "avg_tok_per_sec": round(sum(tps_list) / len(tps_list), 1) if tps_list else 0,
                        "p50_request_seconds": round(sorted(times)[len(times)//2], 2),
                        "avg_request_seconds": round(sum(times) / len(times), 2),
                    }
                    all_summaries.append(s)

        if "b" in approaches:
            results_b = run_approach_b(provider, files)
            if results_b:
                successful = [r for r in results_b if not r.get("error")]
                if successful:
                    times = [r["wall_time_ms"] / 1000 for r in successful]
                    tps_list = [
                        r["output_tokens"] / (r["wall_time_ms"] / 1000)
                        for r in successful
                        if r["wall_time_ms"] > 0 and r["output_tokens"] > 0
                    ]
                    s = {
                        "provider": provider,
                        "approach": "b",
                        "total": len(results_b),
                        "completed": len(successful),
                        "failed": len(results_b) - len(successful),
                        "elapsed_seconds": round(sum(times), 2),
                        "total_input_tokens": sum(r["input_tokens"] for r in successful),
                        "total_output_tokens": sum(r["output_tokens"] for r in successful),
                        "avg_tok_per_sec": round(sum(tps_list) / len(tps_list), 1) if tps_list else 0,
                        "p50_request_seconds": round(sorted(times)[len(times)//2], 2),
                        "avg_request_seconds": round(sum(times) / len(times), 2),
                    }
                    all_summaries.append(s)

    # Save overall summary
    if all_summaries:
        save_pilot_summary(all_summaries)

    # Final comparison
    if len(all_summaries) > 1:
        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")
        for s in all_summaries:
            label = f"{s['provider']}_approach_{s['approach']}"
            print(f"  {label:<25s} | {s['avg_request_seconds']:.2f}s/req "
                  f"| {s['avg_tok_per_sec']:.0f} tok/s "
                  f"| {s['completed']}/{s['total']} ok "
                  f"| {s['elapsed_seconds']:.0f}s total")
        print()


if __name__ == "__main__":
    main()
