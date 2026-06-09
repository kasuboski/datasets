# GPU Selection Guide

## Decision Framework

1. **How much VRAM do you need?** Model size + optimizer states + activations + batch size.
2. **Is the workload memory-bound or compute-bound?** Small-batch inference = memory-bound → cheaper GPU fine. Large-batch training = compute-bound → faster GPU matters.
3. **Check utilization after a test run.** If GPU util <50%, you're overpaying.

## VRAM Estimation for Training

QLoRA (4-bit quantized) uses ~4 bits/param for weights + optimizer state. Rough rule:

| Model size | QLoRA VRAM needed | Recommended GPU |
|-----------|-------------------|-----------------|
| 1–3B | 6–10 GB | L4 |
| 3–7B | 10–16 GB | L4, A10 |
| 7–14B | 16–30 GB | A100-40GB, L40S |
| 14–30B | 30–60 GB | A100-80GB, RTX PRO 6000 |
| 30–70B | 60–140 GB | H100, H200 |
| 70B+ | 140+ GB | H200, B200, or multi-GPU |

**Sequence length matters.** Doubling max_seq_length roughly doubles activation memory. A model that fits at 2048 may OOM at 4096.

Full fine-tune (not QLoRA) uses ~16–20 bytes per parameter. 7B model = ~112–140 GB VRAM → needs H100 or A100-80GB:2.

## Pricing (per-hour, from modal.com/pricing)

| GPU | $/sec | $/hr | VRAM | Multi-GPU |
|-----|-------|------|------|-----------|
| T4 | $0.000164 | **$0.59** | 16 GB | Up to 8 |
| L4 | $0.000222 | **$0.80** | 24 GB | Up to 8 |
| A10 | $0.000306 | **$1.10** | 24 GB | Up to 4 |
| L40S | $0.000542 | **$1.95** | 48 GB | Up to 8 |
| A100-40GB | $0.000583 | **$2.10** | 40 GB | Up to 8 |
| A100-80GB | $0.000694 | **$2.50** | 80 GB | Up to 8 |
| RTX PRO 6000 | $0.000842 | **$3.03** | 48 GB | — |
| H100 | $0.001097 | **$3.95** | 80 GB | Up to 8 |
| H200 | $0.001261 | **$4.54** | 141 GB | Up to 8 |
| B200 | $0.001736 | **$6.25** | 192 GB | Up to 8 |

## Cost Optimization Tips

1. **Start with the cheapest GPU that fits.** You can always upgrade.
2. **Use QLoRA/LoRA** instead of full fine-tuning — drops VRAM 3–5x.
3. **L4 is the dev/debug workhorse.** $0.80/hr, 24 GB. Good for testing pipelines on small batches.
4. **H100 auto-upgrades to H200** for free — same price, more VRAM. Don't assume exact 80 GB.
5. **A100 requests may auto-upgrade to A100-80GB** — same price, double VRAM.
6. **Multi-GPU (`gpu="H100:8"`) costs 8x the single-GPU price.** More than 2 GPUs = longer queue times.
7. **Region selection costs 1.5–1.75x base price.** Only use if latency/ compliance requires it.
8. **Non-preemptible execution costs 3x.** Only for jobs that absolutely cannot restart.

## Real-World Examples

### QLoRA fine-tune a 4.5B model (e.g., Gemma 4 E4B)
- VRAM needed: ~10 GB at seq_len=2048, ~17 GB at seq_len=4096
- GPU: L4 for 2048 ($0.80/hr), A100-40GB for 4096 ($2.10/hr)
- 4 hours on A100-40GB ≈ $8.40

### Full fine-tune a 7B model
- VRAM needed: ~120 GB
- GPU: H100:2 ($7.90/hr) or A100-80GB:2 ($5.00/hr)
- 12 hours on A100-80GB:2 ≈ $60

### Serve a 70B model (FP8)
- VRAM needed: ~70 GB weights + KV cache
- GPU: H100 ($3.95/hr) or H200 ($4.54/hr) for more KV cache headroom
- Continuous deployment = ~$95–$109/day

### Batch inference on 8B model
- VRAM needed: ~16 GB
- GPU: L4 ($0.80/hr) — memory-bound, no need for faster GPU
