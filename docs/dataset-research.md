# Dataset Research for Mixed SFT Fine-Tuning

## Executive Summary

Based on research of HuggingFace datasets and existing SFT practices, here are the findings for creating a mixed training set for Gleam SFT fine-tuning.

---

## 1. Functional Code Datasets

### dhuck/functional_code (Raw Code - NOT SFT)
- **URL**: https://huggingface.co/datasets/dhuck/functional_code
- **License**: AFL-3.0 (permissive)
- **Format**: Parquet, raw code files
- **Size**: 611,738 train / 152,935 test examples
- **Languages**: 7 functional programming languages (including Elixir, Erlang, Haskell)
- **Schema**:
  ```json
  {
    "_id": string,
    "repository": string,
    "name": string,
    "content": string,
    "license": null,
    "download_url": string,
    "language": string,  // Key for filtering!
    "comments": string,
    "code": string
  }
  ```
- **Status**: ✅ CONFIRMED - Raw code, NOT instruction pairs. Would need to generate SFT format from this.
- **Recommendation**: Use this as seed data to generate instruction pairs for Elixir/Erlang/Haskell using the same OSS-Instruct method as Gleam.

### Pre-made SFT Datasets for Functional Languages

#### Haskell: eswar-2001/haskell-repo-funcs-as-instruct
- **URL**: https://huggingface.co/datasets/eswar-2001/haskell-repo-funcs-as-instruct
- **License**: Unknown (auto-gated)
- **Format**: CSV/Parquet
- **Status**: ⚠️ Unclear - Limited documentation, gated access. Not recommended as primary source.

#### Finding: **No High-Quality Pre-made Functional SFT Datasets Found**
After extensive search, there are **no** high-quality, pre-made instruction-tuning datasets specifically for:
- Elixir
- Erlang
- Haskell

The closest options are:
1. Raw code datasets (like `dhuck/functional_code`)
2. Multi-language code datasets that may contain some functional language samples

**Recommendation**: Generate instruction pairs from `dhuck/functional_code` using the same OSS-Instruct pipeline used for Gleam data. Create a separate `functional-code-sft` dataset.

---

## 2. General Instruction Tuning Datasets

### HuggingFaceH4/ultrachat_200k
- **URL**: https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k
- **License**: MIT (very permissive)
- **Format**: Parquet, ChatML format
- **Size**: 207,865 train_sft + 231,110 test_sft
- **Schema**:
  ```json
  {
    "prompt": string,
    "prompt_id": string,
    "messages": [
      {
        "content": string,
        "role": string  // "user" or "assistant"
      }
    ]
  }
  ```
- **Description**: Heavily filtered version of UltraChat, used to train Zephyr-7B-β
- **Split to use**: `train_sft` (already formatted for SFT)
- **Status**: ✅ RECOMMENDED

### allenai/tulu-3-sft-mixture
- **URL**: https://huggingface.co/datasets/allenai/tulu-3-sft-mixture
- **License**: ODC-BY-1.0 (Open Data Commons Attribution)
  - ⚠️ Note: Some subsets have non-commercial licenses
- **Format**: Parquet
- **Size**: 939,344 samples
- **Contains**: Multiple datasets mixed (FLAN v2, CoCoNot, WildChat, etc.)
- **Status**: ⚠️ USE WITH CAUTION
  - Mixed licensing makes commercial deployment tricky
  - More complex than needed for ~3K samples

### Comparison: ultrachat_200k vs tulu-3-sft-mixture

| Factor | ultrachat_200k | tulu-3-sft-mixture |
|--------|---------------|-------------------|
| License | MIT (clear) | ODC-BY-1.0 + mixed (complex) |
| Format | ChatML (ready) | Multiple formats (needs conversion) |
| Size | 208K SFT-ready | 939K mixed |
| Quality | High (Zephyr-trained) | High (Tulu 3) |
| Simplicity | ✅ Simple | ⚠️ Complex |
| Commercial use | ✅ Clear | ⚠️ Mixed |

**Recommendation**: Use `HuggingFaceH4/ultrachat_200k` for ~3,000 general instruction pairs. It's simpler, has clearer licensing, and is already in ChatML format matching Unsloth's expectations.

---

## 3. General Code Datasets

### nvidia/OpenCodeInstruct
- **URL**: https://huggingface.co/datasets/nvidia/OpenCodeInstruct
- **License**: ✅ CC-BY-4.0 (confirmed)
- **Format**: Parquet
- **Size**: 5,000,000 samples
- **Schema**:
  ```json
  {
    "id": string,
    "input": string,      // User instruction
    "output": string,     // Assistant response
    "domain": string,
    "generation_algorithm": string,
    "llm_judgement": string,
    "unit_tests": string,
    "tests_execution_status": string,
    "average_test_score": string
  }
  ```
- **Format for training**: Needs conversion to ChatML:
  ```python
  messages = [
    {"role": "user", "content": row["input"]},
    {"role": "assistant", "content": row["output"]}
  ]
  ```
- **Python content**: ~100% Python (confirmed in paper)
- **Status**: ✅ CONFIRMED - Use sparingly (~500 pairs) as general code anchor
- **ChatML-compatible**: Yes, with simple conversion

### Alternative: iamtarun/code_instructions_120k_alpaca
- **URL**: https://huggingface.co/datasets/iamtarun/code_instructions_120k_alpaca
- **License**: Unknown (derived from sahil2801/code_instructions_120k)
- **Format**: Alpaca style (instruction/input/output/prompt)
- **Size**: 121,959 samples
- **Status**: ⚠️ Less preferred to OpenCodeInstruct (unclear licensing, Alpaca format needs more conversion)

**Recommendation**: Use `nvidia/OpenCodeInstruct` for ~500 Python pairs. It has clear CC-BY-4.0 licensing and is high-quality.

---

## 4. Multi-language Code Datasets (Functional Language Samples)

### MathAndMagic/Code-290k-ShareGPT-MarkedLanguage
- **URL**: https://huggingface.co/datasets/MathAndMagic/Code-290k-ShareGPT-MarkedLanguage
- **License**: Unknown (derived from ajibawa-2023/Code-290k-ShareGPT)
- **Format**: Parquet, ShareGPT format
- **Size**: 289,094 samples
- **Schema**:
  ```json
  {
    "id": string,
    "conversations": [
      {
        "from": string,  // "human" or "gpt"
        "value": string
      }
    ],
    "language": string  // Detected programming language!
  }
  ```
- **Status**: ⚠️ POTENTIALLY USEFUL
  - Can filter for functional languages (Elixir, Erlang, Haskell)
  - Language detection is heuristic-based, not 100% accurate
  - Needs format conversion (ShareGPT → ChatML)

**Research Opportunity**: Could query this dataset for Elixir/Erlang/Haskell samples to supplement functional anti-forgetting data without generating from scratch.

---

## 5. Data Mixing Strategy for Unsloth SFTTrainer

### Approach 1: Concatenate Datasets (Recommended)
Use HuggingFace's `datasets.concatenate_datasets()` after converting all to common schema:

```python
from datasets import load_dataset, concatenate_datasets

# The implemented pipeline samples and mixes these JSONL files:
gleam_data = load_dataset("json", data_files="data/sft/gleam-instruct.jsonl", split="train")
functional_data = load_dataset("json", data_files="data/sft/functional_683_chatml.jsonl", split="train")
ultrachat_data = load_dataset("json", data_files="data/sft/ultrachat_3k.jsonl", split="train")
opencode_data = load_dataset("json", data_files="data/sft/opencode_500_chatml.jsonl", split="train")

# scripts/create_mixed_dataset.py concatenates and shuffles all four sources.
mixed_dataset = concatenate_datasets([
    gleam_data,
    functional_data,
    ultrachat_data,
    opencode_data,
]).shuffle(seed=42)
```

### Approach 2: Interleave Datasets
Maintain proportion by interleaving rather than concatenating:

```python
from datasets import interleave_datasets

# Calculate ratios for desired 70/15/15 split
# 14K Gleam, ~3K general, ~3K functional = 20K total
# Ratios: 0.7, 0.15, 0.15

mixed_dataset = interleave_datasets(
    [gleam_data, ultrachat_sampled, functional_sampled],
    probabilities=[0.7, 0.15, 0.15],
    seed=42,
    stopping_strategy="all_exhausted"
)
```

### Approach 3: Custom Dataset with Weighted Sampling
For fine-grained control:

```python
import numpy as np

class WeightedMixedDataset(torch.utils.data.Dataset):
    def __init__(self, datasets, weights):
        self.datasets = datasets
        self.weights = np.array(weights) / np.sum(weights)
        self.cumulative_weights = np.cumsum(self.weights)

    def __len__(self):
        return max(len(d) for d in self.datasets) // min(self.weights)

    def __getitem__(self, idx):
        r = np.random.random()
        for i, cw in enumerate(self.cumulative_weights):
            if r <= cw:
                dataset_idx = i
                break
        sample_idx = idx % len(self.datasets[dataset_idx])
        return self.datasets[dataset_idx][sample_idx]
```

### Recommended: Approach 1 (Concatenate + Shuffle)
- **Pros**: Simple, deterministic, easy to debug
- **Cons**: Less control over exact per-epoch distribution
- **Best for**: First training runs, simpler setups

### Schema Normalization
All datasets must be converted to the same schema before mixing:

**Target schema (ChatML)**:
```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

**Conversion mappings**:
- **ultrachat_200k**: Already has `messages` array ✅
- **OpenCodeInstruct**: `input`→user, `output`→assistant
- **functional_code (generated)**: Same as gleam-instruct
- **gleam-instruct**: Already in ChatML ✅

---

## 6. Practical Proportions: Historical Research Proposal

### The Strategy Document's Proposal: 70% Gleam / 15% Functional / 15% General

| Component | Count | Percentage | Purpose |
|-----------|-------|------------|---------|
| Gleam | 14,053 | 70% | Domain specialization |
| Functional | ~3,000 | 15% | Paradigm reinforcement |
| General | ~3,000 | 15% | Instruction following |

### Research Supporting This Approach

#### 1. "Dual-stage Mixed Fine-tuning" (AWS Nova, various papers)
- **Finding**: Mixing 10-20% general data during domain SFT prevents catastrophic forgetting
- **Rationale**: General data acts as gradient stabilizer for large domain shifts
- **Application**: Gleam is extremely OOD for Gemma → justifies 30% general data (higher than 10-20%)

#### 2. OpenCodeInstruct Paper (NVIDIA, 2025)
- **Finding**: 10-20K diverse, high-quality pairs is sufficient for specialization
- **Finding**: Volume beyond this has diminishing returns
- **Application**: 14K Gleam pairs is in the sweet spot

#### 3. GRAPE Paper (Liu et al., 2024)
- **Finding**: Domain specialization works best with targeted data rather than volume
- **Finding**: Multi-task learning (mixing domains) improves generalization

#### 4. Practical Guides (Medium, various blogs)
- **Finding**: General instruction data preserves safety and reasoning
- **Finding**: Code data preserves coding patterns
- **Finding**: Functional code data specifically helps for functional paradigms

### Why 70/15/15 is Reasonable

1. **70% Gleam**:
   - Primary training signal
   - Sufficient volume for specialization (14K in 10-20K sweet spot)
   - Consistent with research on domain-specific SFT

2. **15% Functional (~3K pairs)**:
   - Reinforces paradigm (BEAM VM, immutability, pattern matching)
   - Avoids competing signals from imperative Python
   - Elixir/Erlang share BEAM VM with Gleam → high transfer
   - Haskell shares ML-family type system → high transfer

3. **15% General (~3K pairs)**:
   - Preserves instruction following
   - Preserves general reasoning
   - Prevents catastrophic forgetting
   - Higher than typical 10% due to Gleam's extreme OOD nature

### Alternative Proportions to Consider

If training shows forgetting or poor generalization:

| Scenario | Adjusted Proportions | Rationale |
|----------|---------------------|-----------|
| High Gleam accuracy, low general | 60/20/20 | More general data |
| Good general, weak Gleam | 80/10/10 | More domain focus |
| Overfitting Gleam patterns | 65/17.5/17.5 | Balance both |
| Paradigm conflict (Python patterns) | 70/20/10 | More functional, less Python |

### Recommended: Start with 70/15/15
- Follows research-backed practices
- Aligns with "dual-stage mixed fine-tuning"
- Provides room for adjustment based on results
- 30% general is conservative for extreme OOD shift

---

## 7. Tools & Libraries for Data Mixing

### HuggingFace Datasets Library
```python
from datasets import load_dataset, concatenate_datasets, interleave_datasets
```

### Unsloth/TRL Integration
```python
from trl import SFTTrainer
from unsloth import FastLanguageModel

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=mixed_dataset,  # Your mixed dataset
    dataset_text_field="messages",  # ChatML format
    max_seq_length=2048,
    # ... other args
)
```

### Data Quality Checks
```python
# Validate ChatML format
def validate_chatml(example):
    assert isinstance(example["messages"], list)
    assert all(isinstance(m, dict) for m in example["messages"])
    assert all("role" in m and "content" in m for m in example["messages"])
    roles = [m["role"] for m in example["messages"]]
    assert roles[-1] == "assistant"  # Last message should be assistant
    return example

validated_dataset = mixed_dataset.map(validate_chatml)
```

---

## 8. Implementation Plan

The implemented JSONL pipeline produced and validated `data/sft/mixed_training.jsonl` with 13,538 records:

| Source | Records | Share |
|--------|--------:|------:|
| `gleam-instruct` | 9,355 | 69.1% |
| `ultrachat` | 3,000 | 22.2% |
| `code-290k-functional` | 683 | 5.0% |
| `opencodeinstruct` | 500 | 3.7% |
| **Total** | **13,538** | **100%** |

### Step 1: Sample General and Functional Datasets
```bash
uv run python scripts/sample_general_datasets.py
```
Outputs:
- `data/sft/ultrachat_3k.jsonl`
- `data/sft/opencode_500_chatml.jsonl`
- `data/sft/functional_683_chatml.jsonl`

### Step 2: Create and Validate the Mixed Dataset
```bash
uv run python scripts/create_mixed_dataset.py
uv run python scripts/validate_dataset.py --path data/sft/mixed_training.jsonl
```

### Step 3: Train
```bash
uv run modal run --detach -m src.gleam.train
```

---

## 9. Historical Key Findings Summary

### ✅ Confirmed
The research recommendations below are historical; the implemented JSONL pipeline and current dataset are documented in Section 8.

1. **nvidia/OpenCodeInstruct**: CC-BY-4.0 license, ChatML-compatible with conversion
2. **HuggingFaceH4/ultrachat_200k**: MIT license, ChatML format ready, best for general instruction
3. **dhuck/functional_code**: Raw functional code, needs SFT generation, good source
4. **Mixing strategy**: `concatenate_datasets()` + shuffle is simple and effective

### ⚠️ Caution
1. **No pre-made functional SFT datasets**: Must generate from `dhuck/functional_code`
2. **tulu-3-sft-mixture licensing**: Mixed ODC-BY + NC licenses, use ultrachat instead
3. **OpenCodeInstruct is Python-only**: Use sparingly to avoid paradigm conflict

### 📋 Recommendations
1. **Use ultrachat_200k** for 3K general instruction pairs (simple, MIT, ChatML-ready)
2. **Generate functional SFT** from `dhuck/functional_code` using OSS-Instruct (~3K pairs)
3. **Use OpenCodeInstruct** for ~500 Python pairs (general code anchor, CC-BY-4.0)
4. **Mix proportion**: 70% Gleam / 15% Functional / 15% General is research-backed
5. **Mixing method**: `concatenate_datasets()` then shuffle for simplicity

### 🎯 Next Steps
1. [x] Sample the implemented JSONL sources
2. [x] Create and validate `data/sft/mixed_training.jsonl` (13,538 records)
3. [ ] Run training with `uv run modal run --detach -m src.gleam.train`
4. [ ] Evaluate and adjust proportions if needed

---

## References

1. NVIDIA OpenCodeInstruct: https://arxiv.org/abs/2504.04030
2. UltraChat 200k: https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k
3. Tulu 3: https://huggingface.co/datasets/allenai/tulu-3-sft-mixture
4. Functional Code Dataset: https://huggingface.co/datasets/dhuck/functional_code
5. AWS Nova SFT Guide (internal)
6. GRAPE Paper: Liu et al., 2024
7. Dual-stage Mixed Fine-tuning: Various sources
