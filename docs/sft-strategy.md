# SFT Strategy Design: Gleam Specialist Fine-Tuning

## Context

- **Target model**: `google/gemma-4-E4B-it` (4.5B effective params, instruction-tuned)
- **Starting point**: IT model, so we need SFT (not CPT) to preserve instruction following
- **Available data**: 22,581 filtered Gleam source files from 1,528 repos
- **Goal**: Model that writes idiomatic, compiling Gleam code and can explain/reason about it

## Training Approach

**SFT on IT model + data mixing to prevent forgetting**

Starting from the IT checkpoint, we do SFT with a mix of:
1. Gleam-specific instruction pairs (the main payload)
2. General instruction-following data (to preserve instruction tuning)
3. General code instruction data (to preserve code capabilities)

This follows the "Dual-stage Mixed Fine-tuning" approach — blending domain data with general data prevents catastrophic forgetting.

---

## Dataset Composition

### What Goes in  (Published Dataset)

**Only Gleam-focused pairs** — every response contains Gleam code or Gleam explanations.

| Category | Percentage | Approx Count | Purpose |
|----------|-----------|-------------|---------|
| **Gleam code generation** | 35% | ~7,000 pairs | Core skill: write Gleam from instructions |
| **Gleam explanation/understanding** | 15% | ~3,000 pairs | Read/understand/explain Gleam code |
| **Gleam translation** | 10% | ~2,000 pairs | Convert TS/JS/Elm/Elixir → Gleam |
| **Gleam completion & refactoring** | 10% | ~2,000 pairs | Complete partial code, improve idiomaticity |

** total: ~14,000 Gleam-focused pairs**

This is intentionally conservative — research shows 10-20K diverse, high-quality pairs is sufficient for specialization. Volume beyond this hits diminishing returns (OpenCodeInstruct paper, GRAPE paper).

### What Gets Mixed at Training Time (NOT in gleam-instruct)

Anti-forgetting data is sampled from existing public datasets and combined with  during training. This keeps the published dataset Gleam-only and lets others choose their own mix ratios.

| Source | Approx Count | Purpose |
|--------|-------------|---------|
|  | ~3,000 pairs | General instruction following |
|  | ~500 pairs | General code (Python) |
|  (future) | ~2,000 pairs | Elixir/Erlang/Haskell generic pairs (if we generate them as a separate dataset) |

**Training-time total: ~20,000 pairs** (14K Gleam + 6K general)

### Why 30% General Data at Training Time?

The "Dual-stage Mixed Fine-tuning" research and practical guides (AWS Nova, Medium guides on catastrophic forgetting) recommend mixing 10-20% general data during domain SFT to prevent forgetting. We use ~30% because Gleam is extremely out-of-distribution for Gemma — the model has barely seen Gleam tokens, so the domain data will cause large gradient updates. More general data acts as a stabilizer.

**Key benefit of this split**: Someone training a different model (e.g., Qwen, Llama) on our  data can choose their own anti-forgetting mix instead of being locked into ours.

---

## Task Type Designs

### 1. Gleam Code Generation (35% — ~7,000 pairs)

**Method**: OSS-Instruct adapted for Gleam
- Take a Gleam code snippet from corpus as seed
- Prompt a frontier model to generate a natural-language instruction that would produce this code
- The code becomes the response; the generated instruction is the prompt

**Prompt template for generation**:
```
You are an expert Gleam programmer. Given this Gleam code snippet as inspiration, 
generate a realistic programming instruction/task that would result in code like this. 
The instruction should be specific enough to guide implementation but not simply 
describe the code verbatim.

Gleam code:
{seed_code}

Generate an instruction and then provide the complete Gleam implementation as the answer.
```

**Quality filters**:
- Response must contain valid Gleam syntax (regex check for `pub fn`, `import`, `fn`, `type`, etc.)
- Response must be different from seed (not just copying)
- Instruction must be specific enough (not "write some code")
- Prefer seeds from official + high-star repos (weighted sampling)

### 2. Gleam Explanation/Understanding (15% — ~3,000 pairs)

**Method**: Take corpus code → generate explanation questions

**Sub-types**:
- "Explain what this code does" (30%)
- "What Gleam concepts does this code demonstrate?" (20%)
- "How does the `use` expression work in this context?" (20%)
- "Explain the type signature of this function" (15%)
- "What is the difference between `let` and `assert`?" (15%)

**Source**: Prioritize files with `///` doc comments (33% of corpus) — richer explanation targets

### 3. Gleam Translation (10% — ~2,000 pairs)

**Method**: Generate pairs where user provides code in another language, model produces Gleam

This is uniquely valuable because:
- Most Gleam learners come from JS/TS/Elixir/Elm
- Translation forces understanding of Gleam idioms
- Gemma already knows these languages well — it's leveraging existing knowledge

**Sub-types**:
- TypeScript → Gleam (40%) — largest incoming population
- JavaScript → Gleam (20%)
- Elixir → Gleam (20%) — similar BEAM ecosystem
- Elm → Gleam (20%) — similar ML-family syntax

**Generation**: Use frontier model to take a Gleam snippet from corpus, translate it to the source language, then the instruction is "Convert this {source_lang} to idiomatic Gleam" with the translated code as input and original Gleam as response.

### 4. Gleam Completion & Refactoring (10% — ~2,000 pairs)

**Completion** (50%): Take a corpus file, remove the last 30-50%, instruction is "Complete this Gleam module"

**Refactoring** (50%): Take a corpus file, prompt frontier model to produce a non-idiomatic version, then instruction is "Refactor this to be more idiomatic Gleam"

**Gleam-specific refactoring targets**:
- Nested `case` → `use` expression
- Explicit pattern matching → record accessors
- Verbose error handling → `result.try` / `use` piping
- Raw tuples → custom types

### 5. Anti-Forgetting Mix (Training-Time Only — NOT in gleam-instruct)

These are sampled from existing public datasets and combined with `gleam-instruct` at training time. They do NOT go into the published `gleam-instruct` dataset.

**Strategy**: Bias toward functional/BEAM languages that share Gleam's paradigm

**Problem with Python-heavy datasets**: nvidia/OpenCodeInstruct is 100% Python (confirmed in paper: "the most extensive code instruction dataset (in Python)"). Training on imperative Python patterns while trying to teach functional Gleam creates competing gradient signals — pattern matching vs loops, immutability vs mutation, algebraic types vs classes. We want the anti-forgetting mix to *reinforce* functional programming patterns.

**Sources (sampled at training time)**:
- **dhuck/functional_code** (765K files, 7 functional languages, CC license) — raw code, not instruction pairs
  - **Elixir** (~XK files) — same BEAM VM as Gleam, shares actor model, pattern matching, immutability
  - **Erlang** (~XK files) — same BEAM VM, shares module system, message passing conventions
  - **Haskell** (~XK files) — ML-family like Gleam, shares type system DNA, type annotations, ADTs
  - If we generate instruction pairs from these, they'd be a separate dataset (e.g., `functional-code-sft`)
- **nvidia/OpenCodeInstruct** (5M pairs, CC-BY-4.0) — sample only ~500 Python pairs as a small general-code anchor
- **HuggingFaceH4/ultrachat_200k** — sample ~3,000 pairs for general instruction following

**Why Elixir/Erlang is ideal anti-forgetting data for a Gleam model**:
- Runs on the same BEAM VM — shared concepts: supervisors, actors, OTP behaviors
- Pattern matching as a core construct (same as Gleam's `case`)
- Immutable data, no loops (same as Gleam)
- Module/function organization (similar to Gleam's `pub fn`)
- Result/error handling conventions (Elixir `{:ok, val} | {:error, reason}` ≈ Gleam `Ok(val) | Error(reason)`)

**Why Haskell is good supplementary anti-forgetting data**:
- Same ML-family type system as Gleam (algebraic data types, type annotations, type inference)
- Similar syntax for type declarations (`type`, `opaque type`)
- Same paradigm: pure functions, composition, pipelines

**Approximate training-time mix**:
| Source | Count | Purpose |
|--------|-------|---------|
| Elixir (from functional_code) | ~1,200 pairs | BEAM VM reinforcement |
| Erlang (from functional_code) | ~500 pairs | BEAM VM reinforcement |
| Haskell (from functional_code) | ~800 pairs | ML-family type system reinforcement |
| Python (from OpenCodeInstruct) | ~500 pairs | General code skill preservation |
| General instruction (from ultrachat) | ~3,000 pairs | Instruction following preservation |

### 6. General Instruction Following (Training-Time Only — NOT in gleam-instruct)

Sampled from existing public datasets at training time. See section 5 for the full anti-forgetting mix table.

**Candidates**:
- **HuggingFaceH4/ultrachat_200k** or similar — diverse conversational instructions (~3,000 pairs)
- **allenai/tulu-3-sft-mixture** — used in Tulu3 post-training, covers math, coding, chat, safety

**Purpose**: This is the "anti-forgetting" buffer. Keeps the model's general reasoning, instruction following, and safety training intact. Does NOT go into the published `gleam-instruct` dataset.

---

## Generation Infrastructure

### Frontier Model for Generation

**Recommendation: GLM-5.1 (Z.AI Coding Plan)** — $0 cost, rate-limited streaming

| Factor | GLM-5.1 (Coding Plan) | Gemini 2.5 Flash | GLM-5.1 (Batch API) |
|--------|----------------------|-------------------|----------------------|
| Cost | **$0** (Coding Plan) | ~$24 | ~$40 (50% batch discount) |
| Quality | Strong (77.8% SWE-bench) | Good | Same |
| Access | OpenAI-compatible endpoint, rate-limited | Streaming only | Batch upload/poll |
| Context window | 200K (131K max output) | 1M | 200K |
| Architecture | 754B MoE (40B activated) | MoE | Same |

**Endpoint**: `https://api.z.ai/api/coding/paas/v4` (OpenAI-compatible)

**Generation flow (rate-limited streaming)**:
1. Prepare all ~14K seed prompts in memory (seed code + task template + Gleam tour context)
2. Loop through seeds, call `POST /v4/chat/completions` with `model: "glm-5.1"`
3. Rate limit: ~3-5 sec between requests (~12-14 hours for full run)
4. Save results incrementally to `.jsonl` (resumable — skip already-completed seeds)
5. Quality filter: syntax validation, dedup, LLM-as-judge

**Why rate-limited streaming over batch**: The Coding Plan is a subscription with generous quotas (not pay-per-token). The batch API is pay-per-token on the standard API. Since we have the Coding Plan, streaming with a simple rate limiter is the cheapest path. The OpenAI-compatible endpoint means we can use `openai` Python SDK with a custom `base_url`.

**Concurrency**: Single-threaded with sleep between requests. Respects the shared Coding Plan quota. Resumable via progress tracking (skip seeds already in output file).

**Estimated time**: ~14K requests × 4 sec avg = ~15 hours as a background job.

### Reference Context for Generation

Before generating, we need to inject Gleam language knowledge into the generation context. This is critical — most frontier models have limited Gleam knowledge.

**Gleam Language Tour**: Scrape `tour.gleam.run` table of contents + key pages
- ~50 pages covering: basics, functions, flow control, data types, advanced
- Estimated ~100-150KB of text
- This goes into the system prompt for the generation model

**System prompt for generation**:
```
You are generating training data for fine-tuning a code LLM to become a Gleam specialist.
Gleam is a type-safe language for the BEAM VM and JavaScript. Here is a reference:

[EXCERPTS FROM LANGUAGE TOUR]

Rules:
- All generated Gleam code must be syntactically valid
- Prefer idiomatic Gleam patterns (use expressions, pipelines, Result types)
- Never use patterns from other languages (no classes, no null, no exceptions)
- Include type annotations where natural
```

### Gleam Language Tour Collection

The tour at `tour.gleam.run` covers:
- **Basics** (18 sections): hello world, modules, types, strings, bools, lists, constants
- **Functions** (10 sections): HOFs, anonymous, captures, generics, pipelines, labelled args
- **Flow control** (10 sections): case, patterns, recursion, guards
- **Data types** (10 sections): tuples, custom types, records, results, bit arrays
- **Advanced** (multiple): external types, use expressions, etc.

**Action**: Scrape all tour pages and the "Everything!" page into a structured reference document.

---

## Quality Control Pipeline

### Automated Checks (every generated pair)

1. **Syntax validation**: `gleam check` on generated code (if we can run it)
   - Fallback: regex check for valid Gleam tokens (`pub fn`, `import`, `fn`, `type`, `case`, `let`)
2. **No hallucinated APIs**: Check imports against known Gleam packages (stdlib, lustre, etc.)
3. **Length bounds**: Response 50-10,000 chars, instruction 20-2,000 chars
4. **Instruction quality**: Must be specific, not generic ("write code" → reject)
5. **Deduplication**: MinHash on instructions to avoid repetitive tasks
6. **No direct copying**: Response must not be identical to the seed code

### LLM-as-Judge Quality Scoring (sample)

On 500 random pairs, use a second model to rate:
- Instruction specificity (1-5)
- Response correctness (1-5)
- Response idiomaticity (1-5)
- Response completeness (1-5)

Reject any pair scoring <3 on any dimension.

### Manual Spot-Check

Review 50-100 pairs manually for:
- Valid Gleam syntax
- Idiomatic patterns (not direct translations from other languages)
- Instruction clarity
- No hallucinated modules/functions

---

## Output Schema

```jsonc
{
  "id": "sft_gleam_00001",
  "source_file_id": "gleam-lang/stdlib/src/list.gleam",  // link back to corpus (nullable)
  "messages": [
    {"role": "system", "content": "You are a helpful Gleam programming assistant."},
    {"role": "user", "content": "Write a function that reverses a list in Gleam"},
    {"role": "assistant", "content": "pub fn reverse(list: List(a)) -> List(a) { ... }"}
  ],
  "task_type": "code_generation",  // code_generation | explanation | translation | completion | refactoring
  "source_language": null,  // null except for translation tasks (e.g., "typescript")
  "difficulty": "intermediate",  // beginner | intermediate | advanced
  "gleam_concepts": ["lists", "recursion"],
  "quality_score": null,  // filled by LLM judge if sampled
  "model_used": "glm-5.1",
  "license": "Apache-2.0",
  "schema_version": 1
}
```

**Why `messages` format (ChatML)**: This is the standard format Unsloth expects. Key benefits:
- Maps directly to Unsloth's `get_chat_template(tokenizer, chat_template="gemma-4")` — no manual template conversion needed
- Supports multi-turn conversations naturally (Unsloth's `conversation_extension` can merge single-turn into multi-turn)
- `train_on_responses_only` can mask user/system tokens so the model only learns to produce assistant responses
- Universal format: works with TRL, axolotl, Unsloth, HuggingFace trainers

**System prompt consistency**: All Gleam pairs use the same system prompt: `"You are a helpful Gleam programming assistant."` General data pairs use `"You are a helpful assistant."` Consistency in system prompts helps the model learn role boundaries.

---

## Execution Plan

### Step 1: Reference Collection (1 day)
- [ ] Scrape Gleam language tour into structured reference
- [ ] Identify and download general code + instruction datasets
- [ ] Build reference document for generation context

### Step 2: Generator Implementation (2-3 days)
- [ ] Build `datasets/gleam/sft_generator.py`
  - Load corpus Parquet
  - Sample strategically (weight by stars, official, diversity)
  - Call GLM-5.1 API (Z.AI Coding Plan) with task-specific prompts + Gleam reference context
  - Write output as JSONL with schema above
- [ ] Implement quality filters
- [ ] Implement deduplication

### Step 3: Pilot Run (1 day)
- [ ] Generate 500 pairs across all task types
- [ ] Run quality checks
- [ ] Manual spot-check 50 pairs
- [ ] Adjust prompts and sampling strategy based on quality

### Step 4: Full Generation (2-3 days, mostly API time)
- [ ] Generate full ~20K pair dataset
- [ ] Run full quality pipeline
- [ ] Mix with general data samples
- [ ] Final dedup and validation

### Step 5: Publish (1 day)
- [ ] Validate all pairs are in standard ChatML format (already should be)
- [ ] Push to HF as `kasuboski/gleam-instruct` — Gleam-focused pairs only, model-agnostic ChatML
- [ ] Document training-time mix instructions in dataset card (how to combine with ultrachat/OpenCodeInstruct/functional_code)
- [ ] Update STATUS.md

**Note on chat templates**: We do NOT convert to Gemma's native template at publish time. The dataset stays in standard ChatML (`role`/`content` messages). The training script applies the correct template via Unsloth's `get_chat_template(tokenizer, chat_template="gemma-4")`. This preserves model-agnosticism and avoids template drift if Gemma updates their format (which happened recently — Google updated all Gemma chat templates for tool calling fixes).

---

---

## Training Configuration (Unsloth + Gemma 4 E4B)

Based on Unsloth's Gemma 4 guide and hyperparameter recommendations.

### Model Loading
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gemma-4-E4B-it",
    max_seq_length=2048,       # start conservative; Unsloth supports 4× longer
    dtype=None,               # auto-detect
    load_in_4bit=True,        # QLoRA — Unsloth recommends E4B QLoRA over E2B LoRA
)
```

### Chat Template
```python
tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")  # non-thinking for E4B
```

### LoRA Configuration
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `r` (rank) | 32 | Higher than default 16 for specialization tasks; 4.5B model has capacity |
| `lora_alpha` | 64 | 2× rank (aggressive learning for domain shift) |
| `use_rslora` | True | Rank-stabilized LoRA — theoretically optimal scaling |
| `target_modules` | all major linear (q,k,v,o,gate,up,down) | Unsloth: "crucial for matching full fine-tuning" |
| `lora_dropout` | 0 | Short training runs; Unsloth optimizes for 0 |
| `bias` | "none" | Faster, less memory, negligible quality impact |
| `use_gradient_checkpointing` | "unsloth" | 30% less VRAM, supports long context |

### Training Hyperparameters
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `learning_rate` | 2e-4 | Unsloth default for SFT; good starting point |
| `num_train_epochs` | 2 | Balance learning vs overfitting; monitor loss |
| `per_device_train_batch_size` | 2 | Conservative for VRAM |
| `gradient_accumulation_steps` | 8 | Effective batch size = 16 (recommended) |
| `weight_decay` | 0.01 | Standard regularization |
| `warmup_steps` | 5% of total | Gradual LR ramp-up |
| `lr_scheduler_type` | "cosine" | Smooth decay |
| `optim` | "adamw_8bit" | Memory-efficient optimizer |

### Train on Completions Only
```python
trainer = train_on_responses_only(trainer, instruction_part="<start_of_turn>user\n", response_part="<start_of_turn>model\n")
```
**Why**: The QLoRA paper shows ~1% accuracy improvement from masking user/system tokens. The model only learns to produce assistant responses, not to reproduce user instructions. Free accuracy win.

### Multi-turn via Conversation Extension
Instead of manually generating multi-turn conversations, use Unsloth's `conversation_extension` parameter to automatically merge single-turn examples into multi-turn:
- Set `conversation_extension=3` to randomly merge 3 rows into 1 conversation
- Apply to ~20% of the dataset for multi-turn coverage
- The rest stays single-turn

### Expected Training Behavior
- **Initial loss of 13-15 is NORMAL** for Gemma 4 E4B (multimodal model quirk)
- Target final loss: 1-3 (converged)
- **Loss below 0.2 = overfitting** — stop training
- If overfitting: scale `lora_alpha` down by 0.5 post-training, or reduce epochs

### Overfitting Mitigations (if needed)
1. Reduce epochs (2 → 1)
2. Increase `weight_decay` (0.01 → 0.1)
3. Add `lora_dropout` (0 → 0.1)
4. Post-training alpha scaling: multiply LoRA alpha by 0.5
5. Weight averaging: merge original + finetuned weights with 50/50 split

---

## Key Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Frontier model generates invalid Gleam | Medium | Syntax validation + Gleam tour reference context |
| Generated instructions are too generic | Medium | Specific prompt engineering + LLM judge filtering |
| Model still forgets general capabilities | Low-Medium | 30% general data mix (15% functional code + 15% general instruction); LoRA with rsLoRA; train_on_responses_only |
| Gleam ecosystem too small for diverse tasks | Medium | Translation tasks add diversity; Evol-Instruct for complexity |
| `monks_of_style` repetitive code pollutes dataset | Low | Exclude from sampling (single repo filter) |

---

## Public Datasets to Use

### For Gleam Generation (seeds)
- **Our corpus**: `data/processed/gleam_corpus.parquet` (22,581 files)

### For Anti-Forgetting Mix (training-time only, NOT in gleam-instruct)
- **dhuck/functional_code** (765K files, 7 functional languages) — primary source for functional code anti-forgetting
  - URL: https://huggingface.co/datasets/dhuck/functional_code
  - Filter for Elixir, Erlang, Haskell (languages closest to Gleam's paradigm)
  - If we generate instruction pairs, they'd be a separate dataset (e.g., `functional-code-sft`)
- **nvidia/OpenCodeInstruct** (5M pairs, CC-BY-4.0, **Python-only**) — sample only ~500 pairs as minimal general-code anchor
  - URL: https://huggingface.co/datasets/nvidia/OpenCodeInstruct
  - Note: Confirmed 100% Python in paper. Use sparingly to avoid paradigm conflict with functional Gleam.
- **HuggingFaceH4/ultrachat_200k** — sample ~3K general instruction following pairs
- Alternative: **allenai/tulu-3-sft-mixture** if license allows

### For Gleam Reference Context (not training data)
- **Gleam Language Tour**: https://tour.gleam.run/table-of-contents/
- **Gleam Writing Guide**: https://gleam.run/writing-gleam/
- **Gleam Externals Guide**: https://gleam.run/documentation/externals/
