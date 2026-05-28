# Gleam Code Corpus Pipeline

A data pipeline that collects the [Gleam](https://gleam.run/) programming language ecosystem's code into a structured, attributed corpus for continued pre-training and fine-tuning.

**Published dataset → [kasuboski/gleam-code-corpus](https://huggingface.co/datasets/kasuboski/gleam-code-corpus)** (22,581 files from 1,528 repos, permissively licensed, deduplicated)

## What it does

1. **Discovers** Gleam repositories via GitHub search (`language:gleam`), enriched with Hex.pm metadata
2. **Collects** all `.gleam` source files via shallow clones with full attribution (license, stars, authors, commit SHA)
3. **Filters** out generated files, stubs, unlicensed code, exercism solutions, and duplicate content
4. **Publishes** to Hugging Face as Parquet with three configs: `all`, `official`, `community_top`

## Quick start

```bash
# Install dependencies
uv sync

# Set up tokens (needed for collection)
export GH_TOKEN=$(gh auth token)

# Collect from GitHub (discovers + clones + extracts)
uv run python main.py collect

# Convert JSONL → filtered Parquet
uv run python main.py convert

# Validate the raw corpus
uv run python main.py validate

# Push to Hugging Face (login first: uv run huggingface-cli login)
uv run python main.py push --dataset-id yourname/gleam-code-corpus
```

## CLI commands

| Command | Description |
|---------|-------------|
| `collect` | Discover repos, clone, extract `.gleam` files → JSONL |
| `convert` | JSONL → filtered Parquet (with `--skip-filters` option) |
| `validate` | Run validation checks on JSONL corpus |
| `push` | Upload processed files to Hugging Face |

## Project structure

```
main.py                          # CLI entry point
datasets/
  core/
    license_checker.py           # Normalize + validate license strings
    git_tools.py                 # Shallow clone + file extraction
    hf_hub.py                    # Parquet conversion, filters, HF push
    filters.py                   # Quality filters (generated, stubs, dedup)
  gleam/
    registry.py                  # GitHub search + Hex.pm enrichment
    build.py                     # Collection orchestrator
recon/                           # One-off exploration scripts
```

## Dataset configs

| Config | Records | Description |
|--------|---------|-------------|
| `all` | 22,581 | Full filtered corpus (permissive license, deduplicated) |
| `official` | 296 | Official repos (`gleam-lang`, `gleam-wisp`, `lustre-labs`) |
| `community_top` | 6,702 | Community repos with stars ≥ 10 or Hex downloads ≥ 1,000 |

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) for dependency management
- GitHub token (`GH_TOKEN`) for collection
- Hugging Face token for publishing

## License

MIT
