<div align="center">

# LELA

**A modular, end-to-end entity-linking pipeline.**

*Find entities in text → match them to a knowledge base → swap any stage with one line of JSON.*

[![License](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-≥3.10-blue.svg)](https://www.python.org)
[![Conference](https://img.shields.io/badge/IJCAI--ECAI-2026_Demo-orange)](https://arxiv.org/abs/2605.26956)
[![YAGO Ecosystem](https://img.shields.io/badge/part_of-YAGO_ecosystem-yellow)](https://yago-knowledge.org/ecosystem)

</div>

## Entity Linking with LELA

Entity linking is the task of finding and mapping mentions of entities in text (such as “Paris”) to their corresponding entities in a Knowledge Base (KB) (such as “[yago:Paris](https://yago-knowledge.org/resource/Paris)”). Entity linking usually proceeds in several steps:

```text
  ┌────────┐   ┌──────┐   ┌────────────┐   ┌──────────┐   ┌──────────────┐   ┌──────────┐
  │  text  │ → │ NER  │ → │ candidates │ → │ reranker │ → │ disambiguator│ → │ entities │
  └────────┘   └──────┘   └────────────┘   └──────────┘   └──────────────┘   └──────────┘
                              ▲                                   ▲
                              └────── KB (Custom/YAGO 4.5) ───────┘
```

These steps are often executed by tools that are limited to linking to Wikipedia. **LELA is a modular entity linking system that unites different tools for each step in one unified interface**. Each of the 5 steps (loader → NER → candidate generation → reranking → disambiguation), and even the KB can be chosen from a wide range of pre-configured sources -- with a single config file.

LELA features:
- **A zero-config quickstart** — `git clone && uv sync && uv run python -m lela.cli ...` works on CPU with no model downloads. YAGO 4.5 fetches itself on first use.
- **Compatibility with any KB** — any JSONL file with `id`, `title`, `description` plugs straight in.
- **Choice of different modules** — regex/spaCy/GLiNER for NER, BM25/fuzzy/dense for candidates, cross-encoder/embedder rerankers, and vLLM / Hugging Face Transformers / OpenAI-compatible API disambiguators.
- **Two interfaces** — Python API for embedding into your workflows, a Gradio web UI for hands-on exploration.
- **CPU-friendly defaults, GPU when you need it** — vLLM is an optional extra; everything else runs on a laptop.

## Installing LELA

**Requirements:** Python ≥3.10. A GPU + CUDA 12.x are required only for the `vllm` extra (local LLM disambiguation/reranking).

**Platform support:**
- **Linux** — fully supported, including the `vllm` extra.
- **macOS** — core + `ui` extra supported. `vllm` is not available; use `openai_api` disambiguator pointing at a remote server (or the `transformers` disambiguator for small models on CPU).
- **Windows** — only the commmand line interface is supported

**Installation**: Clone this repository or download it as a ZIP file and unzip it.

### Linux

```bash
cd lela
uv sync
uv sync --extra ui                 # + Gradio web UI
uv sync --extra vllm               # + local vLLM (needs CUDA)
uv sync --all-extras               # everything
uv run python -m lela.cli \
  --config config/quickstart.json \
  --input data/test/sample_doc.txt \
  --output outputs.jsonl
```

### Windows terminal

```bash
cd lela
python -m pip install --upgrade pip
python -m pip install -e .
python -m lela.cli --config config/quickstart.json --input data/test/sample_doc.txt --output outputs.jsonl
```

This runs on CPU with **no model downloads**. The first invocation fetches YAGO 4.5 (a few hundred MB; one-time, cached under `.ner_cache/`). On the sample document `"Albert Einstein was born in Germany. Marie Curie was a pioneering scientist."` you should see:

```jsonl
{"text": "Albert Einstein", "entity_id": "yago:Albert_Einstein", ...}
{"text": "Germany",         "entity_id": "yago:Germany",         ...}
{"text": "Marie Curie",     "entity_id": "yago:Marie_Curie",     ...}
```

For ambiguous mentions you'll want a heavier config — see the [recommended configurations](#recommended-configurations) below.

A pinned core-only `requirements.txt` is also provided for environments where `pip install -e .` doesn't fit; install extras separately with `python -m pip install gradio` / `python -m pip install "vllm>=0.19.0"`.

## Configuring LELA

Pick a row that matches your hardware and quality target:

| Use case | NER | Candidates | Reranker | Disambiguator | Hardware | Config |
|---|---|---|---|---|---|---|
| **Fast / instant demo** | `regex` | `fuzzy` | none | `first` | CPU only | [`config/quickstart.json`](config/quickstart.json) |
| **Better NER, still CPU** | `gliner` | `bm25` | none | `first` | CPU | [`config/lela_bm25_only.json`](config/lela_bm25_only.json) |
| **Strong, no LLM** | `gliner` | `dense` (0.6B) | `cross_encoder` (0.6B) | `first` | CPU works; 1× GPU much faster | [`config/lela_strong_cpu.json`](config/lela_strong_cpu.json) |
| **Strong + LLM via llama.cpp** | `gliner` | `dense` (0.6B) | `cross_encoder` (0.6B) | `openai_api` → `llama-server` | CPU only (quantized model) | [`config/lela_strong_llamacpp.json`](config/lela_strong_llamacpp.json) |
| **Best quality** | `gliner` | `dense` (4B, +context) | `cross_encoder` (4B) | `vllm` (Qwen3-4B) | 1× GPU (~24+ GB) | [`config/lela_example.json`](config/lela_example.json) |
| **API-only (no local GPU)** | `gliner` | `bm25` | none | `openai_api` | CPU + remote LLM | build your own — see [`docs/API.md`](docs/API.md) |

Rough quality / cost trade-off:
- `regex + fuzzy + first` works perfectly when mentions are canonical entity titles (e.g. "Albert Einstein"), and fails on ambiguous mentions.
- Adding `gliner` improves NER quality on noisy/typed text and supports custom entity labels.
- Adding a `dense` or `cross_encoder` reranker is the biggest quality jump when the KB is large (BM25/fuzzy top-1 isn't great by itself).
- An LLM disambiguator (`vllm`, `transformers`, or `openai_api`) handles ambiguity from context through LLM-based reasoning — but costs the most.

The components can be configured either in a JSON configuration file or directly in Python.

```json
{
    "loader": {
        "name": "text"  # or: pdf, docx, html, jsonl, json
    },
    "ner": {
        "name": "gliner",  # or: regex, spacy
        "params": {"labels": ["person", "organization", "location"]},
    },
    "candidate_generator": {"name": "bm25"},
    # or: fuzzy, dense, openai_api_dense
    "reranker": {"name": "llama_server"},
    # or: none, cross_encoder, cross_encoder_vllm, embedder_transformers, embedder_vllm, vllm_api_client
    "disambiguator": {
        "name": "vllm",  # or: first, openai_api, transformers
        "params": {"model_name": "Qwen/Qwen3-4B"},
    },
    "knowledge_base": { # omit entirely to default to YAGO 4.5
        "name": "jsonl",
        "params": {"path": "my_kb.jsonl"},
    },
}
```

See here for a full per-component reference: [`docs/PIPELINE.md`](docs/PIPELINE.md) · [`docs/API.md`](docs/API.md)

## Running LELA

### Command line interface

```bash
python -m lela.cli --config config/quickstart.json --input data/test/sample_doc.txt --output outputs.jsonl
```

Replace the config file by your configuration file, and the input file by your input file.

### Python interface

```Python
config = { ... }   # see above
lela = Lela(config)
results = lela.run("docs/file1.txt")
```

### Web UI

Requires the `ui` extra (see [Install](#install)), and works only on Linux an MacOS:

```bash
uv run python app.py        # or: python app.py
```

Open `http://localhost:7860` and configure the pipeline through the UI. See [`docs/WEB_APP.md`](docs/WEB_APP.md) for details.

### Conversion utilities

The following script will convert YAGO labels to a JSONL KB:
  ```bash
  python -m lela.scripts.convert_yago_labels data/kb/yagoLabels.tsv data/kb/yago_labels_en.jsonl
  ```

## Output format

Each line of the output JSONL contains one document:

```json
{
  "id": "sample_doc",
  "text": "Albert Einstein was born in Germany. ...",
  "entities": [
    {
      "text": "Albert Einstein",
      "start": 0, "end": 15,
      "label": "ENT",
      "context": "Albert Einstein was born in Germany.",
      "entity_id": "yago:Albert_Einstein",
      "entity_title": "Albert_Einstein",
      "entity_description": "...",
      "candidates": [{"entity_id": "...", "score": 1.0, "description": "..."}, ...]
    }
  ],
  "meta": {"source": "data/test/sample_doc.txt"}
}
```

Cache is keyed by file path, mtime, and size, and lives in `.ner_cache/`.

## Documentation

- [Full paper](https://arxiv.org/abs/2601.05192) — full description of the disambiguation method
- [Demo paper](https://arxiv.org/abs/2605.26956) — full description of the pipeline
- [`docs/PIPELINE.md`](docs/PIPELINE.md) — component architecture and the spaCy integration.
- [`docs/API.md`](docs/API.md) — Python API and component config reference.
- [`docs/CLI.md`](docs/CLI.md) — command-line reference and example configs.
- [`docs/WEB_APP.md`](docs/WEB_APP.md) — Gradio web UI.
- [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) — installation and runtime issues.
- [`docs/REQUIREMENTS.md`](docs/REQUIREMENTS.md) — hardware sizing.
- [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) — contributing.


## Citation

If you use LELA in your research, please cite:

```bibtex
@inproceedings{haffoudhi2026lela,
  title     = {{LELA}: an {LLM}-based Entity Linking Approach with Zero-Shot Domain Adaptation},
  author    = {Haffoudhi, Samy and Suchanek, Fabian M. and Holzenberger, Nils},
  booktitle = {The Semantic Web -- ISWC 2026: 25th International Semantic Web Conference,
               Bari, Italy, October 25--29, 2026, Proceedings},
  series    = {Lecture Notes in Computer Science},
  publisher = {Springer},
  address   = {Cham},
  year      = {2026},
  url       = {https://arxiv.org/abs/2601.05192}
}

@inproceedings{haffoudhi2026lelademo,
  title     = {{LELA}: An End-to-End {LLM}-based Entity Linking Framework with Zero-shot Domain Adaptation},
  author    = {Haffoudhi, Samy and Dobri{\v{c}}i{\'{c}}, Nikola and Suchanek, Fabian M. and Holzenberger, Nils},
  booktitle = {Proceedings of the 35th International Joint Conference on Artificial Intelligence,
               {IJCAI-ECAI} 2026, Demonstrations Track},
  year      = {2026},
  url       = {https://arxiv.org/abs/2605.26956}
}
```

## Authors

- [Samy Haffoudhi](https://samyhaff.github.io/)
- [Nikola Dobričić](https://github.com/NDobricic)
- [Fabian Suchanek](https://suchanek.name/)
- [Nils Holzenberger](https://perso.telecom-paristech.fr/holzenberger/)

## Acknowledgements

LELA is part of the [YAGO knowledge graph ecosystem](https://yago-knowledge.org). The work was partially supported by Agence de l’Innovation
de Defense – AID - via Centre Interdisciplinaire d’Etudes pour la Defense et la Securite – CIEDS - (project 2024 - KB- LM).

## License

LELA is licensed under the [Apache License 2.0](LICENSE).
