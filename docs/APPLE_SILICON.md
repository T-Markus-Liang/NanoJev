# Apple Silicon inference

NanoJev's released checkpoints can run on Apple Silicon through PyTorch MPS. Training and the CUDA-specific research diagnostics remain unchanged; this path is for checkpoint inference and the local HTTP service.

## Setup

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements-toy.txt
```

Download one checkpoint directory containing `config.json`, `best.safetensors`, `backbone_config/`, and `tokenizer/`. Then run:

```bash
.venv/bin/python scripts/predict_toy_decisions.py \
  --checkpoint-dir /path/to/checkpoint \
  --input research/toy_inference_example.json \
  --device auto \
  --precision auto
```

`auto` selects CUDA with BF16 when supported, otherwise MPS or CPU with FP32. MPS and CPU intentionally reject BF16 in this implementation so numerical behavior is explicit.

Start the local service with:

```bash
.venv/bin/python scripts/serve_decisions.py \
  --checkpoint-dir /path/to/checkpoint \
  --device auto \
  --precision auto
```

## Reproduction result

The `local_atomic_seed17` checkpoint was verified on an Apple M5 Max with 128 GB unified memory using PyTorch 2.14.0 and Transformers 5.17.0:

| Split | Questions | Accuracy | NLL | Brier |
|---|---:|---:|---:|---:|
| test | 176 | 0.778409 | 0.440683 | 0.287282 |
| OOD | 64 | 0.765625 | 0.446637 | 0.312017 |

These accuracies match the published checkpoint results. A warm request containing three questions and eight candidate paths had a median latency of about 65 ms. A 255-candidate Choice request completed in about 1.28 seconds. These timings describe this machine and FP32 MPS only; they are not cross-provider benchmarks.

## Scope

- The model is a structured decision model, not a chat or code-generation model.
- The released checkpoint stores roughly 596 million FP32 parameters and occupies about 2.39 GB on disk.
- MPS inference is verified. Full training reproducibility still requires substantial CUDA-specific changes and has not been claimed here.
- Candidate paths currently repeat shared prefixes. MPS support does not add tree-prefix sharing.
