# vLLM on RunPod — the recipe that doesn't kill CUDA

**The recurring failure.** `pip install vllm` (latest) pulls a bleeding-edge torch built
for a newer CUDA than the pod's driver → `libcudart.so.13: cannot open shared object file`
→ `torch.cuda.is_available() == False`, and the whole run is dead. (e.g. vllm 0.23 → torch
2.11 + cu130, which needs a CUDA-13 driver; a 12.4/12.8 pod has no `libcudart.so.13`.)

**Root cause.** vLLM pins a specific torch + CUDA build. `pip install vllm` silently
*uninstalls your torch and installs vLLM's*. If that torch's CUDA is newer than the pod
driver, CUDA goes dark. It is a **version-alignment** problem, not a vLLM bug.

## Verified working recipe — 2026-07-10, RTX 3090, driver 580, base CUDA 12.4

Base image: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` (torch 2.4.1+cu124).

```bash
# 1. Pin an OLD vLLM whose torch matches the 12.x era (NOT latest).
pip install vllm==0.6.3.post1            # torch -> 2.4.0+cu121; CUDA STAYS available

# 2. Remove the extras that now mismatch CUDA and block `import vllm`.
#    (Text LLMs don't need audio/vision; they trip a CUDA-version check on import.)
pip uninstall -y torchaudio torchvision

# 3. Install the training stack ON TOP (order matters: after vLLM; never reinstall torch).
pip install trl==0.17.0 transformers==4.54.1 peft datasets==3.5.0
```

Verified end-to-end on a real pod:

```
torch 2.4.0+cu121   cuda_available = True
vllm 0.6.3.post1 | trl 0.17.0 | transformers 4.54.1 | peft   (all coexist)
vLLM generation:  ~257 tok/s on opt-125m
```

*(The `vllm requires torchvision==0.19` pip warning is harmless — text generation never
touches it.)*

## Rules of thumb (so it never wastes a run again)
1. **`nvidia-smi` first.** Match vLLM → torch → CUDA to the *driver*. Base 12.4 → vllm 0.6.3.
   Newer driver (12.8+) → you can go higher (vllm 0.7/0.8 + torch 2.5/2.6). Never `pip install
   vllm` unpinned.
2. **Install vLLM FIRST**, then transformers/trl/peft. **Never reinstall torch after vLLM.**
3. **Uninstall torchaudio/torchvision** if they came from a different cu build.
4. **Alternative:** use `vllm/vllm-openai` as the pod image (self-consistent torch+CUDA+vLLM),
   but its SSH differs from `runpod/pytorch` — the recipe above keeps the familiar SSH flow.

## Why it matters for GRPO
`train/train_grpo.py --vllm` sets `GRPOConfig(use_vllm=True)` → trl generates rollouts with
vLLM instead of HF `.generate()`. That is the dominant cost (this repo's HF-generate run was
~30 s/step + slow eval); vLLM cuts rollout time **~5–10×**.

## trl + vLLM-GRPO — the version nuance (researched 2026-07-10, NOT yet verified end-to-end)
The inference recipe above (vllm 0.6.3) does **not** work with trl-0.17 GRPO:
* trl 0.17 GRPO needs vllm **≥ 0.7** (`StatelessProcessGroup` in `vllm.distributed.utils`) —
  vllm 0.6.3 lacks it → import error → even non-vLLM GRPO breaks while vllm is installed.
* Worse: **trl 0.17 is *server-mode only*** — you must run `trl vllm-serve` as a separate
  process (weight-sync over a process group), ideally on a 2nd GPU. In-process **colocate**
  mode (single GPU, just `use_vllm=True`) only arrives in **trl ≥ 0.18**.

Two ways to get fast GRPO:
1. **Stay on trl 0.17:** `vllm==0.7.2` + `torch==2.5.1` (cu121 wheel runs on a 12.4/12.8
   driver) — but wire up *server mode* (`trl vllm-serve` + trainer), fiddly on 1 GPU.
2. **⭐ Recommended — upgrade to trl 0.18–0.19:** `use_vllm=True, vllm_mode="colocate"` on a
   single GPU + a matching `vllm 0.8.x` + `torch 2.6` (cu124). Simple, one process.
   (trl **1.x** breaks GRPO — stay in the 0.18–0.19 band.)

Verify the chosen combo on first use (CUDA-align to `nvidia-smi` as above).
