# MOSS-Image on RDNA2 (RX 6000 / gfx1030) — ComfyUI Windows fix

MOSS-Image (OpenMOSS's 5B text-to-image model) works on an AMD RX 6950 XT under
ComfyUI on Windows — but the stock install produces **pure noise**, crashes on
int8 models, or corrupts output via AMD's dynamic-VRAM streaming. This repo
contains the exact fixes, verified end-to-end: 512² in ~10–40 s with the 8-step
LoRA, correct text rendering ("OPEN 7 DAYS" renders letter-perfect at 1024²).

**Who this is for:** RX 6000/5000-series (RDNA1/RDNA2, gfx103x/gfx101x) owners
on Windows running a recent ComfyUI portable with the ROCm nightly wheels —
especially for models released after Oct 2026 (MOSS-Image, etc.) that use
ComfyUI's `comfy-kitchen` accelerated kernels.

## Symptoms → causes → fixes

| Symptom | Cause | Fix in this repo |
|---|---|---|
| Output is pure noise/static, no error, on any sampler/seed | `comfy_kitchen` HIP kernels broken on RDNA2: fused `rms_adaln` ~300% wrong, fused RMS+RoPE returns NaN | Patch forces torch reference paths; kitchen stubbed out |
| int8/quantized checkpoints crash (`at::cuda::_int_mm`, access violation) | `torch._int_mm` → hipBLASLt broken on RDNA2 (confirmed in [ComfyUI-INT8-Fast-ROCM](https://github.com/patientx/ComfyUI-INT8-Fast-ROCM); kitchen issue #92) | Use bf16 model files (see Models); kitchen stub also blocks the dispatch |
| Text encoder produces garbage embeddings (noise output even with patched DiT) | Same broken kernel family in the TE's fused RoPE | `llama.py` rope fallback under the stub's disable flag |
| Fully/partially corrupted images with dynamic VRAM on | comfy-aimdo weight streaming + ROCr VMM bug ([#16337](https://github.com/Comfy-Org/ComfyUI/issues/16337), fixed in ROCm 10.1) | Launch flag `--disable-dynamic-vram` |
| Torch HIP init access violation at startup | Stock ROCm wheels unstable on gfx1030 | Install the gfx103X nightly wheels (below) |
| You added `ModelSamplingSD3` to the workflow → noise on **any** GPU | MOSS bakes its shift into the model config (multiplier 1.0); that node rescales timesteps ×1000 → garbage conditioning | Don't. The official template is already correct |

## Quick start (Windows portable)

1. **ComfyUI portable** with MOSS support (≥ 0.42.x, [AMD build](https://github.com/comfyanonymous/ComfyUI/releases/latest)).
2. **Torch nightly for gfx103X** (fixes HIP init crashes):
   ```
   cd ComfyUI_windows_portable
   python_embeded\python.exe -m pip install --pre -U --upgrade-strategy eager torch torchvision torchaudio --index-url https://rocm.nightlies.amd.com/v2-staging/gfx103X-all/
   ```
3. **Models** into `ComfyUI\models\` (~28 GB — int8/quantized variants are *exactly*
   what's broken on RDNA2, so bf16 is required, sorry):
   | File | Folder | Size | Source |
   |---|---|---|---|
   | `moss_image_sft_bf16.safetensors` | `diffusion_models\` | 9.5 GB | [Comfy-Org/MOSS-Image](https://huggingface.co/Comfy-Org/MOSS-Image) |
   | `qwen3.5_9b_bf16.safetensors` | `text_encoders\` | 18 GB | [Comfy-Org/MOSS-Image](https://huggingface.co/Comfy-Org/MOSS-Image/blob/main/text_encoders/qwen3.5_9b_bf16.safetensors) |
   | `flux2-vae_small.safetensors` | `vae\` | 238 MB | Comfy-Org/MOSS-Image |
   | `moss_image_sft_8steps_dmd.safetensors` | `loras\` | 1.2 GB | lodestones (8-step distill) |
4. **Apply the fixes** (backs up originals; `--revert` undoes):
   ```
   python install.py C:\path\to\ComfyUI_windows_portable
   ```
5. **Launch** `scripts\run_moss_gfx1030.bat`, or add these flags yourself:
   ```
   --use-pytorch-cross-attention --disable-comfy-compiler --disable-dynamic-vram
   ```
6. **Generate**: use ComfyUI's built-in template (Workflow → Browse Templates →
   MOSS-Image) and add the 8-step LoRA, or use `workflows/moss_image_8step_gfx1030_api.json`.
   Settings: steps **8**, cfg **1.0**, euler/simple, LoRA strength 1.0.
   Base model without LoRA: 50 steps, cfg 3.0, plus SkipLayerGuidanceDiTSimple.

### UI workflow (8 nodes)

`UNETLoader(moss bf16) → LoraLoaderModelOnly(8steps_dmd, 1.0) → KSampler → VAEDecode → SaveImage`,
`CLIPLoader(type=moss_image) → 2× CLIPTextEncode`, `EmptyFlux2LatentImage` → KSampler latent.
**No ModelSamplingSD3 / no sampler-scaling nodes.**

## What the patch actually changes

`patches/comfyui_gfx1030.patch` (3 files, ~25 lines):
- `comfy/ldm/moss_image/model.py` — `modulated_rms_norm` and the fused
  rms+rope attention pre-pass use the torch reference path (kitchen's
  `rms_adaln` measured rel err **3.07**; `rms_rope_split_half` returns NaN).
- `comfy/text_encoders/llama.py` — rope falls back to torch when the stub's
  `_gfx1030_disabled` flag is set (garbage rope → garbage embeddings → noise).
- `comfy/ldm/modules/attention.py` — tolerates the kitchen stub at import time.

`stub/comfy_kitchen/` — a 3-file local package that **shadows** the installed
comfy-kitchen (local dir wins on `sys.path`): availability probes answer False,
kernel calls raise with a clear message, and the `_gfx1030_disabled` flag
signals the source patches. Nothing int8/fp8/fp4 ever dispatches to the broken
HIP kernels.

### Why flags, not just the patch

- `--use-pytorch-cross-attention` — torch SDPA works on this stack and is ~3×
  faster than ComfyUI's sub-quadratic fallback.
- `--disable-comfy-compiler` — the malloc-graph path calls kitchen, which is stubbed.
- `--disable-dynamic-vram` — see corruption table above.

## Verified on

- RX 6950 XT (gfx1030, RDNA2), 16 GB, Windows 11, Adrenalin 26.x
- ComfyUI 0.42.1 portable (AMD), torch 2.10.0+rocm7.14.0a20260612, torchvision 0.25.*, comfy-kitchen 0.2.39 (stubbed)
- NVDA-free zone: none of this affects NVIDIA cards.

If you test on RDNA3/RDNA4, Strix Halo, or Linux, open an issue with your
arch (`python_embeded\python.exe -c "import torch; print(torch.cuda.get_device_name())"`)
and whether the stock install worked — the RDNA2 patches should be harmless
there, but data beats guesses.

## Troubleshooting

- **Patch fails to apply** → your ComfyUI is newer than the tested one; context
  drifted. Apply the diff by hand (it's ~25 lines, each hunk has a comment) or
  open an issue with your `git diff` of the affected files.
- **Still noise after patching** → check the startup log: you should see no
  `comfy-kitchen`/`aimdo`/`DynamicVRAM` activation lines, and all three launch
  flags present. Also confirm you're **not** using ModelSamplingSD3 (see table).
- **First run slow** → the 19 GB text encoder streams from disk; subsequent
  runs are faster.
- **Undervolted/offset GPU profile?** Put the card to stock before debugging —
  an offset as small as −225 mV produces software-bug-looking corruption
  ([#12839](https://github.com/comfyanonymous/ComfyUI/issues/12839)).

## Context & credits

This is one machine's verified answer to a family of open AMD issues:
[#16337](https://github.com/Comfy-Org/ComfyUI/issues/16337) (aimdo corruption),
[#15314](https://github.com/Comfy-Org/ComfyUI/issues/15314) (quant DiT noise),
[#14868](https://github.com/Comfy-Org/ComfyUI/issues/14868) (RDNA2 int8 hang),
[ComfyUI-INT8-Fast-ROCM](https://github.com/patientx/ComfyUI-INT8-Fast-ROCM)
(int8 replacement kernels), and the
[localaimaster ROCm noise/black/crash guide](https://localaimaster.com/blog/comfyui-amd-rocm-image-generation).

MIT licensed. No warranty — but it does render a very nice apple.
