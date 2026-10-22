@echo off
rem MOSS-Image on RX 6950 XT (gfx1030) - launch flags are REQUIRED:
rem  --use-pytorch-cross-attention : use torch SDPA (3x faster + correct; sub-quadratic path is unreliable here)
rem  --disable-comfy-compiler      : malloc-graph path calls comfy_kitchen, which is stubbed out on this GPU
rem  --disable-dynamic-vram        : aimdo weight streaming produced corrupted renders on this card
cd /d "%~dp0"
python_embeded\python.exe -s ComfyUI\main.py --windows-standalone-build --disable-assets --use-pytorch-cross-attention --disable-comfy-compiler --disable-dynamic-vram
pause
