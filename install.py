#!/usr/bin/env python3
"""Apply (or revert) the gfx1030/RDNA2 fixes to a ComfyUI Windows portable install.

Fixes applied:
  1. patches/comfyui_gfx1030.patch  -> forces torch reference paths where
     comfy_kitchen's HIP kernels are numerically broken on RDNA2.
  2. stub/comfy_kitchen/*           -> local package that shadows the installed
     comfy-kitchen so its broken kernels are never dispatched.

Tested on: ComfyUI 0.42.1 portable (AMD), torch 2.10.0+rocm7.14.0a20260612,
comfy-kitchen 0.2.39, RX 6950 XT (gfx1030), Windows 11.

Usage:
  python install.py <path-to-ComfyUI_windows_portable>            # apply
  python install.py <path-to-ComfyUI_windows_portable> --revert   # undo
  python install.py <path-to-ComfyUI_windows_portable> --dry-run  # show plan
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
PATCH = REPO / "patches" / "comfyui_gfx1030.patch"
STUB_SRC = REPO / "stub" / "comfy_kitchen"
BACKUP_DIR = ".gfx1030_backup"

CORE_FILES = [
    "comfy/ldm/modules/attention.py",
    "comfy/ldm/moss_image/model.py",
    "comfy/text_encoders/llama.py",
]
STUB_FILES = ["__init__.py", "tensor.py", "int8_decode.py"]


def fail(msg: str) -> None:
    print(f"ERROR: {msg}")
    sys.exit(1)


def find_comfyui_dir(portable: Path) -> Path:
    c = portable / "ComfyUI"
    if not (c / "comfy" / "model_management.py").exists():
        # allow being pointed at the ComfyUI subdir directly
        if (portable / "comfy" / "model_management.py").exists():
            return portable
        fail(f"{portable} does not look like a ComfyUI portable install (no ComfyUI/comfy/).")
    return c


def backup_file(comfyui: Path, rel: str) -> Path:
    src = comfyui / rel
    dst = comfyui / BACKUP_DIR / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)
    return dst


def git_apply(comfyui: Path, revert: bool) -> bool:
    git = shutil.which("git")
    if not git:
        return False
    args = [git, "-C", str(comfyui), "apply", "--3way", str(PATCH)]
    if revert:
        args = [git, "-C", str(comfyui), "apply", "--reverse", "--3way", str(PATCH)]
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode == 0:
        return True
    print(f"  git apply failed:\n{r.stdout}\n{r.stderr}")
    return False


def patch_apply(comfyui: Path, revert: bool) -> bool:
    patch_exe = shutil.which("patch")
    if not patch_exe:
        return False
    args = [patch_exe, "-p1", "--forward" if not revert else "--reverse"]
    r = subprocess.run(args, cwd=str(comfyui), input=PATCH.read_text(encoding="utf-8"),
                       capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    if "previously applied" in out:
        # idempotent re-apply: hunks already in place — clean stray .rej files and succeed
        for rej in comfyui.rglob("*.rej"):
            rej.unlink(missing_ok=True)
        return True
    if r.returncode == 0:
        return True
    print(f"  patch failed:\n{r.stdout}\n{r.stderr}")
    return False


def normalize_eol(comfyui: Path) -> None:
    """Match each patched file's EOL style to its backup, so the tree stays consistent."""
    for rel in CORE_FILES:
        bak = comfyui / BACKUP_DIR / rel
        cur = comfyui / rel
        if not bak.exists():
            continue
        b = bak.read_bytes()
        if b"\r\n" in b and b"\r\n" not in cur.read_bytes():
            data = cur.read_bytes().replace(b"\n", b"\r\n")
            cur.write_bytes(data)


def apply(comfyui: Path, dry: bool) -> None:
    for rel in CORE_FILES:
        if not (comfyui / rel).exists():
            fail(f"missing {rel} — is this a compatible ComfyUI version?")
    if dry:
        print(f"Would patch: {', '.join(CORE_FILES)}")
        print(f"Would install stub: ComfyUI/comfy_kitchen/{{{','.join(STUB_FILES)}}}")
        return

    already = all(
        "gfx1030 (RDNA2) fix" in (comfyui / rel).read_text(encoding="utf-8", errors="ignore")
        for rel in CORE_FILES
    )
    if already:
        print("Core files already carry the gfx1030 patch — skipping patch step.")
    else:
        print("Backing up originals to ComfyUI/%s/ ..." % BACKUP_DIR)
        for rel in CORE_FILES:
            backup_file(comfyui, rel)

        if not (git_apply(comfyui, revert=False) or patch_apply(comfyui, revert=False)):
            fail("could not apply patch (no git/patch, or context drift from a newer "
                 "ComfyUI). Apply patches/comfyui_gfx1030.patch manually, then re-run.")
        normalize_eol(comfyui)

    kdir = comfyui / "comfy_kitchen"
    if kdir.exists() and not (kdir / "__init__.py").read_text(encoding="utf-8", errors="ignore").startswith("# gfx1030"):
        shutil.move(str(kdir), str(kdir) + ".pre_gfx1030")
        print("  moved existing ComfyUI/comfy_kitchen -> comfy_kitchen.pre_gfx1030")
    kdir.mkdir(exist_ok=True)
    for f in STUB_FILES:
        shutil.copy2(STUB_SRC / f, kdir / f)
    print("Installed comfy_kitchen stub.")

    print("\nDone. Launch with scripts/run_moss_gfx1030.bat (or add these flags yourself):")
    print("  --use-pytorch-cross-attention --disable-comfy-compiler --disable-dynamic-vram")
    print("\nIMPORTANT: do NOT add a ModelSamplingSD3 node to MOSS-Image workflows —")
    print("it rescales timesteps x1000 and produces pure noise. The official template")
    print("(Workflow > Browse Templates > MOSS-Image) is already correct.")


def revert(comfyui: Path, dry: bool) -> None:
    bak = comfyui / BACKUP_DIR
    if not bak.exists():
        fail(f"no backup dir at {bak} — nothing to revert (or never applied).")
    if dry:
        print(f"Would restore {len(CORE_FILES)} core files from {bak} and remove the stub.")
        return
    for rel in CORE_FILES:
        b = bak / rel
        if b.exists():
            shutil.copy2(b, comfyui / rel)
    kdir = comfyui / "comfy_kitchen"
    if kdir.exists():
        shutil.rmtree(kdir)
        print("Removed comfy_kitchen stub.")
    print("Reverted core files. (If a comfy_kitchen.pre_gfx1030 dir exists, you may delete it.)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("portable", help="path to ComfyUI_windows_portable (or the ComfyUI dir)")
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    comfyui = find_comfyui_dir(Path(a.portable))
    print(f"ComfyUI dir: {comfyui}")
    (revert if a.revert else apply)(comfyui, a.dry_run)


if __name__ == "__main__":
    main()
