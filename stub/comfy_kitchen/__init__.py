# gfx1030 (RX 6950 XT) fix: stub that shadows the real comfy_kitchen package so the
# server falls back to pure-torch ops. Kitchen's HIP kernels (rms_adaln, rms_rope_split_half,
# mm_int8/dequantize) are broken on RDNA2 nightlies and its import alone corrupted server state.
# gfx1030 (RX 6950 XT) fix: shadow the real comfy_kitchen package with a lazy disabled stub.
# Why: kitchen's HIP kernels (rms_adaln, rms_rope_split_half, mm_int8/w4a8 dequantize) are
# numerically broken on RDNA2 (gfx1030) with the rocm 7.14 nightly, and importing the real
# package corrupts long-running server state (all-bf16 renders came out as pure noise while
# the identical pipeline in a kitchen-free process rendered perfectly).
# Any attribute access returns a falsy dummy; availability probes (name ends in
# "_is_available") answer False, real kernel calls raise so failures are loud.
# _gfx1030_disabled lets patched call sites (comfy/text_encoders/llama.py apply_rope)
# take their torch fallback instead of hitting the raiser.
_gfx1030_disabled = True

class _DisabledCall:
    def __init__(self, name):
        self._name = name
    def __call__(self, *a, **k):
        if self._name.endswith("_is_available"):
            return False
        raise ImportError(f"comfy_kitchen.{self._name} is disabled on this machine (gfx1030 broken HIP kernels)")
    def __bool__(self):
        return False
    def __getattr__(self, name):
        return _DisabledCall(f"{self._name}.{name}")

class _Disabled:
    def __bool__(self):
        return False
    def __getattr__(self, name):
        return _DisabledCall(name)

def __getattr__(name):
    return _DisabledCall(name)
