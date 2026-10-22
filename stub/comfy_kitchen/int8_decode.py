# gfx1030 stub: int8 KV-cache decode disabled — bf16 encoders never touch this.
class Int8DecodeCache:
    def __init__(self, *a, **k):
        raise ImportError("comfy_kitchen.int8_decode is disabled on this machine (gfx1030 broken HIP kernels)")

def is_available(*a, **k):
    return False
