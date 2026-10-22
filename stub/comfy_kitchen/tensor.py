# gfx1030 stub: quant layout API disabled — bf16/fp16 models never touch these;
# any real use raises loudly instead of silently corrupting output.
class _No:
    def __init__(self, *a, **k):
        raise ImportError("comfy_kitchen.tensor is disabled on this machine (gfx1030 broken HIP kernels)")
    def __getattr__(self, name):
        raise ImportError("comfy_kitchen.tensor is disabled on this machine (gfx1030 broken HIP kernels)")

class QuantizedTensor(_No):
    pass

class QuantizedLayout(_No):
    pass

class TensorCoreFP8Layout(_No):
    pass

class TensorCoreNVFP4Layout(_No):
    pass

class TensorCoreConvRotW4A4Layout(_No):
    pass

class TensorCoreMXFP8Layout(_No):
    pass

class TensorWiseINT8Layout(_No):
    pass

class AsymW4A8Int8Layout(_No):
    pass

def register_layout_op(*a, **k):
    raise ImportError("comfy_kitchen.tensor is disabled on this machine (gfx1030 broken HIP kernels)")

def register_layout_class(*a, **k):
    raise ImportError("comfy_kitchen.tensor is disabled on this machine (gfx1030 broken HIP kernels)")

def get_layout_class(*a, **k):
    raise ImportError("comfy_kitchen.tensor is disabled on this machine (gfx1030 broken HIP kernels)")
