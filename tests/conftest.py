import sys
import types
import importlib.util


def make_fake_module(name: str):
    """Creating a fake module with a valid spec so importlib does not crash"""
    fake = types.ModuleType(name)
    fake.__spec__ = importlib.util.spec_from_loader(name, loader=None)
    return fake


# blocking hardware acceleration backends
BACKEND_MODULES = ["flash_attn", "flash_attn.ops", "flash_attn.flash_attn_interface", "habana_frameworks", "habana_frameworks.torch", "habana_frameworks.torch.hpu", "torch_npu", "optimum", "optimum.habana", "optimum.neuron", "optimum.graphcore"]


for name in BACKEND_MODULES:
    sys.modules[name] = make_fake_module(name)
