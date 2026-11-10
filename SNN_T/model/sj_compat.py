"""SpikingJelly version compatibility.

The model was written against spikingjelly 0.0.0.0.14, which has
`spikingjelly.clock_driven.neuron.MultiStepLIFNode`. spikingjelly >= 2.0 (what a
plain `pip install spikingjelly` gives since 2026) removed `clock_driven`; the
same neuron is `activation_based.neuron.LIFNode(step_mode='m')` there.

LIF neurons hold no trainable weights, so checkpoints load under either version.
"""
try:  # spikingjelly 0.0.0.0.x
    from spikingjelly.clock_driven.neuron import MultiStepLIFNode as _MultiStepLIF

    def MultiStepLIFNode(**kwargs):
        kwargs.setdefault("backend", "torch")
        return _MultiStepLIF(**kwargs)

    from spikingjelly.clock_driven import functional  # reset_net without MemoryModule warnings

    LIF_TYPES = (_MultiStepLIF,)
    SJ_API = "clock_driven"
except ImportError:  # spikingjelly >= 2.0
    from spikingjelly.activation_based import neuron as _neuron

    def MultiStepLIFNode(**kwargs):
        kwargs.pop("backend", None)  # 2.0 picks its own backend
        return _neuron.LIFNode(step_mode="m", **kwargs)

    from spikingjelly.activation_based import functional

    LIF_TYPES = (_neuron.LIFNode,)
    SJ_API = "activation_based"

try:
    from spikingjelly.activation_based import surrogate  # noqa: F401  (both versions)
except ImportError as e:  # pragma: no cover
    raise ImportError("spikingjelly is required: pip install spikingjelly==0.0.0.0.14") from e


def load_checkpoint(path, map_location="cpu"):
    """torch.load that works on PyTorch < 1.13 and on >= 2.6 (weights_only default changed)."""
    import torch
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)
