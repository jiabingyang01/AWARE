"""AWARE: Action-entropy-Weighted, Attention-guided Retrieval of observation Evidence.

A training-free, inference-time rule for Vision-Language-Action policies. At every decoder layer it
reads how undecided each element of the predicted action chunk still is, and blends observation
evidence back into that element's FFN output in proportion to that uncertainty and to network depth.

Nothing is installed unless ``AWARE_ALPHA`` is set, so a policy with the import in place but the
variable unset is byte-identical to the released one.

Entry points
------------
``hooks.install(vla, use_proprio=...)``          OpenVLA-OFT (action chunk decoded in parallel)
``hooks.install_generic(lm, n_obs=, n_act=...)`` any decoder-only backbone; see integrations/
``pi0.maybe_install(model)``                     openpi pi0 / pi0.5, PyTorch path

The rule itself lives once, in ``hooks._State``. Every backbone adapter only says where the
observation rows and the action rows are; none of them re-implements the operator.
"""
from . import paths                                 # noqa: F401  (no torch needed)

# hooks pulls in torch; keep the package importable for path/config use without it.
try:
    from . import hooks                             # noqa: F401
    from .hooks import (install, install_generic, enabled,          # noqa: F401
                        mark_episode, set_instruction_span)
    _HOOKS = True
except ImportError:                                  # torch missing
    _HOOKS = False

__all__ = ["paths"] + (["hooks", "install", "install_generic", "enabled",
                        "mark_episode", "set_instruction_span"] if _HOOKS else [])
__version__ = "0.1.0"
