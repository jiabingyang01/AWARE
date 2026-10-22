"""Where the host repositories and caches live.

Every path AWARE's scripts need is read from the environment, with the values used for the paper as
defaults so that nothing breaks on the machines the results came from. Override whatever does not
match your setup:

    export AWARE_OPENVLA_OFT=/path/to/openvla-oft
    export AWARE_OPENPI=/path/to/openpi
    export AWARE_LIBERO=/path/to/LIBERO
    export AWARE_SIMPLER=/path/to/SimplerEnv
    export AWARE_LLAVA_VLA=/path/to/LLaVA-VLA
    export AWARE_HF_HUB=$HF_HOME/hub
    export AWARE_RUNS=/path/to/where/eval/runs/are/written
    export LIBERO_CONFIG_PATH=$HOME/.libero
"""
import os

def _p(name: str, default: str) -> str:
    return os.environ.get(name, default)

OPENVLA_OFT = _p("AWARE_OPENVLA_OFT", "./third_party/openvla-oft")
OPENPI      = _p("AWARE_OPENPI",      "./third_party/openpi")
LIBERO      = _p("AWARE_LIBERO",      "./third_party/LIBERO")
SIMPLER     = _p("AWARE_SIMPLER",     "./third_party/SimplerEnv")
LLAVA_VLA   = _p("AWARE_LLAVA_VLA",   "./third_party/LLaVA-VLA")
HF_HUB      = _p("AWARE_HF_HUB", os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub"))
RUNS        = _p("AWARE_RUNS",        os.path.join(OPENVLA_OFT, "eval_runs"))

# The single implementation of the rule. Adapters that cannot import the package (because they run
# inside another repo's interpreter) load this file by path; everything else should `import aware`.
HOOKS_FILE  = _p("AWARE_HOOKS_FILE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "hooks.py"))

__all__ = ["OPENVLA_OFT", "OPENPI", "LIBERO", "SIMPLER", "LLAVA_VLA", "HF_HUB", "RUNS", "HOOKS_FILE"]
