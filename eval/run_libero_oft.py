"""Screen observation reinjection with an instruction-derived selection over the observation memory.

The reinjection operator itself is the one already in `aware/hooks.py`. Two things
change and both are switches on that file: the uncertainty threshold is removed (UAOR_TRIGGER=all),
and the retrieval is weighted by how strongly the language tokens themselves retrieve each
observation token (UAOR_SELECT=instruction). The completion checks, the eight-shard verification and
the paired recovered/lost accounting are reused from run_visual_feedback.cell; the visual-feedback
rule is left disabled, so the only active intervention is the reinjection.
"""
import argparse
import json
import os
from pathlib import Path

import run_visual_feedback as rvf
from run_omr_screen import OFT

rvf.ROOT = Path(OFT) / 'eval_runs/icra27/uaor_select_20260911'


def cell(tag, suite, tasks, trials, alpha, trigger, select, gamma, seed, anchor_rows='language',
         soft_power='1.0', soft_shuffle='0', extra=()):
    env = dict(UAOR_ALPHA=alpha, UAOR_TRIGGER=trigger, UAOR_SELECT=select,
               UAOR_GAMMA=gamma, UAOR_LOG=str(rvf.ROOT / tag), VFB_ANCHOR_ROWS=anchor_rows,
               UAOR_SOFT_POWER=soft_power, UAOR_SOFT_SHUFFLE=soft_shuffle)
    for kv in extra:
        k, v = kv.split('=', 1)
        if not k.startswith('UAOR_'):
            raise ValueError('extra switches must be UAOR_* variables')
        env[k] = v
    previous = os.environ.get('VFB_SEED_OVERRIDE')
    os.environ['VFB_SEED_OVERRIDE'] = str(seed)
    try:
        if select.startswith('attention'):
            # The attention selection reads the per-layer language-to-visual anchor published by the
            # visual_feedback provider, which writes nothing back; beta=1 only keeps the completion
            # check (all 32 layers seen) active.
            return rvf.cell(tag, 1, mode='anchor_provider', trials=trials, tasks=tasks, suite=suite,
                            extra_env=env)
        return rvf.cell(tag, 0, trials=trials, tasks=tasks, suite=suite, extra_env=env)
    finally:
        if previous is None:
            os.environ.pop('VFB_SEED_OVERRIDE', None)
        else:
            os.environ['VFB_SEED_OVERRIDE'] = previous


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--suite', default='libero_10')
    ap.add_argument('--tasks', default='0,6')
    ap.add_argument('--trials', type=int, default=50)
    ap.add_argument('--alpha', default='0.05')
    ap.add_argument('--trigger', default='all', choices=('entropy', 'confident', 'soft', 'all', 'random', 'none'))
    ap.add_argument('--select', default='none',   # the paper's operator is the plain lookup of Eq. 9;
                    # 'instruction' reweights the observation tokens and is a variant, not the default

                    choices=('none', 'instruction', 'instruction_inverted',
                             'instruction_shuffled', 'instruction_layerwise',
                             'attention', 'attention_inverted', 'attention_shuffled'))
    ap.add_argument('--gamma', default='0.80')
    ap.add_argument('--anchor-rows', default='language', choices=('language', 'instruction'),
                    help='language rows feeding the attention anchor: every prompt row, or only the task text')
    ap.add_argument('--soft-power', default='1.0')
    ap.add_argument('--soft-shuffle', default='0', choices=('0', '1'))
    ap.add_argument('--extra', action='append', default=[], help='additional UAOR_* env switch, KEY=VAL')
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--tag', required=True)
    a = ap.parse_args()
    tasks = tuple(int(x) for x in a.tasks.split(','))
    rvf.ROOT.mkdir(parents=True, exist_ok=True)
    print(json.dumps(cell(a.tag, a.suite, tasks, a.trials, a.alpha,
                          a.trigger, a.select, a.gamma, a.seed, a.anchor_rows,
                          a.soft_power, a.soft_shuffle, tuple(a.extra)), indent=2))
