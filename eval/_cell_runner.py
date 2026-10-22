"""Run one fixed visual-feedback diagnostic through the official evaluator.

This runner performs no dose/layer search and no automatic promotion. Modes
separate minimal nonopposing feedback, all action-source removal, and a
same-row-magnitude direction mismatch. Only a completed intervention probe may
precede a scheduled rollout; the caller checks that evidence before invoking.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time

from run_omr_screen import active_evals, episodes, OFT, W

ROOT = OFT / 'eval_runs/icra27/vfb_20260909'


def cell(tag, beta, mode='nonopposing', trials=50, tasks=(0, 6), suite='libero_10', pool='standard',
         extra_env=None):
    """extra_env carries switches for operators other than the visual-feedback rule. It is recorded in
    settings.json so a reused directory still refuses to mix configurations, and it defaults to
    nothing, which reproduces every completed run exactly."""
    path = ROOT / tag
    if (path / 'result.json').exists():
        return json.loads((path / 'result.json').read_text())
    if active_evals():
        raise RuntimeError(f'Another evaluation is active: {active_evals()}')
    reuse = path.exists()
    path.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(('RSI_', 'UAOR_', 'DRO_', 'OMR_', 'IOR_', 'VFB_', 'RDPROBE',
                           'STALL_', 'OBS_DEGRADE', 'OFT_RANDOM_CROP', 'TEMP_ENS')):
            env.pop(key)
    env.update(SUITE=suite, ARM='base', SEED=os.environ.get('VFB_SEED_OVERRIDE', '7'), NGPU='8', TRIALS=str(trials),
               OUT=str(path), ONLY_TASKS=','.join(map(str, tasks)),
               LIBERO_INIT_POOL=pool, LIBERO_CONFIG_PATH=os.path.expanduser(os.environ.get('LIBERO_CONFIG_PATH','~/.libero')),
               OFT_SKIP_SYNC='1', TF_FORCE_GPU_ALLOW_GROWTH='true',
               OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
               NUMEXPR_NUM_THREADS='4', VFB_BETA=str(beta), VFB_MODE=mode)
    env.update({k: str(v) for k, v in (extra_env or {}).items()})
    settings = {k: env[k] for k in ('SUITE', 'ARM', 'SEED', 'NGPU', 'TRIALS',
                'ONLY_TASKS', 'VFB_BETA', 'VFB_MODE', 'LIBERO_INIT_POOL')}
    settings['extra_env'] = {k: str(v) for k, v in sorted((extra_env or {}).items())}
    rc = 0
    if reuse:
        if json.loads((path / 'settings.json').read_text()) != settings:
            raise RuntimeError('Existing run has different settings')
    else:
        (path / 'settings.json').write_text(json.dumps(settings, indent=2))
        print(time.strftime('%Y-%m-%d %H:%M:%S'), 'START', tag, flush=True)
        with (path / 'launcher.log').open('w') as log:
            rc = subprocess.call(['bash', str(OFT / 'experiments/robot/libero/run_parallel_libero_eval.sh')],
                                 env=env, stdout=log, stderr=subprocess.STDOUT)
    rows, nfiles = episodes(path)
    expected = {(t, e) for t in tasks for e in range(trials)}
    shards = {(t * trials + e) % 8 for t, e in expected}
    if rc or set(rows) != expected or nfiles != len(shards):
        raise RuntimeError(f'Incomplete {tag}: rc={rc}, files={nfiles}, episodes={len(rows)}')
    for shard in sorted(shards):
        summary = json.loads((path / f'vfb_summary_shard{shard}.json').read_text())
        if beta and (summary['active_layers'] != list(range(32)) or summary['forward_count'] == 0):
            raise RuntimeError(f'Inactive visual feedback intervention in shard {shard}')
        log = (path / f'shard{shard}.log').read_text()
        if 'Episode error:' in log or 'Final results:' not in log:
            raise RuntimeError(f'Internal evaluator exception in shard {shard}')
        for match in re.finditer('Traceback \\(most recent call last\\)', log):
            preceding = log[:match.start()].rstrip().splitlines()[-1]
            if not re.match(r'Exception ignored in: <function (MjRenderContext|EGLGLContext)\.__del__', preceding):
                raise RuntimeError(f'Unexpected traceback in shard {shard}')
    result = dict(success=sum(rows.values()), episodes=len(rows), beta=beta, mode=mode,
                  extra_env=settings['extra_env'],
                  per_task={str(t): sum(rows[t, e] for e in range(trials)) for t in tasks})
    if pool == 'standard':
        baseline, _ = episodes(OFT / f'eval_runs/icra27/{suite}_base_s7')
        if expected <= baseline.keys():
            result.update(baseline_success=sum(baseline[k] for k in rows),
                          recovered=sum(rows[k] and not baseline[k] for k in rows),
                          lost=sum(baseline[k] and not rows[k] for k in rows))
    (path / 'result.json').write_text(json.dumps(result, indent=2))
    print(time.strftime('%Y-%m-%d %H:%M:%S'), 'DONE', tag, result, flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('nonopposing', 'remove_action', 'shuffle_direction', 'orthogonal_control'), default='nonopposing')
    parser.add_argument('--tasks', default='0,6')
    parser.add_argument('--trials', type=int, default=50)
    parser.add_argument('--suite', default='libero_10')
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    tasks = tuple(int(x) for x in args.tasks.split(','))
    if not tasks or len(set(tasks)) != len(tasks) or not all(0 <= t < 10 for t in tasks):
        raise ValueError('Choose distinct benchmark task IDs from 0 to 9')
    if not 1 <= args.trials <= 50:
        raise ValueError('Standard evaluation supports up to 50 initial states')
    ROOT.mkdir(parents=True, exist_ok=True)
    cell(args.tag, 1, mode=args.mode, trials=args.trials, tasks=tasks, suite=args.suite)
