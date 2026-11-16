"""Entry point. Only the explicit 'train' command can load training code."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ('diat-capacity', 'diat-signal', 'diat-stability', 'diat-reference', 'shd')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def jobs(experiment, seed=None):
    seeds = [42, 123, 999]
    result = []
    def add(model, width=128, view='original', steps=64, coupling=None, seeds=seeds):
        for s in seeds:
            result.append(dict(model=model, width=width, view=view, steps=steps, coupling=coupling, seed=s))
    if experiment == 'diat-capacity':
        for w in (64, 128, 256):
            for model in ('doppler', 'bank_library', 'bank_stable'):
                add(model, w)
        for model in ('doppler', 'bank_stable'):
            add(model, seeds=[7, 2024])
    elif experiment == 'diat-signal':
        for view in ('original', 'difference'):
            for model in ('vanilla', 'parametric', 'doppler'):
                add(model, view=view)
    elif experiment == 'diat-stability':
        for steps in (16, 64):
            for coupling in (0.20, 0.28, 0.32, None):
                add('ei', steps=steps, coupling=coupling)
    elif experiment == 'diat-reference':
        result = read(ROOT / 'reference/experiment_plan.json')['jobs']
    elif experiment == 'shd':
        c = read(ROOT / 'configs/shd.json')
        result = [dict(model=m, condition=cell['name'], seed=s)
                  for s in c['seeds'] for m in c['models'] for cell in c['conditions']]
    else:
        raise ValueError(experiment)
    return [x for x in result if seed is None or x['seed'] == seed]


def data_checks(args):
    if args.experiment.startswith('diat'):
        expected = read(ROOT / 'configs/diat.json')['raw_data_sha256']
        paths = {name: args.data / name for name in expected}
    else:
        expected = read(ROOT / 'configs/shd.json')['raw_sha256']
        paths = {name: args.data / f'shd_{name}.h5' for name in expected}
    for name, path in paths.items():
        if not path.is_file() or digest(path) != expected[name]:
            raise ValueError(f'Missing or different input: {path}')


def main():
    p = argparse.ArgumentParser(description='NEUNET experiments: raw numeric results and checkpoints only.')
    p.add_argument('command', choices=['plan', 'check', 'train'])
    p.add_argument('--experiment', choices=EXPERIMENTS, required=True)
    p.add_argument('--seed', type=int)
    p.add_argument('--data', type=Path, help='DIAT folder with X.npy/y.npy, or SHD folder with shd_train.h5/shd_test.h5')
    p.add_argument('--output', type=Path, help='A new, dedicated output directory')
    p.add_argument('--sources', type=Path, default=ROOT / '.sources')
    p.add_argument('--resume', action='store_true', help='Continue this repository\'s own output only')
    args = p.parse_args()
    from runtime import verify_package
    verify_package()
    selected = jobs(args.experiment, args.seed)
    if not selected:
        p.error('Seed is not part of this experiment.')
    if args.command == 'plan':
        print(json.dumps(dict(experiment=args.experiment, count=len(selected), jobs=selected), indent=2))
        return
    if args.data is None:
        p.error('--data is required for check/train.')
    args.data = args.data.resolve()
    data_checks(args)
    if args.command == 'check':
        print(json.dumps(dict(status='INPUT_HASHES_VERIFIED', jobs=len(selected), training_started=False)))
        return
    if args.seed is None or args.output is None:
        p.error('train requires --seed and --output.')
    args.output = args.output.resolve()
    if args.output == args.data or args.output in args.data.parents or args.data in args.output.parents:
        p.error('Input and output directories must be separate.')
    marker = args.output / 'pipeline.json'
    identity = dict(experiment=args.experiment, seed=args.seed, data=str(args.data),
                    pipeline_sha256=digest(ROOT / 'manifest.json'))
    if args.output.exists():
        if not args.resume or not marker.is_file() or read(marker) != identity:
            p.error('Output exists. Use a new directory, or --resume with this pipeline\'s matching output.')
    elif args.resume:
        p.error('Cannot resume a missing output.')
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        marker.write_text(json.dumps(identity, indent=2), encoding='utf-8')
    # Single-writer lock; released automatically on exit. No action is performed on old campaigns.
    import fcntl
    with (args.output / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.experiment == 'shd':
            from shd import run
        elif args.experiment == 'diat-reference':
            from reference.run_frozen_reference import run_cli as run
        else:
            from diat import run
        run(args, selected)


if __name__ == '__main__':
    main()
