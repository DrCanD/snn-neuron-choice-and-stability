"""Entry point. Only the explicit 'train' command can load training code."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parent
STUDY = json.loads((ROOT / 'configs/study.json').read_text(encoding='utf-8'))
EXPERIMENTS = ('diat-capacity', 'diat-signal', 'diat-stability', 'diat-reference', 'shd') + tuple(STUDY)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def jobs(experiment, seed=None):
    if experiment in STUDY:
        if seed is not None:
            raise ValueError('Original study stages retain their complete method/seed order; omit --seed.')
        return [STUDY[experiment]]
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
    kind = STUDY[args.experiment]['input'] if args.experiment in STUDY else ('diat' if args.experiment.startswith('diat') else 'shd')
    verified = {}
    if kind == 'mixed':
        if not args.data.is_file():
            raise ValueError('Provide the archived shd_T12_m200_v2.pt file as --data.')
        actual = digest(args.data)
        if args.data_sha256 is not None and actual != args.data_sha256.lower():
            raise ValueError('Mixed-SHD cache differs from --data-sha256.')
        return {str(args.data): actual}
    if kind in ('diat', 'both'):
        expected = read(ROOT / 'configs/diat.json')['raw_data_sha256']
        paths = {name: args.data / name for name in expected}
        for name, path in paths.items():
            if not path.is_file() or digest(path) != expected[name]:
                raise ValueError(f'Missing or different input: {path}')
            verified[str(path)] = expected[name]
    if kind in ('shd', 'both'):
        folder = args.shd_data if kind == 'both' else args.data
        if folder is None:
            raise ValueError('This stage requires --data for DIAT and --shd-data for SHD.')
        expected = read(ROOT / 'configs/shd.json')['raw_sha256']
        for name in expected:
            path = folder / f'shd_{name}.h5'
            if not path.is_file() or digest(path) != expected[name]:
                raise ValueError(f'Missing or different input: {path}')
            verified[str(path)] = expected[name]
    return verified


def main():
    p = argparse.ArgumentParser(description='NEUNET experiments: raw numeric results and checkpoints only.')
    p.add_argument('command', choices=['plan', 'check', 'train'])
    p.add_argument('--experiment', choices=EXPERIMENTS, required=True)
    p.add_argument('--seed', type=int)
    p.add_argument('--data', type=Path, help='DIAT folder with X.npy/y.npy, or SHD folder with shd_train.h5/shd_test.h5')
    p.add_argument('--output', type=Path, help='A new, dedicated output directory')
    p.add_argument('--shd-data', type=Path, help='SHD folder for a stage that also requires DIAT')
    p.add_argument('--data-sha256', help='Trusted SHA-256 of the historical mixed-SHD cache')
    p.add_argument('--resume', action='store_true', help='Continue this repository\'s own output only')
    args = p.parse_args()
    args.sources = None  # All model sources are bundled and verified locally.
    from runtime import verify_package
    verify_package()
    try:
        selected = jobs(args.experiment, args.seed)
    except ValueError as e:
        p.error(str(e))
    if not selected:
        p.error('Seed is not part of this experiment.')
    if args.command == 'plan':
        print(json.dumps(dict(experiment=args.experiment, unit='complete_stage' if args.experiment in STUDY else 'training_job', count=len(selected), jobs=selected), indent=2))
        return
    if args.data is None:
        p.error('--data is required for check/train.')
    args.data = args.data.resolve()
    if args.shd_data is not None:
        args.shd_data = args.shd_data.resolve()
    try:
        inputs = data_checks(args)
    except ValueError as e:
        p.error(str(e))
    if args.command == 'check':
        mixed = args.experiment in STUDY and STUDY[args.experiment]['input'] == 'mixed'
        print(json.dumps(dict(status='CACHE_FINGERPRINTED_HISTORICAL_IDENTITY_UNVERIFIED' if mixed else 'INPUT_HASHES_VERIFIED', input_sha256=inputs, jobs=len(selected), training_started=False)))
        return
    legacy = args.experiment in STUDY
    if args.output is None or (not legacy and args.seed is None):
        p.error('train requires --output; revision controls also require --seed.')
    if legacy and args.resume:
        p.error('Original study stages require a fresh directory; exact legacy checkpoint continuation is not supported.')
    if legacy and STUDY[args.experiment]['input'] == 'mixed' and not args.data_sha256:
        p.error('Historical mixed-SHD training requires --data-sha256 from your trusted cache archive.')
    args.output = args.output.resolve()
    for data in (args.data, args.shd_data):
        if data is not None and (args.output == data or args.output in data.parents or data in args.output.parents):
            p.error('Input and output directories must be separate.')
    marker = args.output / 'pipeline.json'
    identity = dict(experiment=args.experiment, seed=args.seed, data=str(args.data),
                    input_sha256=inputs, pipeline_sha256=digest(ROOT / 'manifest.json'))
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
        if legacy:
            from study import run
        elif args.experiment == 'shd':
            from shd import run
        elif args.experiment == 'diat-reference':
            from reference.run_frozen_reference import run_cli as run
        else:
            from diat import run
        run(args, selected)


if __name__ == '__main__':
    main()
