"""Pinned sources and environment checks shared by the experiment runners."""
import ast
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_package():
    for name, expected in read(ROOT / 'manifest.json')['files'].items():
        if sha(ROOT / name) != expected:
            raise RuntimeError(f'Pipeline source changed: {name}')


def sources(cache):
    verify_package()
    config = read(ROOT / 'configs/diat.json')
    cache = Path(cache).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    for name, url, commit in config['repositories']:
        path = cache / name
        if not path.exists():
            subprocess.run(['git', 'clone', '--quiet', url, str(path)], check=True)
            subprocess.run(['git', '-C', str(path), 'checkout', '--quiet', commit], check=True)
        head = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
        if head != commit or subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain'], text=True).strip():
            raise RuntimeError(f'Source checkout differs: {path}')
    repo = cache / 'repo'
    candidates = list((cache / 'neurons').glob('**/dikmen_neurons/__init__.py'))
    if len(candidates) != 1:
        raise RuntimeError('Ambiguous neuron package')
    lib = candidates[0].parent
    for name, expected in config['source_hashes'].items():
        path = lib / name.removeprefix('dikmen_neurons/') if name.startswith('dikmen_neurons/') else repo / name
        if sha(path) != expected:
            raise RuntimeError(f'Source hash differs: {name}')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(lib.parent))
    # Load pinned core without the upstream __init__ installer side effect.
    package = types.ModuleType('mstask')
    package.__path__ = [str(repo / 'src/mstask')]
    sys.modules['mstask'] = package
    spec = importlib.util.spec_from_file_location('mstask.core', repo / 'src/mstask/core.py')
    core = importlib.util.module_from_spec(spec)
    sys.modules['mstask.core'] = core
    spec.loader.exec_module(core)
    package.core = core
    return repo, lib, core


def definition(path, name):
    source = Path(path).read_text(encoding='utf-8')
    lines = source.splitlines(keepends=True)
    node = next(n for n in ast.parse(source).body if getattr(n, 'name', None) == name)
    start = min([d.lineno for d in getattr(node, 'decorator_list', [])] + [node.lineno])
    return ''.join(lines[start - 1:node.end_lineno])


def environment(expected, flags):
    import torch
    import numpy as np
    import scipy
    import sklearn
    import h5py
    if not torch.cuda.is_available():
        raise RuntimeError('Recorded experiments require CUDA; no training started.')
    actual = dict(python=platform.python_version(), numpy=np.__version__, torch=str(torch.__version__),
                  cuda_runtime=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                  gpu=torch.cuda.get_device_name(0), scikit_learn=sklearn.__version__,
                  scipy=scipy.__version__, h5py=h5py.__version__)
    if any(actual[k] != v for k, v in expected.items()):
        raise RuntimeError(f'Environment differs from recorded protocol: {actual}')
    torch.backends.cuda.matmul.allow_tf32 = flags['matmul_tf32']
    torch.backends.cudnn.allow_tf32 = flags['cudnn_tf32']
    torch.backends.cudnn.benchmark = flags['cudnn_benchmark']
    torch.use_deterministic_algorithms(flags['deterministic_algorithms'])
    return actual
