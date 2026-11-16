"""Pinned sources and environment checks shared by the experiment runners."""
import ast
import hashlib
import importlib.util
import json
import platform
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


def sources(cache=None):
    """Load bundled sources without network access or installation."""
    verify_package()
    repo = ROOT / 'sources'
    lib = repo / 'dikmen_neurons'
    for name, expected in read(repo / 'provenance.json')['packaged_sha256'].items():
        if sha(repo / name) != expected:
            raise RuntimeError(f'Packaged source differs: {name}')
    for name in ('dikmen_neurons', 'mstask'):
        existing = sys.modules.get(name)
        if existing is not None:
            path = Path(getattr(existing, '__file__', '') or '.')
            if not path.resolve().is_relative_to(repo.resolve()):
                raise RuntimeError(f'Foreign {name} loaded; use a fresh process.')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(lib.parent))
    # Load pinned core without the upstream __init__ installer side effect.
    package = types.ModuleType('mstask')
    package.__file__ = str(repo / 'src/mstask/__init__.py')
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
