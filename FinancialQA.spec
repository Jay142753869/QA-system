# -*- mode: python ; coding: utf-8 -*-

import importlib.util
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, copy_metadata


datas = [
    ("templates", "templates"),
    ("static", "static"),
    ("models/RE-GCN-master/data/80STOCKS", "models/RE-GCN-master/data/80STOCKS"),
    ("models/RE-GCN-master/models", "models/RE-GCN-master/models"),
    ("models/RE-GCN-master/rgcn", "models/RE-GCN-master/rgcn"),
    ("models/RE-GCN-master/src", "models/RE-GCN-master/src"),
    ("models/TiRGN-main/data/80STOCKS", "models/TiRGN-main/data/80STOCKS"),
    ("models/TiRGN-main/models", "models/TiRGN-main/models"),
    ("models/TiRGN-main/rgcn", "models/TiRGN-main/rgcn"),
    ("models/TiRGN-main/src", "models/TiRGN-main/src"),
]

hiddenimports = [
    "jieba.finalseg",
    "scipy",
    "scipy.sparse",
    "scipy.sparse.csgraph",
    "transformers.models.bert.configuration_bert",
    "transformers.models.bert.modeling_bert",
    "transformers.models.bert.tokenization_bert",
    "transformers.models.bert.tokenization_bert_fast",
    "dgl.backend.pytorch",
    "dgl.function",
    "dgl.heterograph",
    "dgl.convert",
    "dgl.ops",
    "dgl.nn",
    "dgl.nn.pytorch",
]

for package_name in ("transformers", "tokenizers", "huggingface_hub", "safetensors"):
    try:
        datas += collect_data_files(package_name)
    except Exception:
        pass
    try:
        datas += copy_metadata(package_name)
    except Exception:
        pass

dgl_spec = importlib.util.find_spec("dgl")
if dgl_spec and dgl_spec.submodule_search_locations:
    dgl_dir = Path(next(iter(dgl_spec.submodule_search_locations)))
    for dll_path in [
        dgl_dir / "dgl.dll",
        dgl_dir / "dgl_sparse" / "dgl_sparse_pytorch_2.0.0.dll",
        dgl_dir / "dgl_sparse" / "dgl_sparse_pytorch_2.0.1.dll",
        dgl_dir / "tensoradapter" / "pytorch" / "tensoradapter_pytorch_2.0.0.dll",
        dgl_dir / "tensoradapter" / "pytorch" / "tensoradapter_pytorch_2.0.1.dll",
    ]:
        if dll_path.exists():
            relative_parent = dll_path.parent.relative_to(dgl_dir.parent)
            datas.append((str(dll_path), str(relative_parent)))


a = Analysis(
    ["gui_launcher.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "matplotlib",
        "notebook",
        "IPython",
        "pytest",
        "tkinter",
        "torchaudio",
        "torchaudio.backend",
        "torchaudio.datasets",
        "torchaudio.functional",
        "torchaudio.models",
        "torchaudio.pipelines",
        "torchaudio.sox_effects",
        "torchaudio.transforms",
        "torchvision",
        "torchvision.datasets",
        "torchvision.io",
        "torchvision.models",
        "torchvision.ops",
        "torchvision.transforms",
        "torchvision.utils",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="FinancialQA",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="FinancialQA",
)
