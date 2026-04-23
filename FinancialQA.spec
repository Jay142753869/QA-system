# -*- mode: python ; coding: utf-8 -*-
import sys
sys.setrecursionlimit(5000)

from PyInstaller.utils.hooks import collect_all


datas = [
    ('templates', 'templates'),
    ('static', 'static'),
    ('models/RE-GCN-master/src', 'models/RE-GCN-master/src'),
    ('models/RE-GCN-master/rgcn', 'models/RE-GCN-master/rgcn'),
    ('models/RE-GCN-master/data/80STOCKS', 'models/RE-GCN-master/data/80STOCKS'),
    (
        'models/RE-GCN-master/models/80STOCKS-uvrgcn-convtranse-ly2-dilate1-his3-weight_0.5-discount_1.0-angle_10-dp0.2_0.2_0.2_0.2-gpu0',
        'models/RE-GCN-master/models',
    ),
    ('models/TiRGN-main/src', 'models/TiRGN-main/src'),
    ('models/TiRGN-main/rgcn', 'models/TiRGN-main/rgcn'),
    ('models/TiRGN-main/data/80STOCKS', 'models/TiRGN-main/data/80STOCKS'),
    (
        'models/TiRGN-main/models/gl_rate_0.3-80STOCKS-convgcn-timeconvtranse-ly2-dilate1-his9-weight_0.5-discount_1.0-angle_14-dp0.2_0.2_0.2_0.2-gpu0-checkpoint',
        'models/TiRGN-main/models',
    ),
]
binaries = []
hiddenimports = [
    'dgl', 'torch', 'transformers', 'jieba', 'ahocorasick', 'pandas', 
    'sklearn', 'scipy', 'networkx', 'threading', 'collections', 'hashlib',
    'logging', 'csv', 'json', 're', 'os', 'sys', 'time'
]

tmp_ret = collect_all('dgl')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('transformers')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('jieba')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('webview')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

a = Analysis(
    ['gui_launcher.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torchvision.datasets', 'torchvision.io', 'torchvision.models', 'torchvision.ops', 'torchvision.transforms', 'torchvision.utils', 'torchaudio'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='FinancialQA',
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
    name='FinancialQA',
)
