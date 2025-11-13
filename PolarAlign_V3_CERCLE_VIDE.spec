# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['polar_align.py'],
    pathex=[],
    binaries=[],
    datas=[('templates', 'templates'), ('config.ini', '.')],
    hiddenimports=['astropy.time', 'astropy.coordinates', 'astropy.units', 'flask'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='PolarAlign_V3_CERCLE_VIDE',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
