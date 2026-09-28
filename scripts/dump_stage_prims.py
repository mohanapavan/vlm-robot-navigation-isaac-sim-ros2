"""Kit startup script: once the scene is loaded, write every prim's world-space bounding box to JSON.

This is step 1 of how `evaluation/hospital_ground_truth.json` was made (step 2: `scripts/build_ground_truth.py`). Run it inside
Isaac Sim, next to the scene launcher:

    scripts/launch_isaac.sh scene/slam.usd --exec "$PWD/scripts/dump_stage_prims.py" \\
        --/vlm_nav/prims_out=$HOME/hospital_prims.json

Settings: `--/vlm_nav/prims_out=<file>` (default ~/hospital_prims.json), `--/vlm_nav/prims_root=/World/hospital`.
The result for the hospital scene (5031 prims) is committed as `evaluation/hospital_prims.json.gz`.
"""
import asyncio
import json
import os
import time

import carb
import omni.kit.app
import omni.usd
from pxr import Usd, UsdGeom

MAX_DEPTH = 9
LOAD_TIMEOUT_S = 1200.0


async def _dump():
    settings = carb.settings.get_settings()
    out = settings.get('/vlm_nav/prims_out') or os.path.expanduser('~/hospital_prims.json')
    root = settings.get('/vlm_nav/prims_root') or '/World/hospital'
    app = omni.kit.app.get_app()
    ctx = omni.usd.get_context()
    started = time.monotonic()
    while time.monotonic() - started < LOAD_TIMEOUT_S:            # wait until the scene and its payloads are in
        await app.next_update_async()
        stage = ctx.get_stage()
        if stage is None:
            continue
        prim = stage.GetPrimAtPath(root)
        _, loaded, total = ctx.get_stage_loading_status()
        if prim and prim.IsValid() and prim.IsLoaded() and list(prim.GetChildren()) and total == 0:
            for _ in range(60):
                await app.next_update_async()
            break
    stage = ctx.get_stage()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render], useExtentsHint=False)
    rows = []
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        if not path.startswith(root) or prim.GetPath().pathElementCount > MAX_DEPTH or not prim.IsA(UsdGeom.Imageable):
            continue
        try:
            box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        except Exception:
            continue
        if box.IsEmpty():
            continue
        lo, hi = box.GetMin(), box.GetMax()
        rows.append({'path': path, 'type': prim.GetTypeName(), 'depth': prim.GetPath().pathElementCount,
                     'min': [lo[0], lo[1], lo[2]], 'max': [hi[0], hi[1], hi[2]]})
    with open(out, 'w') as f:
        json.dump(rows, f)
    carb.log_warn(f'vlm_nav dump_stage_prims: wrote {len(rows)} prims to {out}')


asyncio.ensure_future(_dump())
