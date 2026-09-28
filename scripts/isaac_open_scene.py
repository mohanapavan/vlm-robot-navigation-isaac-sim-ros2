"""Kit startup script: open the scene, wait for its (streamed) assets, then press Play.

Not run directly: `scripts/launch_isaac.sh` passes it to Isaac Sim with `--exec`. Settings come from
carb settings (`--/vlm_nav/scene=<usd>`, `--/vlm_nav/play=false` to only open the scene,
`--/vlm_nav/remove=/World/A,/World/B` to delete prims from the loaded stage).
"""
import asyncio
import time

import carb
import omni.kit.app
import omni.timeline
import omni.usd

STARTUP_FRAMES = 120      # let the application finish starting its extensions first
SETTLE_FRAMES = 30        # consecutive frames with no pending asset loads before the scene counts as loaded
LOAD_TIMEOUT_S = 900.0    # the hospital scene streams a few hundred MB on the first run


async def _open_and_play():
    settings = carb.settings.get_settings()
    scene = settings.get('/vlm_nav/scene')
    play = settings.get('/vlm_nav/play')
    play = True if play is None else bool(play)
    if not scene:
        carb.log_error('vlm_nav: pass --/vlm_nav/scene=<path to .usd>')
        return
    app = omni.kit.app.get_app()
    for _ in range(STARTUP_FRAMES):
        await app.next_update_async()

    ctx = omni.usd.get_context()
    carb.log_warn(f'vlm_nav: opening {scene}')
    ok, err = await ctx.open_stage_async(scene)
    if not ok:
        carb.log_error(f'vlm_nav: could not open {scene}: {err}')
        return

    idle, started = 0, time.monotonic()
    while idle < SETTLE_FRAMES:
        await app.next_update_async()
        _, loaded, total = ctx.get_stage_loading_status()
        idle = idle + 1 if total == 0 or loaded >= total else 0
        if time.monotonic() - started > LOAD_TIMEOUT_S:
            carb.log_error('vlm_nav: timed out waiting for scene assets; playing anyway')
            break
    carb.log_warn('vlm_nav: scene loaded')

    # Prims to delete from the loaded stage (the file on disk is never touched), e.g. --/vlm_nav/remove=/World/TestObstacles
    remove = settings.get('/vlm_nav/remove')
    names = [remove] if isinstance(remove, str) else list(remove or [])
    for path in filter(None, (p.strip() for item in names for p in item.split(','))):
        if ctx.get_stage().GetPrimAtPath(path):
            ctx.get_stage().RemovePrim(path)
            carb.log_warn(f'vlm_nav: removed {path}')

    if play:
        timeline = omni.timeline.get_timeline_interface()
        timeline.stop()
        await app.next_update_async()
        timeline.play()
        carb.log_warn('vlm_nav: timeline playing (ROS topics are live)')


asyncio.ensure_future(_open_and_play())
