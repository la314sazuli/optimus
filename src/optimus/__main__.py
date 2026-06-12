"""``python -m optimus`` / ``optimus``: the single entrypoint.

In the default ``OPTIMUS_MODE=simple`` this composes and runs the whole bot in
one process (see :mod:`optimus.app.simple`). In ``OPTIMUS_MODE=distributed`` there
is no single process to run — each service is its own entrypoint
(``python -m optimus.services.<name>``) — so this prints that guidance and exits
non-zero rather than pretending to start something.
"""

from __future__ import annotations

import asyncio
import sys

from optimus.core.config import get_settings

_DISTRIBUTED_HELP = (
    "OPTIMUS_MODE=distributed has no single-process entrypoint. Run each service "
    "on its own:\n"
    "  python -m optimus.services.gateway\n"
    "  python -m optimus.services.ingest\n"
    "  python -m optimus.services.detection\n"
    "  python -m optimus.services.moderation\n"
    "  python -m optimus.services.interactions\n"
    "  python -m optimus.services.scheduler\n"
    "Or unset OPTIMUS_MODE to run everything in one process (simple mode)."
)


def main() -> None:
    """Console entrypoint: run simple mode, or point distributed at its services."""
    settings = get_settings()
    if not settings.is_simple_mode:
        print(_DISTRIBUTED_HELP, file=sys.stderr)
        raise SystemExit(2)

    from optimus.app.simple import run_simple
    from optimus.app.startup import StartupError

    try:
        asyncio.run(run_simple())
    except StartupError as exc:
        # A misconfiguration we caught on purpose: show the one-line fix, not a
        # traceback that buries it.
        print(f"Optimus could not start: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        raise SystemExit(0) from None


if __name__ == "__main__":
    main()
