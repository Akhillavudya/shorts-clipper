"""Serving the built SPA (WEB_UI_PLAN.md section 6).

``frontend/`` builds into ``dist/`` *inside this package*, and that bundle is
committed. The reason is setup cost: the deliverable is "it works after ``uv
sync``", and requiring Node to *run* a Python CLI is a real regression against
that. Node is needed to change the frontend, never to use it.

Three rules this module exists to keep:

**The catch-all is registered last, and it must be.** It answers every path
that is not already a route with ``index.html``, because the SPA uses real
URLs rather than a hash router - a deep link typed into the address bar is a
genuine request to Flask before the router ever sees it. Registered before the
API, that same rule would swallow ``/api/jobs``. ``review/app.py`` calls
:func:`register` after the blueprint for exactly this reason, and
``tests/test_webui.py`` asserts it.

**A missing bundle is an instruction, not a traceback.** A fresh clone whose
``dist/`` has not been built, or a working tree mid-rebase, should say ``cd
frontend && npm install && npm run build`` in the browser. A 404 there reads as
a broken install; a stack trace reads as a broken program.

**No caching.** The bundle is built with fixed filenames rather than
content-hashed ones (see ``vite.config.ts``), so there is nothing to bust a
cache with. ``max_age=0`` is what keeps a rebuild visible on refresh instead of
producing a bug report about a change that "did not apply".

Flask is imported inside :func:`register` rather than at module scope, so
``clipper moments`` can import this package - it will, transitively - without
paying for a web framework.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

#: The built bundle. Populated by ``npm run build`` in ``frontend/``.
DIST: Path = Path(__file__).resolve().parent / "dist"

#: Prefixes the catch-all must never answer. Every one of them is a real route
#: on the same app, and the catch-all's job is to be the thing that runs when
#: none of them matched - Flask's own matcher handles that for registered
#: routes, so this list only guards the *404* case: a mistyped ``/api/jbos``
#: should be a JSON 404 from the API's error handler, not an HTML page the SPA
#: then fails to parse.
RESERVED_PREFIXES: tuple[str, ...] = ("api/", "media/", "healthz", "legacy/")

_MISSING_PAGE = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>clipper - the UI is not built</title>
    <style>
      :root {{ color-scheme: light dark; }}
      body {{
        margin: 0; padding: 3rem 1rem; line-height: 1.55;
        font: 16px/1.55 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
      }}
      main {{ max-width: 44rem; margin: 0 auto; }}
      code, pre {{ font-family: ui-monospace, "Cascadia Mono", Consolas, monospace; }}
      pre {{ padding: 0.9rem 1rem; border-radius: 10px; background: #8881; overflow-x: auto; }}
      p {{ margin: 0.8rem 0; }}
    </style>
  </head>
  <body>
    <main>
      <h1>The web UI has not been built</h1>
      <p>
        <code>{dist}</code> is empty or missing, so there is no bundle to serve.
        It is normally committed; build it with:
      </p>
      <pre>cd frontend
npm install
npm run build</pre>
      <p>
        Everything else on this server is still up:
        <a href="/legacy/">the review pages</a>,
        <a href="/api/jobs">/api/jobs</a> and
        <a href="/api/queue">/api/queue</a>.
      </p>
    </main>
  </body>
</html>
"""


def is_built(dist: Path | str | None = None) -> bool:
    """Is there a bundle to serve?

    The test is ``index.html``, not the directory: ``--emptyOutDir`` means a
    build that failed halfway can leave the directory present and empty, and a
    present-but-empty ``dist/`` should produce the instructions page rather
    than a 404 on every asset.
    """
    return (Path(dist if dist is not None else DIST) / "index.html").is_file()


def register(app: Any, *, dist: Path | str | None = None) -> None:
    """Mount the SPA on ``app``. **Call this after every other route.**

    Two rules, one endpoint each:

    ``/assets/<path>`` serves the built JS and CSS, guarded against traversal
    the same way the media route is - resolve, then check containment. Nothing
    reaches it from a browser except the two filenames ``index.html`` names,
    but the guard costs one comparison and the alternative is a directory
    escape on a server whose whole job is to read files off this disk.

    ``/<path:...>`` returns ``index.html`` for anything left, which is what
    makes ``/jobs/abc123`` work when it is typed rather than navigated to.
    """
    from flask import abort, send_file

    # ``DIST`` is read here rather than bound as a default argument, so that a
    # test can point the whole thing at a fixture directory by setting the
    # module attribute - which is also how ``create_app`` stays a one-liner
    # instead of growing a ``dist=`` parameter it would only ever pass on.
    root = Path(dist if dist is not None else DIST).resolve()

    @app.get("/assets/<path:relpath>")
    def spa_assets(relpath: str):
        target = (root / "assets" / relpath).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            abort(404, description=f"{relpath} is not part of the built UI")
        return send_file(target, max_age=0)

    @app.get("/", defaults={"relpath": ""})
    @app.get("/<path:relpath>")
    def spa_index(relpath: str):
        # The reserved prefixes only get here when nothing matched them, which
        # means the client asked for a route that does not exist. Answering
        # with the SPA's HTML would hand a fetch() a page to choke on, so it
        # falls through to the app's 404 handler - which returns JSON for
        # these prefixes.
        if relpath.startswith(RESERVED_PREFIXES):
            abort(404, description=f"/{relpath} is not a route on this server")

        index = root / "index.html"
        if not index.is_file():
            return (
                _MISSING_PAGE.format(dist=root),
                200,
                {"Content-Type": "text/html; charset=utf-8"},
            )

        return send_file(index, max_age=0)


__all__ = ["DIST", "RESERVED_PREFIXES", "is_built", "register"]
