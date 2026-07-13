"""Postgres row-level-security isolation test for multi-tenant mode.

Proves the ``optimus.guild_id`` GUC wiring
(:func:`optimus.db.engine.session_scope` / :func:`create_session_scope`) makes
Postgres RLS (migration ``0002``) actually enforce tenant isolation:

* with the GUC set to guild A, guild B's rows are invisible and a cross-tenant
  insert is rejected; and
* the *pre-fix* behaviour — an RLS-subject role with **no** GUC set — returns
  zero rows, which is exactly why the control was inert/broken before this
  change; setting the GUC (the fix) corrects it.

RLS is a Postgres feature, so this suite needs a real Postgres reachable via
``OPTIMUS_TEST_POSTGRES_URL`` (an async SQLAlchemy URL, e.g.
``postgresql+asyncpg://user:pass@localhost:5432/optimus``). It is skipped when
that variable is unset — including the default SQLite-only CI job, which has no
RLS to exercise. See ``docs/security-audit.md``.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy import make_url, text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from optimus.app.migrate import run_migrations
from optimus.db.engine import SessionScope, create_session_factory, create_session_scope
from optimus.db.models import Detection

PG_URL = os.environ.get("OPTIMUS_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    PG_URL is None,
    reason="OPTIMUS_TEST_POSTGRES_URL not set; RLS enforcement requires Postgres",
)

GUILD_A = 111
GUILD_B = 222

# A dedicated NOSUPERUSER/NOBYPASSRLS login role: RLS (even FORCE RLS) is always
# bypassed by superusers and BYPASSRLS roles, so the assertions only mean
# something when run as a role that is actually subject to the policy.
_PROBE_ROLE = "optimus_rls_probe"
_PROBE_PASSWORD = "optimus_rls_probe"


async def _current_role_bypasses_rls(engine: AsyncEngine) -> bool:
    async with engine.connect() as conn:
        row = (
            await conn.execute(
                text("SELECT rolsuper OR rolbypassrls FROM pg_roles WHERE rolname = current_user")
            )
        ).scalar_one()
    return bool(row)


async def _make_rls_subject_url(admin: AsyncEngine, admin_url: str) -> str:
    """Return a URL for a role guaranteed to be subject to RLS.

    If the admin role already is (a non-privileged app role, the documented
    multi-tenant deployment), reuse it. Otherwise create a restricted role and
    grant it table access so it can run the isolation assertions.
    """
    if not await _current_role_bypasses_rls(admin):
        return admin_url
    async with admin.begin() as conn:
        # Idempotent: the fixture runs once per test, so create the role only if
        # it is missing (dropping it would fail while it still holds grants) and
        # re-grant every time.
        await conn.execute(
            text(
                "DO $do$ BEGIN "
                f"IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '{_PROBE_ROLE}') THEN "
                f"CREATE ROLE {_PROBE_ROLE} LOGIN PASSWORD '{_PROBE_PASSWORD}' NOBYPASSRLS; "
                "END IF; END $do$"
            )
        )
        await conn.execute(text(f"GRANT USAGE ON SCHEMA public TO {_PROBE_ROLE}"))
        await conn.execute(
            text(
                "GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES "
                f"IN SCHEMA public TO {_PROBE_ROLE}"
            )
        )
        await conn.execute(
            text(f"GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO {_PROBE_ROLE}")
        )
    return str(make_url(admin_url).set(username=_PROBE_ROLE, password=_PROBE_PASSWORD))


@pytest_asyncio.fixture
async def rls_scopes() -> AsyncIterator[tuple[SessionScope, SessionScope]]:
    """Yield ``(multi_tenant_scope, single_tenant_scope)`` bound to an RLS role.

    The multi-tenant scope sets the ``optimus.guild_id`` GUC per transaction; the
    single-tenant scope never does (reproducing the pre-fix code path).
    """
    assert PG_URL is not None
    await run_migrations(PG_URL)

    admin = create_async_engine(PG_URL)
    try:
        # Start from a clean slate regardless of prior runs; TRUNCATE is a
        # table-level op the table owner may run irrespective of row policies.
        async with admin.begin() as conn:
            await conn.execute(text("TRUNCATE detections RESTART IDENTITY CASCADE"))
        subject_url = await _make_rls_subject_url(admin, PG_URL)
    finally:
        await admin.dispose()

    engine = create_async_engine(subject_url)
    factory = create_session_factory(engine)
    try:
        yield (
            create_session_scope(factory, multi_tenant=True),
            create_session_scope(factory, multi_tenant=False),
        )
    finally:
        await engine.dispose()
        admin = create_async_engine(PG_URL)
        try:
            async with admin.begin() as conn:
                await conn.execute(text("TRUNCATE detections RESTART IDENTITY CASCADE"))
        finally:
            await admin.dispose()


def _detection(guild_id: int, key: str) -> Detection:
    return Detection(
        guild_id=guild_id,
        message_id=1,
        channel_id=2,
        attachment_id=3,
        uploader_id=4,
        verdict="scam",
        idempotency_key=key,
    )


async def _visible_guild_ids(scope: SessionScope, guild_id: int) -> set[int]:
    # Read WITHOUT the app-layer guild filter so any isolation observed is the
    # database's RLS policy, not the repository's WHERE clause.
    async with scope(guild_id) as session:
        rows = (await session.execute(text("SELECT guild_id FROM detections"))).scalars().all()
    return set(rows)


async def test_guc_scopes_reads_to_the_tenant(
    rls_scopes: tuple[SessionScope, SessionScope],
) -> None:
    """With the GUC set, guild A sees only its rows and never guild B's."""
    multi, _single = rls_scopes
    async with multi(GUILD_A) as s:
        s.add(_detection(GUILD_A, "a-1"))
    async with multi(GUILD_B) as s:
        s.add(_detection(GUILD_B, "b-1"))

    assert await _visible_guild_ids(multi, GUILD_A) == {GUILD_A}
    assert await _visible_guild_ids(multi, GUILD_B) == {GUILD_B}


async def test_cross_tenant_insert_is_rejected(
    rls_scopes: tuple[SessionScope, SessionScope],
) -> None:
    """The policy's WITH CHECK blocks writing another tenant's guild_id."""
    from sqlalchemy.exc import ProgrammingError

    multi, _single = rls_scopes
    with pytest.raises(ProgrammingError):
        async with multi(GUILD_A) as s:
            s.add(_detection(GUILD_B, "wrong-tenant"))


async def test_no_guc_returns_zero_rows_and_guc_corrects_it(
    rls_scopes: tuple[SessionScope, SessionScope],
) -> None:
    """Pre-fix reproduction: no GUC => zero rows; setting the GUC (fix) restores them."""
    multi, single = rls_scopes
    async with multi(GUILD_A) as s:
        s.add(_detection(GUILD_A, "a-1"))

    # Pre-fix code path: a guild-scoped read on an RLS-subject role that never
    # sets optimus.guild_id sees nothing at all — the bot was broken, not merely
    # missing a defence-in-depth layer.
    async with single(GUILD_A) as session:
        pre_fix = (await session.execute(text("SELECT count(*) FROM detections"))).scalar_one()
    assert pre_fix == 0

    # The fix — setting the GUC — makes the tenant's own rows visible again.
    assert await _visible_guild_ids(multi, GUILD_A) == {GUILD_A}
