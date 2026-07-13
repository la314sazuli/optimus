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
from dataclasses import dataclass
from datetime import UTC, datetime

import pytest
import pytest_asyncio
from sqlalchemy import make_url, text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from optimus.app.migrate import run_migrations
from optimus.db.engine import (
    SessionScope,
    create_maintenance_scope,
    create_session_factory,
    create_session_scope,
)
from optimus.db.models import Detection, Guild
from optimus.db.repositories import GuildListRepository, UserOptoutRepository
from optimus.services.scheduler import tasks

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

# A dedicated BYPASSRLS login role mirroring the deployment's `optimus_maintenance`
# role: cross-tenant maintenance (scheduler sweeps, GDPR erasure) must NOT be
# filtered by FORCE RLS, so it runs as a role that bypasses the policy.
_MAINT_ROLE = "optimus_rls_maint"
_MAINT_PASSWORD = "optimus_rls_maint"


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


async def _make_maintenance_url(admin: AsyncEngine, admin_url: str) -> str:
    """Return a URL for a BYPASSRLS role that can run cross-tenant maintenance."""
    async with admin.begin() as conn:
        await conn.execute(
            text(
                "DO $do$ BEGIN "
                f"IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '{_MAINT_ROLE}') THEN "
                f"CREATE ROLE {_MAINT_ROLE} LOGIN PASSWORD '{_MAINT_PASSWORD}' BYPASSRLS; "
                f"ELSE ALTER ROLE {_MAINT_ROLE} BYPASSRLS; END IF; END $do$"
            )
        )
        await conn.execute(text(f"GRANT USAGE ON SCHEMA public TO {_MAINT_ROLE}"))
        await conn.execute(
            text(
                "GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES "
                f"IN SCHEMA public TO {_MAINT_ROLE}"
            )
        )
        await conn.execute(
            text(f"GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO {_MAINT_ROLE}")
        )
    return str(make_url(admin_url).set(username=_MAINT_ROLE, password=_MAINT_PASSWORD))


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


# --------------------------------------------------------------------------- #
# Cross-tenant maintenance under RLS
#
# The scheduler's retention/purge/rollup/enumeration jobs and the /forget_me GDPR
# erasure are genuinely account-wide. Run on the RLS-subject role with no GUC they
# see ZERO rows (FORCE RLS filters everything), so every job silently no-ops in the
# exact deployment where isolation works. The fix runs them on a BYPASSRLS
# maintenance role. These tests prove maintenance ops affect the right rows on that
# role while the RLS-subject role reproduces the zero-rows bug.
# --------------------------------------------------------------------------- #

_MAINT_TABLES = ("guilds", "detections", "appeals", "mod_actions", "stats_rollups")


@dataclass
class MaintEnv:
    """Scopes bound to the two roles, plus a seeding factory on the superuser."""

    subject_unscoped: SessionScope  # RLS-subject role, no GUC (the buggy path)
    subject_multi: SessionScope  # RLS-subject role, GUC set per tenant
    maintenance: SessionScope  # BYPASSRLS role, cross-tenant maintenance
    admin_scope: SessionScope  # superuser, for seeding across tenants


async def _truncate(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.execute(text(f"TRUNCATE {', '.join(_MAINT_TABLES)} RESTART IDENTITY CASCADE"))


@pytest_asyncio.fixture
async def maint_env() -> AsyncIterator[MaintEnv]:
    assert PG_URL is not None
    await run_migrations(PG_URL)

    admin = create_async_engine(PG_URL)
    await _truncate(admin)
    subject_url = await _make_rls_subject_url(admin, PG_URL)
    maint_url = await _make_maintenance_url(admin, PG_URL)

    subject_engine = create_async_engine(subject_url)
    maint_engine = create_async_engine(maint_url)
    subject_factory = create_session_factory(subject_engine)
    admin_factory = create_session_factory(admin)
    try:
        yield MaintEnv(
            subject_unscoped=create_session_scope(subject_factory, multi_tenant=False),
            subject_multi=create_session_scope(subject_factory, multi_tenant=True),
            maintenance=create_maintenance_scope(create_session_factory(maint_engine)),
            admin_scope=create_session_scope(admin_factory, multi_tenant=False),
        )
    finally:
        await subject_engine.dispose()
        await maint_engine.dispose()
        await _truncate(admin)
        await admin.dispose()


async def _seed_guild(scope: SessionScope, guild_id: int, *, retention_days: int) -> None:
    async with scope() as session:
        session.add(Guild(guild_id=guild_id, retention_days=retention_days))


async def _seed_detection(
    scope: SessionScope, guild_id: int, key: str, *, uploader_id: int, created_at: datetime
) -> None:
    async with scope() as session:
        det = _detection(guild_id, key)
        det.uploader_id = uploader_id
        det.created_at = created_at
        session.add(det)


async def _count_detections(scope: SessionScope, guild_id: int | None = None) -> int:
    # Runs on the maintenance/admin (BYPASSRLS) scope, so it counts across tenants.
    async with scope(guild_id) as session:
        return int((await session.execute(text("SELECT count(*) FROM detections"))).scalar_one())


async def test_enumeration_sees_all_tenants_only_under_maintenance(
    maint_env: MaintEnv,
) -> None:
    """GuildListRepository.all_ids: every tenant under maintenance, none under RLS."""
    await _seed_guild(maint_env.admin_scope, GUILD_A, retention_days=30)
    await _seed_guild(maint_env.admin_scope, GUILD_B, retention_days=30)

    async with maint_env.maintenance() as session:
        assert set(await GuildListRepository(session).all_ids()) == {GUILD_A, GUILD_B}

    # The pre-fix path: the RLS-subject role with no GUC enumerates nothing, so
    # every per-guild maintenance loop keyed off this list never runs.
    async with maint_env.subject_unscoped() as session:
        assert set(await GuildListRepository(session).all_ids()) == set()


async def test_rollups_written_for_all_tenants_under_maintenance(
    maint_env: MaintEnv,
) -> None:
    """roll_up_stats writes a rollup per guild under maintenance; zero under RLS."""
    now = datetime(2026, 1, 2, 12, 0, tzinfo=UTC)
    in_window = datetime(2026, 1, 2, 11, 30, tzinfo=UTC)  # previous hour bucket
    await _seed_guild(maint_env.admin_scope, GUILD_A, retention_days=30)
    await _seed_guild(maint_env.admin_scope, GUILD_B, retention_days=30)
    await _seed_detection(
        maint_env.admin_scope, GUILD_A, "a-1", uploader_id=7, created_at=in_window
    )
    await _seed_detection(
        maint_env.admin_scope, GUILD_B, "b-1", uploader_id=8, created_at=in_window
    )

    written = await tasks.roll_up_stats(maint_env.maintenance, now=now)
    assert written == 2
    async with maint_env.maintenance() as session:
        rollups = (await session.execute(text("SELECT count(*) FROM stats_rollups"))).scalar_one()
    assert rollups == 2

    # Pre-fix: the RLS-subject role enumerates zero guilds, so nothing is written.
    assert await tasks.roll_up_stats(maint_env.subject_unscoped, now=now) == 0


async def test_retention_deletes_across_tenants_under_maintenance(
    maint_env: MaintEnv,
) -> None:
    """enforce_retention removes old rows for every tenant under maintenance."""
    now = datetime(2026, 1, 2, 12, 0, tzinfo=UTC)
    old = datetime(2000, 1, 1, tzinfo=UTC)
    await _seed_guild(maint_env.admin_scope, GUILD_A, retention_days=1)
    await _seed_guild(maint_env.admin_scope, GUILD_B, retention_days=1)
    await _seed_detection(maint_env.admin_scope, GUILD_A, "a-old", uploader_id=7, created_at=old)
    await _seed_detection(maint_env.admin_scope, GUILD_B, "b-old", uploader_id=8, created_at=old)

    deleted = await tasks.enforce_retention(maint_env.maintenance, default_days=1, now=now)
    assert deleted == 2
    assert await _count_detections(maint_env.admin_scope) == 0

    # Pre-fix: the RLS-subject role enumerates zero guilds and deletes nothing.
    await _seed_detection(maint_env.admin_scope, GUILD_A, "a-old2", uploader_id=7, created_at=old)
    assert await tasks.enforce_retention(maint_env.subject_unscoped, default_days=1, now=now) == 0
    assert await _count_detections(maint_env.admin_scope) == 1


async def test_deployment_purge_deletes_across_tenants_under_maintenance(
    maint_env: MaintEnv,
) -> None:
    """purge_old_data removes old rows deployment-wide under maintenance; zero under RLS."""
    now = datetime(2026, 1, 2, 12, 0, tzinfo=UTC)
    old = datetime(2000, 1, 1, tzinfo=UTC)
    await _seed_guild(maint_env.admin_scope, GUILD_A, retention_days=1)
    await _seed_guild(maint_env.admin_scope, GUILD_B, retention_days=1)
    await _seed_detection(maint_env.admin_scope, GUILD_A, "a-old", uploader_id=7, created_at=old)
    await _seed_detection(maint_env.admin_scope, GUILD_B, "b-old", uploader_id=8, created_at=old)

    # Pre-fix: the unscoped RLS-subject batch delete touches zero rows.
    assert (
        await tasks.purge_old_data(
            maint_env.subject_unscoped, retention_days=1, batch_size=100, now=now
        )
        == 0
    )
    assert await _count_detections(maint_env.admin_scope) == 2

    purged = await tasks.purge_old_data(
        maint_env.maintenance, retention_days=1, batch_size=100, now=now
    )
    assert purged == 2
    assert await _count_detections(maint_env.admin_scope) == 0


async def test_gdpr_erasure_spans_tenants_only_under_maintenance(
    maint_env: MaintEnv,
) -> None:
    """/forget_me erases a user's rows across every guild under maintenance."""
    now = datetime(2026, 1, 2, 12, 0, tzinfo=UTC)
    user = 4242
    await _seed_guild(maint_env.admin_scope, GUILD_A, retention_days=30)
    await _seed_guild(maint_env.admin_scope, GUILD_B, retention_days=30)
    await _seed_detection(maint_env.admin_scope, GUILD_A, "a-u", uploader_id=user, created_at=now)
    await _seed_detection(maint_env.admin_scope, GUILD_B, "b-u", uploader_id=user, created_at=now)

    # Pre-fix DM path: the unscoped RLS-subject role erases nothing.
    async with maint_env.subject_unscoped() as session:
        assert await UserOptoutRepository(session).purge_user(user) == 0
    assert await _count_detections(maint_env.admin_scope) == 2

    async with maint_env.maintenance() as session:
        assert await UserOptoutRepository(session).purge_user(user) == 2
    assert await _count_detections(maint_env.admin_scope) == 0


async def test_request_path_isolation_holds_with_maintenance_seed(
    maint_env: MaintEnv,
) -> None:
    """Per-tenant isolation on the request role is unaffected by maintenance seeding."""
    now = datetime(2026, 1, 2, 12, 0, tzinfo=UTC)
    await _seed_guild(maint_env.admin_scope, GUILD_A, retention_days=30)
    await _seed_guild(maint_env.admin_scope, GUILD_B, retention_days=30)
    await _seed_detection(maint_env.admin_scope, GUILD_A, "a-1", uploader_id=7, created_at=now)
    await _seed_detection(maint_env.admin_scope, GUILD_B, "b-1", uploader_id=8, created_at=now)

    assert await _visible_guild_ids(maint_env.subject_multi, GUILD_A) == {GUILD_A}
    assert await _visible_guild_ids(maint_env.subject_multi, GUILD_B) == {GUILD_B}
