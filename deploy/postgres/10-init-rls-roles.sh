#!/usr/bin/env bash
# Provision the two least-privilege login roles the multi-tenant RLS design needs.
#
# The postgres image runs every executable in /docker-entrypoint-initdb.d/ once,
# on first boot of an empty data directory, connected to POSTGRES_DB as the
# POSTGRES_USER superuser. We use that window to create:
#
#   * optimus_app         NOBYPASSRLS  - the request/detection path role. It is
#                                        subject to FORCE ROW LEVEL SECURITY, so
#                                        it only ever sees the tenant whose id the
#                                        app pushes into optimus.guild_id.
#   * optimus_maintenance BYPASSRLS    - the scheduler / GDPR-erasure role. RLS is
#                                        bypassed so account-wide sweeps (retention,
#                                        purge, rollups, enumeration, forget-me) act
#                                        on every tenant instead of zero rows.
#
# POSTGRES_USER itself remains the schema owner used by `alembic upgrade` (the
# `migrate` compose service); ALTER DEFAULT PRIVILEGES below makes every table and
# sequence it later creates readable/writable by both roles automatically.
#
# Passwords come from the environment so nothing secret is baked into the image.
set -euo pipefail

: "${POSTGRES_USER:?POSTGRES_USER must be set}"
: "${POSTGRES_DB:?POSTGRES_DB must be set}"
APP_PASSWORD="${OPTIMUS_APP_DB_PASSWORD:-optimus_app}"
MAINT_PASSWORD="${OPTIMUS_MAINTENANCE_DB_PASSWORD:-optimus_maintenance}"

psql -v ON_ERROR_STOP=1 \
  --username "$POSTGRES_USER" \
  --dbname "$POSTGRES_DB" \
  --set app_password="$APP_PASSWORD" \
  --set maint_password="$MAINT_PASSWORD" <<'SQL'
DO $$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'optimus_app') THEN
    CREATE ROLE optimus_app LOGIN NOBYPASSRLS;
  END IF;
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'optimus_maintenance') THEN
    CREATE ROLE optimus_maintenance LOGIN BYPASSRLS;
  END IF;
END
$$;

ALTER ROLE optimus_app         WITH PASSWORD :'app_password'   NOBYPASSRLS;
ALTER ROLE optimus_maintenance WITH PASSWORD :'maint_password' BYPASSRLS;

GRANT CONNECT ON DATABASE :"POSTGRES_DB" TO optimus_app, optimus_maintenance;
GRANT USAGE ON SCHEMA public TO optimus_app, optimus_maintenance;

-- Objects already present (none on a fresh DB, but keeps re-runs correct).
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public
  TO optimus_app, optimus_maintenance;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public
  TO optimus_app, optimus_maintenance;

-- Tables/sequences the owner creates during `alembic upgrade` inherit these.
ALTER DEFAULT PRIVILEGES FOR ROLE :"POSTGRES_USER" IN SCHEMA public
  GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO optimus_app, optimus_maintenance;
ALTER DEFAULT PRIVILEGES FOR ROLE :"POSTGRES_USER" IN SCHEMA public
  GRANT USAGE, SELECT ON SEQUENCES TO optimus_app, optimus_maintenance;
SQL
