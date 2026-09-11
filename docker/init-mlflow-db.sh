#!/bin/sh
# Runs once, automatically, the first time the `db` container initializes its
# data directory (see docker-entrypoint-initdb.d in the postgres image's own
# docs) — creates the separate `mlflow` database the `mlflow` service's
# backend store uses, alongside the application's own POSTGRES_DB. Idempotent
# is unnecessary here: initdb.d scripts only ever run against a fresh,
# just-created data volume.
set -eu

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
    CREATE DATABASE mlflow;
EOSQL
