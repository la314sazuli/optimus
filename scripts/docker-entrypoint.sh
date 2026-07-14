#!/bin/sh
# Fix ownership of /data at container *start* (not build) time, then drop from
# root to the unprivileged `optimus` user before exec'ing the real command.
#
# Why this exists: a platform volume (Railway Volume, `docker run -v ...`,
# a Kubernetes PVC, etc.) is bind-mounted over /data when the container
# *starts*, which replaces whatever ownership the image set on /data at
# *build* time. If the volume's backing directory is owned by root (the
# common default for freshly provisioned volumes), the unprivileged
# `optimus` user can no longer write the SQLite database there and the app's
# own startup check fails fast with a permission error. Running this step as
# root, right after the volume is mounted, re-applies the ownership so it
# matches the image's original intent regardless of who provisioned the
# volume or what UID they defaulted it to.
#
# This script must run as root (the image's default USER is switched to
# `optimus` only via `gosu` below, not via the Dockerfile's USER directive)
# so it has permission to chown a directory it does not already own.
set -e

if [ -d /data ]; then
    chown -R optimus:optimus /data
fi

exec gosu optimus "$@"
