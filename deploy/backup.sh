#!/usr/bin/env bash
# Nightly database backup for the hosted instance. Keeps the last 14 days.
#   crontab -e   then add:   15 2 * * * /opt/health-research-agent/deploy/backup.sh >> /var/log/hra-backup.log 2>&1
# Copy the backups off the server now and then (a backup on the same disk does not survive the disk).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p backups
stamp=$(date +%Y-%m-%d)
docker compose exec -T db pg_dump -U research -Fc research > "backups/research-$stamp.dump"
find backups -name 'research-*.dump' -mtime +14 -delete
echo "$(date -Is) backup ok: backups/research-$stamp.dump ($(du -h "backups/research-$stamp.dump" | cut -f1))"
