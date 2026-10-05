# Hosting the beta

One small server runs everything: Postgres with the corpus, the app (which also works the job queue),
and Caddy for HTTPS. Testers sign in with a private invite link. Each sees their own runs plus any run
you share, and can start a few runs a day.

## 1. A server and a domain

- A Linux VPS with 4 vCPUs, 8 GB RAM and enough disk for your database plus backups (check the size of
  your local database first: `docker compose exec db psql -U research -c "SELECT pg_size_pretty(pg_database_size('research'))"`).
  Ubuntu 24.04 is fine.
- A domain or subdomain (for example `research.yourdomain.com`) with an A record pointing at the server's IP.
- Install Docker: `curl -fsSL https://get.docker.com | sh`
- Open only ports 22, 80 and 443: `ufw allow OpenSSH && ufw allow 80 && ufw allow 443 && ufw enable`

## 2. Code and settings

```bash
git clone https://github.com/ADESTO/Health-research-agent.git /opt/health-research-agent
cd /opt/health-research-agent
git checkout beta-0.1            # the server runs a tagged version, never work in progress
cp .env.example .env
nano .env
```

In the server's `.env` set at least:

```
AUTH_REQUIRED=1
PUBLIC_URL=https://research.yourdomain.com
DOMAIN=research.yourdomain.com
POSTGRES_PASSWORD=<a long random string>
ANTHROPIC_API_KEY=...            # and any other provider keys you use
NCBI_API_KEY=...
RUNS_PER_DAY=4
```

The server's `.env` is separate from the one on your laptop and is never committed.

## 3. Move your corpus over

Re-ingesting on the server takes hours. Copy your local database instead.

On your laptop:

```bash
docker compose exec -T db pg_dump -U research -Fc research > research.dump
scp research.dump root@<server-ip>:/opt/health-research-agent/
```

On the server:

```bash
docker compose up -d db
docker compose exec -T db pg_restore -U research -d research --no-owner --clean --if-exists < research.dump
```

This brings your existing runs too, including the malaria sample run.

## 4. Start it

```bash
docker compose --profile hosted up -d --build
docker compose logs -f api       # Ctrl+C to stop watching
```

Open `https://research.yourdomain.com`. Without an invite link it shows "This is a private beta".

## 5. Users

```bash
alias hra='docker compose exec api python -m research_agent.cli'

hra invite "Stompie Ade" --email stompieade@gmail.com --admin    # you: no daily limit, sees every run
hra invite "Dr Jane Doe" --email jane@example.org                 # prints a link: send it privately
hra users                                                          # who, how many runs, last seen
hra relink jane@example.org                                        # lost or leaked link: old one stops working
hra deactivate jane@example.org                                    # switch someone off; their runs are kept
```

Open your own admin link first. Runs made before users existed have no owner, so only admins see them.

## 6. The sample run

Share the malaria run so every tester can explore a finished run before their first one finishes:

```bash
hra runs                          # find its run_id
hra share <run_id>                # every user can read it; only you can change it
hra share <run_id> --off          # private again
```

Testers see it in their list with a "sample" tag. It is read-only for them: follow-ups and drafts on it are
refused with a message telling them to start their own run.

## 7. Backups

```bash
crontab -e
# add:
15 2 * * * /opt/health-research-agent/deploy/backup.sh >> /var/log/hra-backup.log 2>&1
```

It keeps 14 nightly dumps in `backups/`. Copy them off the server now and then, for example with `scp` to
your laptop: a backup on the same disk is lost with the disk. To restore one, use the `pg_restore` command
from step 3.

## 8. Updating

```bash
cd /opt/health-research-agent
git fetch --tags && git checkout beta-0.2
docker compose --profile hosted up -d --build
```

Database changes are applied automatically when the app starts. A run in progress when you restart is
picked up again by the worker and resumes from its last finished step.

## What testers can and cannot do

| | Tester | Admin (you) |
|---|---|---|
| Start runs and maps | Up to RUNS_PER_DAY a day (a map counts as two) | No limit |
| See runs | Their own, plus shared ones | All |
| Follow-ups, drafts, exports, counting panel | On their own runs | On any run |
| Shared sample run | Read and export only | Everything |
| Open-ended researcher | No (set RESEARCHERS_ADMIN_ONLY=0 to allow) | Yes |

Papers and full texts are stored once in the shared corpus, so a full text fetched for one tester's run is
reused by every later run without fetching it again.

## Before inviting anyone

- Set a monthly spending limit and an alert in the Anthropic console (and on Groq, if used).
- Open the site in a private browser window and check you see the private-beta page, not the app.
- Open your admin link, start one small run, and check it finishes.
