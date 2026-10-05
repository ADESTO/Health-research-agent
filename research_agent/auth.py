"""Invite-only access for a hosted instance.

Off unless AUTH_REQUIRED=1, so a local install behaves exactly as before. When on:

    invite        `python -m research_agent.cli invite "Name" --email x@y` makes a user and prints a link.
                  The link carries a random token; only its SHA-256 is stored, so a database leak gives no
                  one a way in. Opening the link sets the token as an httpOnly cookie and goes to the app.
    who           every request carries the cookie (or `Authorization: Bearer <token>` for scripts).
    what          a user reads and changes their own runs, reads runs an admin has shared (the seeded
                  sample run, say), and sees nothing else: other runs answer 404, as if they did not exist.
    how much      RUNS_PER_DAY per user (a map counts twice), counted from the runs table, so it survives
                  restarts and holds across several server processes.
"""
from __future__ import annotations

import hashlib
import re
import secrets
import uuid

from research_agent.config import settings

COOKIE = "hra_session"
COOKIE_DAYS = 90
# Paths anyone may open: the page itself (it explains what to do without a link), sign-in, and health.
PUBLIC = {"/", "/login", "/logout", "/health", "/me"}
_RUN_PATH = re.compile(r"^/runs/([0-9a-fA-F-]{36})(/|$)")
_DRAFT_PATH = re.compile(r"^/drafts/(\d+)(/|$)")
_RESEARCHER_PATH = re.compile(r"^/researchers/(\d+)(/|$)")


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def create_user(pg, name: str, email: str | None = None, admin: bool = False) -> tuple[dict, str]:
    """(the user row, the plain token). The token is shown once and never stored."""
    token = secrets.token_urlsafe(32)
    row = pg.execute("INSERT INTO users (id, name, email, token_hash, is_admin) VALUES (%s,%s,%s,%s,%s) "
                     "RETURNING id, name, email, is_admin, active, created_at",
                     (str(uuid.uuid4()), name.strip()[:120], (email or "").strip()[:200] or None,
                      hash_token(token), admin)).fetchone()
    return row, token


def new_token(pg, user_id: str) -> str:
    """Replace a user's token (a lost or leaked link). The old link stops working at once."""
    token = secrets.token_urlsafe(32)
    pg.execute("UPDATE users SET token_hash=%s WHERE id=%s", (hash_token(token), user_id))
    return token


def invite_link(token: str) -> str:
    return f"{settings.public_url}/login?token={token}"


def user_for_token(pg, token: str | None) -> dict | None:
    if not token:
        return None
    row = pg.execute("SELECT id, name, email, is_admin FROM users WHERE token_hash=%s AND active",
                     (hash_token(token),)).fetchone()
    if row:
        pg.execute("UPDATE users SET last_seen=now() WHERE id=%s AND (last_seen IS NULL OR "
                   "last_seen < now() - interval '5 minutes')", (row["id"],))
    return row


def token_from(request) -> str | None:
    head = request.headers.get("authorization") or ""
    if head.lower().startswith("bearer "):
        return head[7:].strip() or None
    return request.cookies.get(COOKIE)


def run_of_path(pg, path: str) -> str | None | bool:
    """The run a path belongs to; None for a path that belongs to no run; False for an object that does not
    exist (a draft or researcher id with no row)."""
    m = _RUN_PATH.match(path)
    if m:
        return m.group(1).lower()
    for rx, table in ((_DRAFT_PATH, "drafts"), (_RESEARCHER_PATH, "researchers")):
        m = rx.match(path)
        if m:
            row = pg.execute(f"SELECT run_id FROM {table} WHERE id=%s", (int(m.group(1)),)).fetchone()
            return str(row["run_id"]) if row else False
    return None


def access(pg, user: dict, run_id: str, method: str) -> str:
    """"ok", "read_only" (a shared run someone else owns, asked to change) or "hidden". Owners and admins do
    anything with a run; any user may read a shared one. Nothing else."""
    if user.get("is_admin"):
        return "ok"
    row = pg.execute("SELECT owner_id, shared FROM runs WHERE run_id=%s", (run_id,)).fetchone()
    if not row:
        return "hidden"
    if row["owner_id"] and str(row["owner_id"]) == str(user["id"]):
        return "ok"
    if row["shared"]:
        return "ok" if method in ("GET", "HEAD") else "read_only"
    return "hidden"


def runs_today(pg, user_id: str) -> int:
    """Run units this user started in the last 24 hours (a map counts twice)."""
    row = pg.execute("SELECT coalesce(sum(CASE WHEN requested_mode = 'map' THEN 2 ELSE 1 END), 0) AS n FROM runs "
                     "WHERE owner_id=%s AND created_at > now() - interval '24 hours'", (user_id,)).fetchone()
    return int(row["n"] or 0)
