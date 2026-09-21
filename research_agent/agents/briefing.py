"""Compact view of the shared run state, given to each agent when it starts."""
from __future__ import annotations

import json


def brief(ctx, for_agent: str | None = None, note_chars: int = 2500) -> str:
    lines = []
    ids = ctx.shortlist_ids()
    lines.append(f"- Shortlist: {len(ids)} papers")
    ex = ctx.pg.execute(
        """SELECT e.source, count(*) n FROM run_papers rp JOIN extractions e
             ON e.paper_id = rp.paper_id AND e.schema_version = %s
           WHERE rp.run_id = %s GROUP BY e.source""", (ctx.extraction_version, ctx.run_id)).fetchall()
    if ex:
        lines.append("- Extractions: " + ", ".join(f"{r['n']} from {r['source']}" for r in ex))
    cl = ctx.pg.execute("SELECT status, count(*) n FROM claims WHERE run_id=%s GROUP BY status",
                        (ctx.run_id,)).fetchall()
    if cl:
        lines.append("- Claims: " + ", ".join(f"{r['n']} {r['status']}" for r in cl))
    notes = ctx.notes()
    for agent, content in notes.items():
        if agent in (for_agent, "orchestrator", "synthesis"):
            continue
        text = json.dumps(content, ensure_ascii=False, default=str)
        if len(text) > note_chars:
            text = text[:note_chars] + "…"
        lines.append(f"- Output of {agent} agent: {text}")
    if len(lines) == 1 and not ids:
        lines.append("- Nothing has been done yet.")
    return "\n".join(lines)
