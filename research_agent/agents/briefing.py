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
    protocol = (notes.get("protocol") or {}).get("protocol") if isinstance(notes.get("protocol"), dict) else None
    if protocol and protocol.get("fields"):
        lines.append("- Question-specific fields you can count (value_counts, test_claim, propose_claim): " + "; ".join(
            f"{f['name']} ({'values: ' + ', '.join(f['values']) if f.get('values') else 'free text'}): "
            f"{f.get('definition', '')[:160]}" for f in protocol["fields"]))
    for agent, content in notes.items():
        if agent in (for_agent, "orchestrator", "synthesis", "protocol", "fulltext_index", "number_check", "mode", "usage"):
            continue
        text = json.dumps(content, ensure_ascii=False, default=str)
        if len(text) > note_chars:
            text = text[:note_chars] + "…"
        lines.append(f"- Output of {agent} agent: {text}")
    if len(lines) == 1 and not ids:
        lines.append("- Nothing has been done yet.")
    return "\n".join(lines)
