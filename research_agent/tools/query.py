"""Keyword query parsing for Postgres full-text search.

Agents write queries in web-search syntax with grouping, e.g.
    malaria (forecast OR prediction) -vaccine "early warning"
Postgres's websearch_to_tsquery ignores parentheses, so the query above became
    malaria & forecast | prediction & ...
which matches ANY paper mentioning "prediction". This module parses the query properly and builds the
tsquery from Postgres's own functions, so stemming and stop words behave exactly as before:

    phrase "…"   -> phraseto_tsquery      word  -> plainto_tsquery      word*  -> prefix match
    A B, A AND B -> &&                    A OR B -> ||                  -A, NOT A -> !!
    ( … )        -> grouping              precedence: NOT > AND > OR (as in most search engines)

Malformed input never raises: unbalanced parentheses are closed or ignored, and a query with no usable
terms matches nothing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_TOKEN = re.compile(r'"([^"]*)"?|(\()|(\))|(-)(?=[\w"(])|([^\s()"]+)')


@dataclass
class _Tok:
    kind: str   # phrase | word | lpar | rpar | not | or | and
    text: str = ""


def _tokens(q: str) -> list[_Tok]:
    out: list[_Tok] = []
    for m in _TOKEN.finditer(q or ""):
        phrase, lpar, rpar, minus, word = m.groups()
        if phrase is not None:
            if phrase.strip():
                out.append(_Tok("phrase", phrase.strip()))
        elif lpar:
            out.append(_Tok("lpar"))
        elif rpar:
            out.append(_Tok("rpar"))
        elif minus:
            # a leading '-' negates only at the start of a term ("-vaccine"), not inside words
            if m.start() == 0 or q[m.start() - 1] in " \t(":
                out.append(_Tok("not"))
        elif word:
            if word.upper() == "OR" or word == "|":
                out.append(_Tok("or"))
            elif word == "AND" or word == "&":
                out.append(_Tok("and"))
            elif word == "NOT":
                out.append(_Tok("not"))
            else:
                out.append(_Tok("word", word))
    return out


class _Parser:
    def __init__(self, toks: list[_Tok], prefix: str):
        self.t, self.i, self.params, self.prefix = toks, 0, {}, prefix

    def _peek(self) -> _Tok | None:
        return self.t[self.i] if self.i < len(self.t) else None

    def _param(self, value: str) -> str:
        name = f"{self.prefix}{len(self.params)}"
        self.params[name] = value
        return f"%({name})s"

    def parse(self) -> str | None:
        parts = []
        while self._peek() is not None:
            node = self._or()
            if node:
                parts.append(node)
            if self._peek() is not None and self._peek().kind == "rpar":
                self.i += 1   # stray ')': ignore it and carry on
        return " && ".join(f"({p})" for p in parts) if len(parts) > 1 else (parts[0] if parts else None)

    def _or(self) -> str | None:
        nodes = [self._and()]
        while self._peek() is not None and self._peek().kind == "or":
            self.i += 1
            nodes.append(self._and())
        nodes = [n for n in nodes if n]
        return " || ".join(f"({n})" for n in nodes) if len(nodes) > 1 else (nodes[0] if nodes else None)

    def _and(self) -> str | None:
        nodes = []
        while True:
            tok = self._peek()
            if tok is None or tok.kind in ("or", "rpar"):
                break
            if tok.kind == "and":
                self.i += 1
                continue
            node = self._unary()
            if node:
                nodes.append(node)
        return " && ".join(f"({n})" for n in nodes) if len(nodes) > 1 else (nodes[0] if nodes else None)

    def _unary(self) -> str | None:
        tok = self._peek()
        if tok.kind == "not":
            self.i += 1
            if self._peek() is None or self._peek().kind in ("or", "rpar", "and"):
                return None
            inner = self._unary()
            return f"!!({inner})" if inner else None
        return self._primary()

    def _primary(self) -> str | None:
        tok = self._peek()
        self.i += 1
        if tok.kind == "lpar":
            inner = self._or()
            if self._peek() is not None and self._peek().kind == "rpar":
                self.i += 1   # a missing ')' is treated as closed at the end of the query
            return inner
        if tok.kind == "phrase":
            return f"phraseto_tsquery('english', {self._param(tok.text)})"
        if tok.kind == "word":
            w = tok.text
            if w.endswith("*") and re.fullmatch(r"[A-Za-z0-9]+\*", w):
                return f"to_tsquery('english', {self._param(w[:-1].lower() + ':*')})"
            # hyphenated or punctuated words: keep the words together, as websearch_to_tsquery did
            fn = "phraseto_tsquery" if re.search(r"[^\w*]", w) else "plainto_tsquery"
            return f"{fn}('english', {self._param(w.strip('*'))})"
        return None   # a lone ')' or operator in term position


def tsquery_sql(query: str, prefix: str = "tq") -> tuple[str, dict]:
    """SQL expression for a tsquery, and its parameters (named `prefix0`, `prefix1`, ...).

    Use as: f"... WHERE tsv @@ ({sql}) ...", with the returned params merged into the query's params."""
    toks, depth = [], 0
    for t in _tokens(query):          # drop unmatched ')' so they cannot split the query
        if t.kind == "lpar":
            depth += 1
        elif t.kind == "rpar":
            if depth == 0:
                continue
            depth -= 1
        toks.append(t)
    parser = _Parser(toks, prefix)
    sql = parser.parse()
    if not sql:
        return "plainto_tsquery('english', '')", {}   # no usable terms: matches nothing
    return sql, parser.params
