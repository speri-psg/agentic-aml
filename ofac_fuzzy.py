"""
ofac_fuzzy.py — Two-stage fuzzy name screening against the OFAC SDN list.

Stage 1 — Phonetic pre-filter (fast, indexed)
  Double Metaphone codes are pre-computed for every name + alias token at
  build time and stored in a dedicated `sdn_phonetic` table with a B-tree
  index.  A query name is tokenised, its codes looked up in O(k log n),
  and only the ~50-200 phonetically matching candidates proceed to Stage 2.

Stage 2 — Multi-algorithm fuzzy scoring
  Three scores are computed for each candidate name / alias pair:
    • Jaro-Winkler   — transpositions and prefix weight (best for short names)
    • Token-sort ratio  — handles name-part reordering (Last, First vs First Last)
    • Partial ratio   — catches partial matches (Ali Hassan vs Ali Hassan Al-Rashid)
  The maximum of the three is taken.  Hits at or above `threshold` (default 85)
  are returned, sorted by score descending.

Usage
-----
  from ofac_fuzzy import screen_name
  hits = screen_name("John Smith", "/path/to/ofac_sdn.db")
  # → [{"name": ..., "matched_on": ..., "score": 91.2, "country": ..., ...}, ...]

Build requirement
-----------------
  run build_ofac_db.py after installing rapidfuzz + jellyfish to populate the
  sdn_phonetic table.  The table is a no-op query if it doesn't exist yet
  (returns empty list with a warning).
"""

import re
import sqlite3
from typing import Optional

try:
    import jellyfish
    _JELLYFISH = True
except ImportError:
    _JELLYFISH = False

try:
    from rapidfuzz import fuzz, utils as rfutils
    _RAPIDFUZZ = True
except ImportError:
    _RAPIDFUZZ = False

THRESHOLD_DEFAULT = 85   # minimum score (0-100) to include in results
MAX_RESULTS       = 10   # maximum hits returned


# ── Text normalisation ─────────────────────────────────────────────────────────

_STRIP_RE = re.compile(r"[^A-Za-z0-9 ]")

def _normalise(text: str) -> str:
    """Lowercase, remove punctuation, collapse whitespace."""
    return _STRIP_RE.sub(" ", text).lower().strip()

def _tokens(name: str) -> list[str]:
    """Return uppercase tokens, filtering noise words."""
    _NOISE = {"AL", "EL", "BIN", "BEN", "ABU", "UL", "UR", "JR", "SR", "II", "III"}
    return [t for t in name.upper().split() if len(t) > 1 and t not in _NOISE]


# ── Phonetic codes ─────────────────────────────────────────────────────────────

def _phonetic_codes(name: str) -> set[str]:
    """Return all non-empty Double Metaphone codes for all tokens in name."""
    if not _JELLYFISH:
        return set()
    codes = set()
    for tok in _tokens(name):
        p, s = jellyfish.double_metaphone(tok)
        if p:
            codes.add(p)
        if s:
            codes.add(s)
    return codes


# ── Fuzzy scoring ──────────────────────────────────────────────────────────────

def _score_pair(query: str, candidate: str) -> float:
    """Score query vs candidate — returns 0-100."""
    if not _RAPIDFUZZ:
        # Fallback: simple case-insensitive containment
        q, c = query.lower(), candidate.lower()
        if q in c or c in q:
            return 80.0
        return 0.0

    q = rfutils.default_process(query)
    c = rfutils.default_process(candidate)
    if not q or not c:
        return 0.0

    return max(
        fuzz.jaro_winkler_similarity(q, c) * 100,
        fuzz.token_sort_ratio(q, c),
        fuzz.partial_ratio(q, c),
    )


# ── Main screening function ────────────────────────────────────────────────────

def screen_name(
    query: str,
    db_path: str,
    threshold: float = THRESHOLD_DEFAULT,
    max_results: int = MAX_RESULTS,
) -> list[dict]:
    """
    Screen a name against the OFAC SDN list.

    Parameters
    ----------
    query       : name to screen (e.g. "John Smith" or "Ali Hassan Al-Rashidi")
    db_path     : path to ofac_sdn.db (must have been built with build_ofac_db.py)
    threshold   : minimum similarity score 0-100 (default 85)
    max_results : cap on returned hits (default 10)

    Returns
    -------
    List of dicts sorted by score desc, each with keys:
        name, matched_on, aliases, country, sanction_type, sdn_type,
        dob, program_raw, score
    """
    if not query or not query.strip():
        return []

    try:
        con = sqlite3.connect(db_path)
        cur = con.cursor()
    except Exception as e:
        print(f"[ofac_fuzzy] DB open error: {e}")
        return []

    # ── Stage 1: phonetic pre-filter ──────────────────────────────────────────
    codes = _phonetic_codes(query)
    candidate_nums: set[str] = set()

    if codes:
        try:
            placeholders = ",".join("?" * len(codes))
            rows = cur.execute(
                f"SELECT DISTINCT ent_num FROM sdn_phonetic WHERE code IN ({placeholders})",
                list(codes),
            ).fetchall()
            candidate_nums = {r[0] for r in rows}
        except sqlite3.OperationalError:
            # sdn_phonetic table doesn't exist — fall back to full scan
            print("[ofac_fuzzy] WARNING: sdn_phonetic table missing — "
                  "run build_ofac_db.py to rebuild the DB with phonetic index.")
            rows = cur.execute(
                "SELECT DISTINCT ent_num FROM sdn"
            ).fetchall()
            candidate_nums = {r[0] for r in rows}
    else:
        # No phonetic lib — full scan (18k rows, still fast enough for a demo)
        rows = cur.execute("SELECT DISTINCT ent_num FROM sdn").fetchall()
        candidate_nums = {r[0] for r in rows}

    if not candidate_nums:
        con.close()
        return []

    # ── Fetch full records for candidates ──────────────────────────────────────
    placeholders = ",".join("?" * len(candidate_nums))
    records = cur.execute(
        f"SELECT ent_num, name, aliases, country, sanction_type, sdn_type, dob, program_raw "
        f"FROM sdn WHERE ent_num IN ({placeholders})",
        list(candidate_nums),
    ).fetchall()
    con.close()

    # ── Stage 2: fuzzy scoring ─────────────────────────────────────────────────
    results = []
    for ent_num, name, aliases_str, country, sanction_type, sdn_type, dob, program_raw in records:
        best_score  = _score_pair(query, name)
        matched_on  = name

        if aliases_str:
            for alias in aliases_str.split("|"):
                alias = alias.strip()
                if not alias:
                    continue
                s = _score_pair(query, alias)
                if s > best_score:
                    best_score = s
                    matched_on = alias

        if best_score >= threshold:
            results.append({
                "name":          name,
                "matched_on":    matched_on,
                "aliases":       aliases_str or "",
                "country":       country or "Unknown",
                "sanction_type": sanction_type or "Other",
                "sdn_type":      sdn_type or "entity",
                "dob":           dob or "",
                "program_raw":   program_raw or "",
                "score":         round(best_score, 1),
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:max_results]


# ── Formatted result text (for model to copy verbatim) ────────────────────────

def format_results(query: str, hits: list[dict], threshold: float = THRESHOLD_DEFAULT) -> str:
    """Return a PRE-COMPUTED text block the model can copy verbatim."""
    lines = [
        "=== PRE-COMPUTED OFAC NAME SCREENING (copy verbatim, do not alter) ===",
        f"Query: {query}",
        f"Threshold: {threshold}%",
        "",
    ]
    if not hits:
        lines.append("Result: NO MATCH — name not found on the OFAC SDN list above the scoring threshold.")
    else:
        lines.append(f"Result: {len(hits)} HIT(S) FOUND")
        lines.append("")
        for i, h in enumerate(hits, 1):
            lines.append(f"**Match {i}** — Score: {h['score']}%")
            lines.append(f"- SDN Name: {h['name']}")
            if h["matched_on"] != h["name"]:
                lines.append(f"- Matched on alias: {h['matched_on']}")
            lines.append(f"- Type: {h['sdn_type'].title()}")
            lines.append(f"- Country: {h['country']}")
            lines.append(f"- Sanctions Program: {h['sanction_type']}")
            if h["dob"]:
                lines.append(f"- DOB: {h['dob']}")
            lines.append("")
    lines.append("=== END OFAC NAME SCREENING ===")
    return "\n".join(lines)
