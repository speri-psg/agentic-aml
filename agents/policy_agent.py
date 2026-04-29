"""Policy Agent — ChromaDB RAG over AML policy PDFs (OpenAI-compatible client)."""

import os
import re
import sys
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from openai import OpenAI

from .base_agent import OLLAMA_BASE_URL, OLLAMA_MODEL, stop_event
from config import MAX_TOKENS_POLICY

_AGENTS_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_AGENTS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import CHROMA_PATH
import upload_kb as _upload_kb

COLLECTION_NAME = "framl_kb"
TOP_K = 7
_MAX_PER_SOURCE = 3  # diversity cap: no single document crowds out others

# Map query keywords → (source file, retrieval_query_override).
# When the user explicitly names a specific document/resolution, semantic search
# alone may rank it low if the user's phrasing doesn't match the document's language
# (e.g. "of banks" when the resolution addresses "states").  The retrieval_query
# override uses the document's own terminology so the operative clauses surface.
# Set retrieval_query to None to use the user's original query.
_TARGETED_SOURCES: list[tuple[list[str], str, str | None]] = [
    (["1373", "resolution 1373", "unsc 1373"],
     "UN_SC_Resolution_1373_2001.pdf",
     "states shall freeze funds criminalize terrorist financing obligations decides"),
    (["4th amld", "amld4", "amld 4", "2015/849"],
     "EU_4th_AMLD_2015_849.pdf",
     "customer due diligence beneficial ownership obliged entities requirements"),
    (["5th amld", "amld5", "amld 5", "2018/843"],
     "CELEX_32018L0843_EN_TXT.pdf",
     "virtual assets beneficial ownership register enhanced due diligence"),
    (["6th amld", "amld6", "amld 6", "2018/1673"],
     "CELEX_32018L1673_EN_TXT.pdf",
     "predicate offences criminal liability money laundering sanctions"),
    (["amlr", "2024/1624", "eu aml regulation"],
     "EU_AML_Regulation_2024_1624.pdf",
     "obliged entities customer due diligence beneficial ownership requirements"),
    (["unodc", "model provisions"],
     "UNODC_AML_Model_Provisions.pdf",
     "model law provisions money laundering financial intelligence unit"),
    (["fatf", "forty recommendations", "40 recommendation"],
     "FATF_40_Recommendations.pdf",
     None),  # FATF chunks already rank high — use original query
    (["eba gl 2021/02", "eba risk factors"],
     "EBA_GL_2021_02_MLTF_Risk_Factors.pdf",
     "risk factors customer due diligence guidelines obliged entities"),
]

_SYSTEM_PROMPT_TEMPLATE = (
    "You are ARIA — Agentic Risk Intelligence for AML. "
    "You answer AML policy and compliance questions using ONLY the retrieved policy documents shown below. "
    "CITATION RULES — follow exactly:\n"
    "- You may reference a source ONLY if its exact name appears in this list: {source_list}.\n"
    "- You MUST NOT write any of the following anywhere in your response:\n"
    "  * CFR section or part numbers (e.g. 31 CFR 1010.314, 31 CFR Part 1020)\n"
    "  * U.S.C. section references (e.g. 31 U.S.C. § 5318)\n"
    "  * FinCEN advisory or guidance numbers (e.g. FIN-2020-A005, FIN-2014-A008)\n"
    "  * OCC bulletin or manual codes (e.g. BSA-04, OCC 2000-17)\n"
    "  * Named authors, law firm names, or individuals\n"
    "  * Any statute name not in the retrieved documents (e.g. USA PATRIOT Act, Bank Secrecy Act)\n"
    "  * Specific dollar thresholds or dates not explicitly stated in the retrieved documents\n"
    "  * EU Article or paragraph numbers not explicitly quoted in the retrieved documents "
    "(e.g. do NOT write 'Article 13(1) of AMLD4' or 'Article 45(2) of Regulation 2024/1624' "
    "unless that exact article text appears in the retrieved chunks below)\n"
    "  * EU Recital numbers not present in the retrieved documents (e.g. 'Recital 22 of AMLD5')\n"
    "  * CELEX identifiers (e.g. 32015L0849, 32018L0843, 32024R1624)\n"
    "  * Official Journal references (e.g. OJ L 141, 5.6.2015)\n"
    "  * UN Security Council resolution numbers not in the retrieved documents "
    "(e.g. do NOT write 'UNSC Resolution 1267' unless it appears in the retrieved text)\n"
    "- If the retrieved documents address the question, summarise the concepts they discuss and "
    "reference the document by name from the list above only.\n"
    "IMPORTANT: Only trigger the disclaimer below if ALL retrieved documents are clearly off-topic. "
    "Disclaimer (use ONLY when every retrieved chunk is irrelevant): "
    "Begin with exactly: 'Note: The knowledge base does not contain specific guidance on this topic. "
    "The following is general AML knowledge only.' "
    "Then provide only general conceptual guidance — 3 to 5 sentences maximum. No numbers, no citations, no named sources. "
    "Be precise and compliance-focused. "
    "IMPORTANT: You MUST respond entirely in English. Do NOT use any Chinese, Japanese, or other non-English characters. "
)


class PolicyAgent:
    """Handles policy RAG — retrieval + generation, no tool-use loop needed."""

    def __init__(self):
        self.name = "policy"
        self.client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
        self.model = OLLAMA_MODEL
        self.collection = None
        self._load_kb()

    def _load_kb(self):
        if not os.path.exists(CHROMA_PATH):
            print("PolicyAgent: No ChromaDB found — policy RAG disabled.")
            return
        try:
            ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
            chroma = chromadb.PersistentClient(path=CHROMA_PATH)
            self.collection = chroma.get_collection(COLLECTION_NAME, embedding_function=ef)
            count = self.collection.count()
            if count == 0:
                self.collection = None
                print("PolicyAgent: ChromaDB empty.")
            else:
                print(f"PolicyAgent: KB loaded ({count} chunks).")
        except Exception as e:
            print(f"PolicyAgent: ChromaDB error: {e}")

    def retrieve(self, query: str, upload_only: bool = False) -> tuple:
        """Returns (context_text, source_names_list) merged from framl_kb + user_uploads.
        If upload_only=True, skips the regulatory KB and queries only user uploads."""
        sections, sources = [], []

        # ── regulatory KB ────────────────────────────────────────────────────
        if not upload_only and self.collection is not None:
            try:
                q_lower = query.lower()

                # Targeted pull: when user explicitly names a specific source,
                # semantic search may rank it low if phrasing doesn't match the
                # document's language (e.g. "banks" vs "states").  Pre-fetch 2
                # chunks directly from that source by keyword match.
                targeted_docs: list[tuple[str, dict]] = []
                for keywords, source_file, retrieval_query in _TARGETED_SOURCES:
                    if any(kw in q_lower for kw in keywords):
                        try:
                            tgt_q = retrieval_query if retrieval_query else query
                            tgt = self.collection.query(
                                query_texts=[tgt_q],
                                n_results=2,
                                where={"source": {"$eq": source_file}},
                            )
                            for d, m in zip(tgt["documents"][0], tgt["metadatas"][0]):
                                targeted_docs.append((d, m))
                            print(f"[policy] targeted pull from {source_file} "
                                  f"({len(tgt['documents'][0])} chunks)")
                        except Exception:
                            pass

                # Semantic query — fetch 3× candidates, apply per-source diversity cap
                raw = self.collection.query(
                    query_texts=[query], n_results=min(TOP_K * 3, self.collection.count())
                )
                raw_docs  = raw["documents"][0]
                raw_metas = raw["metadatas"][0]

                seen_per_src: dict[str, int] = {}
                chosen_docs, chosen_metas = [], []

                # Insert targeted chunks first (they count toward the per-source cap)
                for d, m in targeted_docs:
                    src = m["source"]
                    chosen_docs.append(d)
                    chosen_metas.append(m)
                    seen_per_src[src] = seen_per_src.get(src, 0) + 1

                # Fill remaining slots from semantic results, skipping duplicates
                targeted_ids = {id(d) for d in targeted_docs}
                for d, m in zip(raw_docs, raw_metas):
                    if len(chosen_docs) >= TOP_K:
                        break
                    src = m["source"]
                    if d in (td for td, _ in targeted_docs):
                        continue  # already included via targeted pull
                    if seen_per_src.get(src, 0) < _MAX_PER_SOURCE:
                        chosen_docs.append(d)
                        chosen_metas.append(m)
                        seen_per_src[src] = seen_per_src.get(src, 0) + 1

                for src in dict.fromkeys(m["source"] for m in chosen_metas):
                    sources.append(src)
                sections += [f"[Policy: {m['source']}]\n{d}"
                             for d, m in zip(chosen_docs, chosen_metas)]
            except Exception as e:
                print(f"PolicyAgent: retrieval error: {e}")

        # ── user-uploaded documents ───────────────────────────────────────────
        upload_ctx, upload_sources = _upload_kb.retrieve(query, top_k=5)
        if upload_ctx:
            sections.append(upload_ctx)
            for s in upload_sources:
                if s not in sources:
                    sources.append(s)

        return "\n\n---\n\n".join(sections), sources

    # Patterns that indicate fabricated inline citations
    _FABRICATED_PATTERNS = [
        # ── US patterns ──────────────────────────────────────────────────────
        re.compile(r"\bFIN-\d{4}-[A-Z]\d+\b"),                      # FIN-2020-A005
        re.compile(r"\b31\s+CFR\s+(Part\s+)?\d+(\.\d+)?\b"),        # 31 CFR 1010.314, 31 CFR Part 1020
        re.compile(r"\b(?:Title\s+)?\d+\s+U\.S\.C\.?\s*(?:§\s*\d+)?\b"),  # 31 U.S.C. § 5318, Title 31 U.S.C.
        re.compile(r"\bU\.S\.C\.?\s*§\s*\d+\b"),                    # U.S.C. § 5318 (standalone)
        re.compile(r"\bOCC\s+\d{4}-\d+\b", re.IGNORECASE),          # OCC 2000-17
        re.compile(r"\bBSA-\d+\b", re.IGNORECASE),                  # BSA-04
        re.compile(r"\(\s*[a-z]\s*\)\s*\(\s*\d+\s*\)\s*\(\s*[A-Z]\s*\)"),  # (a)(2)(A) orphan subsections
        re.compile(r"\(\s*[a-z]\s*\)\s*\(\s*\d+\s*\)"),             # (a)(1) orphan subsections
        # ── EU patterns ──────────────────────────────────────────────────────
        # CELEX identifiers: 3<year><L|R|D><seq> e.g. 32015L0849, 32024R1624
        re.compile(r"\b3\d{4}[LRD]\d{4}\b"),
        # OJ references: OJ L 141, OJ C 123/45
        re.compile(r"\bOJ\s+[LC]\s+\d+(?:/\d+)?\b"),
        # OJ date references: OJ L 141, 5.6.2015
        re.compile(r"\bOJ\s+[LC]\s+\d+,\s*\d+\.\d+\.\d{4}\b"),
        # EU Recital references: Recital 22, Recitals 10–12
        re.compile(r"\bRecital[s]?\s+\d+(?:\s*[–\-]\s*\d+)?\b", re.IGNORECASE),
        # ── UN patterns ──────────────────────────────────────────────────────
        # UNSC resolution numbers other than 1373 (which is in KB): S/RES/NNNN
        re.compile(r"\bS/RES/(?!1373\b)\d{4}\b"),
        # Standalone "Resolution NNNN" for non-1373 resolutions
        re.compile(r"\b(?:Resolution|Res\.)\s+(?!1373\b)\d{3,4}\b", re.IGNORECASE),
    ]

    def _strip_fabricated_citations(self, text: str, allowed_sources: list) -> str:
        """Remove fabricated inline citation patterns and unverified Source: lines."""
        allowed_lower = [s.lower() for s in allowed_sources]

        def _is_in_allowed_source(match_str: str) -> bool:
            """True if this matched string is a substring of an allowed source name."""
            m = match_str.lower()
            return any(m in src for src in allowed_lower)

        lines = text.splitlines()
        filtered = []
        for line in lines:
            # Drop Source: lines citing nothing in the retrieved set
            if re.match(r"^\s*source\s*:", line, re.IGNORECASE):
                cited = line.split(":", 1)[1].strip()
                if not any(s.lower() in cited.lower() for s in allowed_sources):
                    print(f"[policy] stripped fabricated source line: {line.strip()}")
                    continue

            # Strip inline fabricated citation tokens — skip if from an allowed source
            cleaned = line
            for pattern in self._FABRICATED_PATTERNS:
                def _replace(m, pat=pattern):
                    match_str = m.group(0)
                    if _is_in_allowed_source(match_str):
                        return match_str  # keep — it's a legitimate retrieved source
                    print(f"[policy] stripped inline citation: {match_str}")
                    return ""
                cleaned = pattern.sub(_replace, cleaned)

            # Clean up artifacts left by stripping: empty parens, double spaces, leading punctuation
            cleaned = re.sub(r"\(\s*\)", "", cleaned)        # empty ()
            cleaned = re.sub(r"\[\s*\]", "", cleaned)        # empty []
            cleaned = re.sub(r"\s+([,\.;:])", r"\1", cleaned)  # space before punctuation
            cleaned = re.sub(r"  +", " ", cleaned).strip()
            if cleaned:
                filtered.append(cleaned)
        return "\n".join(filtered)

    def run(self, query: str, tool_executor=None, policy_context: str = "", upload_only: bool = False) -> tuple:
        """Returns (response_text, []) — policy agent produces no charts."""
        context, sources = self.retrieve(query, upload_only=upload_only)
        source_list = ", ".join(sources) if sources else "none"
        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(source_list=source_list)

        if context:
            user_content = f"## Retrieved Policy Documents\n{context}\n\n## Question\n{query}"
        else:
            user_content = f"## Retrieved Policy Documents\n(No relevant documents found in the knowledge base.)\n\n## Question\n{query}"

        stream = self.client.chat.completions.create(
            model=self.model,
            max_tokens=MAX_TOKENS_POLICY,
            stream=True,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        parts = []
        try:
            for chunk in stream:
                if stop_event.is_set():
                    stream.close()
                    stop_event.clear()
                    return "Cancelled.", []
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    parts.append(delta)
        except Exception:
            pass
        text = "".join(parts)
        text = self._strip_fabricated_citations(text, sources)
        return text, []
