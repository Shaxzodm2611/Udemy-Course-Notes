#!/usr/bin/env python
"""Convert a Markdown notes file into a summarized LaTeX document.

Goals:
- Preserve the structure of the notes (headings)
- Summarize prose (keep only the most important paragraphs)
- Keep key bullets (cap per section)
- Keep code fences by default

No third-party dependencies.

Example:
  python tools/md_to_latex_summary.py Javascript-Refresher/Notes.md

Outputs:
  Javascript-Refresher/Notes_summary.tex (by default)
"""

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


@dataclass
class Heading:
    level: int
    text: str


@dataclass
class CodeBlock:
    language: str
    code: str


@dataclass
class Paragraph:
    text: str


@dataclass
class Bullet:
    indent: int
    text: str


Block = Heading | CodeBlock | Paragraph | Bullet


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_BULLET_RE = re.compile(r"^(?P<indent>\s*)([-*+]|\d+\.)\s+(?P<text>.*)$")
_FENCE_RE = re.compile(r"^```(?P<lang>[a-zA-Z0-9_-]*)\s*$")


def _latex_escape(text: str) -> str:
    # Order matters: escape backslash first.
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("#", r"\#"),
        ("$", r"\$"),
        ("%", r"\%"),
        ("&", r"\&"),
        ("_", r"\_"),
        ("^", r"\textasciicircum{}"),
        ("~", r"\textasciitilde{}"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def _inline_code_to_latex(text: str) -> str:
    # Convert `inline code` to \texttt{...}
    # Keep it simple; avoid nested backticks.
    parts: List[str] = []
    cursor = 0
    for match in re.finditer(r"`([^`]+)`", text):
        parts.append(_latex_escape(text[cursor : match.start()]))
        parts.append(r"\texttt{" + _latex_escape(match.group(1)) + "}")
        cursor = match.end()
    parts.append(_latex_escape(text[cursor:]))
    return "".join(parts)


def _strip_md_emphasis(text: str) -> str:
    # Remove very simple emphasis markers to keep LaTeX output clean.
    # This intentionally does not implement full Markdown.
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"_(.*?)_", r"\1", text)
    return text


def _md_links_to_latex(text: str) -> str:
    # Convert [label](url) to \href{url}{label}
    def repl(m: re.Match[str]) -> str:
        label = m.group(1)
        url = m.group(2)
        return r"\href{" + _latex_escape(url) + "}{" + _latex_escape(label) + "}"

    return re.sub(r"\[([^\]]+)]\(([^)]+)\)", repl, text)


def parse_markdown(md: str) -> List[Block]:
    blocks: List[Block] = []
    lines = md.splitlines()

    in_code = False
    code_lang = ""
    code_buf: List[str] = []

    para_buf: List[str] = []

    def flush_paragraph() -> None:
        nonlocal para_buf
        text = "\n".join([ln.rstrip() for ln in para_buf]).strip()
        if text:
            blocks.append(Paragraph(text=text))
        para_buf = []

    for raw in lines:
        line = raw.rstrip("\n")

        # Code fences
        if not in_code:
            m_fence = _FENCE_RE.match(line)
            if m_fence:
                flush_paragraph()
                in_code = True
                code_lang = (m_fence.group("lang") or "").strip()
                code_buf = []
                continue
        else:
            if line.strip() == "```":
                blocks.append(CodeBlock(language=code_lang, code="\n".join(code_buf).rstrip()))
                in_code = False
                code_lang = ""
                code_buf = []
                continue
            code_buf.append(raw)  # keep original indentation
            continue

        # Headings
        m_heading = _HEADING_RE.match(line)
        if m_heading:
            flush_paragraph()
            level = len(m_heading.group(1))
            text = m_heading.group(2).strip()
            blocks.append(Heading(level=level, text=text))
            continue

        # Bullets / numbered lists
        m_bullet = _BULLET_RE.match(line)
        if m_bullet:
            flush_paragraph()
            indent = len(m_bullet.group("indent").replace("\t", "    "))
            text = m_bullet.group("text").strip()
            blocks.append(Bullet(indent=indent, text=text))
            continue

        # Blank line ends paragraphs
        if not line.strip():
            flush_paragraph()
            continue

        # Otherwise: paragraph content
        para_buf.append(line)

    flush_paragraph()

    # If file ends while inside a fence, treat as code block anyway
    if in_code and code_buf:
        blocks.append(CodeBlock(language=code_lang, code="\n".join(code_buf).rstrip()))

    return blocks


# ─────────────────────────────────────────────────────────────────────────────
# Advanced multi-factor scoring engine
# ─────────────────────────────────────────────────────────────────────────────

_STOPWORDS: Set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "dare",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "each", "few",
    "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "just",
    "and", "but", "if", "or", "because", "until", "while", "although",
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves", "he", "him",
    "his", "himself", "she", "her", "hers", "herself", "it", "its",
    "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those",
}

_DOMAIN_SIGNAL_TERMS: Dict[str, float] = {
    # JavaScript / programming concepts - weighted by importance
    "function": 3.0, "const": 2.5, "let": 2.5, "var": 2.0,
    "export": 3.5, "import": 3.5, "default": 2.5, "module": 3.0,
    "arrow": 3.0, "callback": 3.0, "async": 3.0, "await": 3.0, "promise": 3.0,
    "class": 3.0, "constructor": 3.5, "object": 2.5, "array": 2.5,
    "destructuring": 4.0, "spread": 3.5, "rest": 3.0,
    "map": 2.5, "filter": 2.5, "reduce": 2.5, "foreach": 2.0,
    "immutable": 3.5, "mutable": 3.0, "reference": 3.0, "primitive": 3.0,
    "scope": 3.0, "closure": 3.5, "hoisting": 3.0,
    "template": 2.5, "literal": 2.0, "string": 2.0,
    # Explanatory signal words
    "definition": 4.0, "syntax": 3.5, "example": 2.5, "note": 3.0,
    "important": 3.5, "remember": 3.0, "key": 3.0, "concept": 3.0,
    "allows": 2.5, "enables": 2.5, "provides": 2.0, "returns": 2.5,
    "creates": 2.0, "defines": 3.0, "declares": 2.5,
    "instead": 2.5, "rather": 2.0, "unlike": 2.5, "similar": 2.5,
    "must": 3.0, "cannot": 3.0, "should": 2.5, "avoid": 2.5,
    # React-specific terms
    "component": 4.0, "props": 3.5, "state": 3.5, "hook": 4.0, "usestate": 4.0,
    "useeffect": 4.0, "jsx": 4.0, "render": 3.0, "virtual": 3.0, "dom": 3.0,
    "lifecycle": 3.5, "context": 3.0, "redux": 3.0, "reducer": 3.0,
}

# ─────────────────────────────────────────────────────────────────────────────
# Definition / Explanation Pattern Detection (LLM-like understanding)
# ─────────────────────────────────────────────────────────────────────────────

_DEFINITION_PATTERNS: List[Tuple[re.Pattern, float]] = [
    # "X is Y" patterns (definitions)
    (re.compile(r"\b(is|are)\s+(a|an|the)?\s*\w+", re.I), 3.0),
    # "X allows/enables Y" patterns
    (re.compile(r"\b(allows?|enables?|lets?|permits?)\s+(us\s+)?(to\s+)?\w+", re.I), 2.5),
    # "X creates/defines/declares Y"
    (re.compile(r"\b(creates?|defines?|declares?|produces?|generates?)\s+", re.I), 2.5),
    # "used for/to" patterns
    (re.compile(r"\bused\s+(for|to)\s+\w+", re.I), 2.0),
    # "means that" / "which means"
    (re.compile(r"\b(means?\s+that|which\s+means?)\b", re.I), 3.0),
    # Contrast patterns ("unlike X", "instead of X", "rather than")
    (re.compile(r"\b(unlike|instead\s+of|rather\s+than|as\s+opposed\s+to)\b", re.I), 2.5),
    # Consequence patterns ("therefore", "thus", "hence", "so")
    (re.compile(r"\b(therefore|thus|hence|consequently|as\s+a\s+result)\b", re.I), 2.0),
    # Emphasis patterns ("importantly", "note that", "remember")
    (re.compile(r"\b(importantly|notably|crucially|note\s+that|remember\s+that)\b", re.I), 3.5),
    # Syntax explanation ("the syntax is", "syntax:")
    (re.compile(r"\b(syntax\s+(is|:)|the\s+syntax)\b", re.I), 4.0),
    # Rule patterns ("must", "cannot", "always", "never")
    (re.compile(r"\b(must\s+be|cannot\s+be|always\s+|never\s+|required\s+to)\b", re.I), 3.0),
    # Parenthetical clarification ("i.e.", "e.g.", "for example")
    (re.compile(r"\b(i\.?e\.?|e\.?g\.?|for\s+example|for\s+instance)\b", re.I), 2.0),
    # Colon-based definitions ("X: description")
    (re.compile(r"^[`\w\s]+\s*:\s*[_*]?.+[_*]?$", re.M), 3.5),
]

_WEAK_CONTENT_PATTERNS: List[Tuple[re.Pattern, float]] = [
    # Vague/filler phrases
    (re.compile(r"\b(basically|essentially|simply|just)\b", re.I), -1.0),
    # Overly casual
    (re.compile(r"\b(stuff|things|etc\.?|and\s+so\s+on)\b", re.I), -0.5),
    # Meta-commentary about the notes themselves
    (re.compile(r"\b(as\s+shown\s+above|see\s+below|as\s+mentioned)\b", re.I), -1.5),
]


def _definition_pattern_score(text: str) -> float:
    """Score based on presence of definition/explanation patterns."""
    score = 0.0
    for pattern, weight in _DEFINITION_PATTERNS:
        if pattern.search(text):
            score += weight
    for pattern, weight in _WEAK_CONTENT_PATTERNS:
        if pattern.search(text):
            score += weight  # weight is negative
    return score


# ─────────────────────────────────────────────────────────────────────────────
# Semantic Coherence & Context Awareness
# ─────────────────────────────────────────────────────────────────────────────

def _heading_relevance_score(text: str, heading: Optional[str]) -> float:
    """Score how well a paragraph relates to its section heading."""
    if not heading:
        return 0.0
    heading_tokens = set(_tokenize(heading)) - _STOPWORDS
    if not heading_tokens:
        return 0.0
    text_tokens = set(_tokenize(text))
    overlap = heading_tokens & text_tokens
    # Jaccard-like similarity with boost
    return len(overlap) / len(heading_tokens) * 5.0


def _code_proximity_score(
    block_index: int,
    all_blocks: List[Block],
    text: str,
) -> float:
    """Boost paragraphs that explain nearby code blocks."""
    # Look for code blocks within ±2 positions
    score = 0.0
    for offset in [-2, -1, 1, 2]:
        idx = block_index + offset
        if 0 <= idx < len(all_blocks):
            b = all_blocks[idx]
            if isinstance(b, CodeBlock):
                # Check if paragraph references code constructs
                code_tokens = set(_tokenize(b.code))
                text_tokens = set(_tokenize(text))
                shared = code_tokens & text_tokens - _STOPWORDS
                if shared:
                    # Closer = higher boost
                    proximity_mult = 2.0 if abs(offset) == 1 else 1.0
                    score += len(shared) * 0.5 * proximity_mult
    return min(score, 8.0)  # Cap the boost


def _redundancy_penalty(text: str, already_kept: List[str]) -> float:
    """Penalize text that's too similar to already-kept content."""
    if not already_kept:
        return 0.0
    text_tokens = set(_tokenize(text)) - _STOPWORDS
    if len(text_tokens) < 3:
        return 0.0
    max_overlap = 0.0
    for kept in already_kept:
        kept_tokens = set(_tokenize(kept)) - _STOPWORDS
        if not kept_tokens:
            continue
        overlap = len(text_tokens & kept_tokens) / max(len(text_tokens), len(kept_tokens))
        max_overlap = max(max_overlap, overlap)
    # High overlap = penalty
    if max_overlap > 0.7:
        return -5.0
    if max_overlap > 0.5:
        return -2.0
    return 0.0


def _structural_importance(text: str) -> float:
    """Detect structurally important content (lists within paragraphs, numbered items)."""
    score = 0.0
    # Numbered items embedded in text
    if re.search(r"\b[1-9]\.", text):
        score += 1.5
    # Contains arrow/pointer notation
    if "=>" in text or "->" in text:
        score += 2.0
    # Contains explicit comparison (vs, versus)
    if re.search(r"\bvs\.?\b|\bversus\b", text, re.I):
        score += 2.0
    return score


def _tokenize(text: str) -> List[str]:
    """Extract lowercase alphanumeric tokens."""
    return re.findall(r"[a-z][a-z0-9]*", text.lower())


def _compute_idf(documents: List[str]) -> Dict[str, float]:
    """Compute inverse document frequency for each term across documents."""
    n = len(documents)
    if n == 0:
        return {}
    doc_freq: Counter[str] = Counter()
    for doc in documents:
        tokens = set(_tokenize(doc))
        for tok in tokens:
            doc_freq[tok] += 1
    return {term: math.log((n + 1) / (freq + 1)) + 1 for term, freq in doc_freq.items()}


def _tf_idf_score(text: str, idf: Dict[str, float]) -> float:
    """Compute TF-IDF weighted sum for a piece of text."""
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    tf: Counter[str] = Counter(tokens)
    score = 0.0
    for term, count in tf.items():
        if term in _STOPWORDS:
            continue
        tf_val = 1 + math.log(count) if count > 0 else 0
        score += tf_val * idf.get(term, 1.0)
    return score


def _domain_signal_score(text: str) -> float:
    """Score based on presence of domain-specific signal terms."""
    tokens = _tokenize(text)
    return sum(_DOMAIN_SIGNAL_TERMS.get(t, 0.0) for t in tokens)


def _information_density(text: str) -> float:
    """Ratio of content words to total words (higher = denser)."""
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    content = [t for t in tokens if t not in _STOPWORDS]
    return len(content) / len(tokens)


def _inline_code_density(text: str) -> float:
    """Fraction of text inside backticks (signals technical content)."""
    code_chars = sum(len(m.group(1)) for m in re.finditer(r"`([^`]+)`", text))
    return code_chars / max(1, len(text))


def _sentence_count(text: str) -> int:
    return max(1, len(re.findall(r"[.!?]+", text)))


def _length_penalty(text: str) -> float:
    """Gentle penalty for very long or very short text."""
    n = len(text)
    if n < 40:
        return 0.7  # too terse
    if n > 500:
        return max(0.5, 1 - (n - 500) / 2000)
    return 1.0


@dataclass
class ScoringContext:
    """Precomputed corpus-level statistics for scoring."""
    idf: Dict[str, float] = field(default_factory=dict)
    total_paragraphs: int = 0
    all_blocks: List[Block] = field(default_factory=list)
    current_heading: Optional[str] = None
    already_kept: List[str] = field(default_factory=list)

    @classmethod
    def from_blocks(cls, blocks: List[Block]) -> "ScoringContext":
        docs = []
        for b in blocks:
            if isinstance(b, Paragraph):
                docs.append(b.text)
            elif isinstance(b, Bullet):
                docs.append(b.text)
        return cls(
            idf=_compute_idf(docs),
            total_paragraphs=len(docs),
            all_blocks=blocks,
        )

    def with_heading(self, heading: Optional[str]) -> "ScoringContext":
        """Return a copy with a specific heading context."""
        return ScoringContext(
            idf=self.idf,
            total_paragraphs=self.total_paragraphs,
            all_blocks=self.all_blocks,
            current_heading=heading,
            already_kept=self.already_kept,
        )

    def mark_kept(self, text: str) -> None:
        """Track kept content for redundancy detection."""
        self.already_kept.append(text)


@dataclass
class SummarizationStats:
    """Track what was kept vs. dropped during summarization."""
    total_paragraphs: int = 0
    kept_paragraphs: int = 0
    total_bullets: int = 0
    kept_bullets: int = 0
    total_codeblocks: int = 0
    kept_codeblocks: int = 0
    total_headings: int = 0
    kept_headings: int = 0
    # Score distributions
    para_scores: List[Tuple[float, str]] = field(default_factory=list)  # (score, preview)
    bullet_scores: List[Tuple[float, str]] = field(default_factory=list)
    dropped_para_scores: List[Tuple[float, str]] = field(default_factory=list)
    dropped_bullet_scores: List[Tuple[float, str]] = field(default_factory=list)

    def merge(self, other: "SummarizationStats") -> None:
        self.total_paragraphs += other.total_paragraphs
        self.kept_paragraphs += other.kept_paragraphs
        self.total_bullets += other.total_bullets
        self.kept_bullets += other.kept_bullets
        self.total_codeblocks += other.total_codeblocks
        self.kept_codeblocks += other.kept_codeblocks
        self.total_headings += other.total_headings
        self.kept_headings += other.kept_headings
        self.para_scores.extend(other.para_scores)
        self.bullet_scores.extend(other.bullet_scores)
        self.dropped_para_scores.extend(other.dropped_para_scores)
        self.dropped_bullet_scores.extend(other.dropped_bullet_scores)

    def compression_ratio(self) -> float:
        total = self.total_paragraphs + self.total_bullets
        kept = self.kept_paragraphs + self.kept_bullets
        if total == 0:
            return 1.0
        return kept / total

    def print_summary(self) -> None:
        print("\n" + "=" * 60)
        print("SUMMARIZATION STATISTICS")
        print("=" * 60)
        print(f"\nContent Filtering:")
        print(f"  Paragraphs: {self.kept_paragraphs}/{self.total_paragraphs} kept ({self._pct(self.kept_paragraphs, self.total_paragraphs)})")
        print(f"  Bullets:    {self.kept_bullets}/{self.total_bullets} kept ({self._pct(self.kept_bullets, self.total_bullets)})")
        print(f"  Code blocks:{self.kept_codeblocks}/{self.total_codeblocks} kept ({self._pct(self.kept_codeblocks, self.total_codeblocks)})")
        print(f"  Headings:   {self.kept_headings}/{self.total_headings} kept")
        print(f"\n  Overall compression: {self.compression_ratio():.1%} of text content retained")

        if self.para_scores:
            print(f"\nKept Paragraph Scores (top 5):")
            for score, preview in sorted(self.para_scores, key=lambda x: -x[0])[:5]:
                print(f"  [{score:6.2f}] {preview}")

        if self.dropped_para_scores:
            print(f"\nDropped Paragraph Scores (bottom 5):")
            for score, preview in sorted(self.dropped_para_scores, key=lambda x: x[0])[:5]:
                print(f"  [{score:6.2f}] {preview}")

        if self.bullet_scores:
            print(f"\nKept Bullet Scores (top 5):")
            for score, preview in sorted(self.bullet_scores, key=lambda x: -x[0])[:5]:
                print(f"  [{score:6.2f}] {preview}")

        if self.dropped_bullet_scores:
            print(f"\nDropped Bullet Scores (bottom 5):")
            for score, preview in sorted(self.dropped_bullet_scores, key=lambda x: x[0])[:5]:
                print(f"  [{score:6.2f}] {preview}")

        print("=" * 60 + "\n")

    def _pct(self, kept: int, total: int) -> str:
        if total == 0:
            return "N/A"
        return f"{100 * kept / total:.0f}%"


def score_text(
    text: str,
    ctx: ScoringContext,
    position_ratio: float = 0.5,
    block_index: int = -1,
) -> float:
    """
    Advanced composite scoring function combining multiple signals (LLM-inspired):
    
      1. TF-IDF relevance (corpus-level term importance)
      2. Domain signal terms (JavaScript/React vocabulary)
      3. Information density (content words ratio)
      4. Inline code density (technical content indicator)
      5. Definition pattern detection (identifies explanations)
      6. Heading relevance (semantic coherence with section)
      7. Code proximity (boosts paragraphs explaining nearby code)
      8. Structural importance (lists, arrows, comparisons)
      9. Redundancy penalty (avoids keeping duplicate info)
      10. Position bias (earlier content slightly preferred)
      11. Length penalty (too short/long gets penalized)
    """
    # Basic signals
    tfidf = _tf_idf_score(text, ctx.idf)
    domain = _domain_signal_score(text)
    density = _information_density(text)
    code_density = _inline_code_density(text)
    
    # LLM-like semantic signals
    definition = _definition_pattern_score(text)
    heading_rel = _heading_relevance_score(text, ctx.current_heading)
    
    # Context-aware signals
    code_prox = 0.0
    if block_index >= 0 and ctx.all_blocks:
        code_prox = _code_proximity_score(block_index, ctx.all_blocks, text)
    
    structural = _structural_importance(text)
    redundancy = _redundancy_penalty(text, ctx.already_kept)
    
    # Multipliers
    length_mult = _length_penalty(text)
    position_mult = 1.0 + 0.3 * (1 - position_ratio)  # up to 30% boost for early content

    # Weighted combination (tuned for educational/technical notes)
    raw = (
        tfidf * 1.0           # Base relevance
        + domain * 2.0        # Domain vocabulary boost
        + density * 8.0       # Information-dense content
        + code_density * 15.0 # Technical inline code
        + definition * 1.5    # Explanation patterns
        + heading_rel * 1.2   # Section coherence
        + code_prox * 1.0     # Code-explaining paragraphs
        + structural * 1.0    # Structural markers
        + redundancy          # Penalty (negative value)
    )
    return raw * length_mult * position_mult


def split_into_sections(blocks: List[Block]) -> List[Tuple[Optional[Heading], List[Block]]]:
    sections: List[Tuple[Optional[Heading], List[Block]]] = []
    current_heading: Optional[Heading] = None
    current_blocks: List[Block] = []

    for b in blocks:
        if isinstance(b, Heading):
            if current_heading is not None or current_blocks:
                sections.append((current_heading, current_blocks))
            current_heading = b
            current_blocks = []
        else:
            current_blocks.append(b)

    sections.append((current_heading, current_blocks))
    return sections


def summarize_section(
    heading: Optional[Heading],
    blocks: List[Block],
    *,
    max_bullets: int,
    max_paragraphs: int,
    keep_code: bool,
    ctx: ScoringContext,
) -> Tuple[List[Block], SummarizationStats]:
    out: List[Block] = []
    stats = SummarizationStats()

    # Create heading-aware context for this section
    heading_text = heading.text if heading else None
    section_ctx = ctx.with_heading(heading_text)

    if heading is not None:
        out.append(heading)
        stats.total_headings += 1
        stats.kept_headings += 1

    bullets: List[Bullet] = [b for b in blocks if isinstance(b, Bullet)]
    paragraphs: List[Paragraph] = [p for p in blocks if isinstance(p, Paragraph)]
    codeblocks: List[CodeBlock] = [c for c in blocks if isinstance(c, CodeBlock)]

    n_para = len(paragraphs)
    n_bullet = len(bullets)
    n_code = len(codeblocks)

    stats.total_paragraphs = n_para
    stats.total_bullets = n_bullet
    stats.total_codeblocks = n_code

    # Find block indices in the global block list for code proximity scoring
    def find_block_index(block: Block) -> int:
        try:
            return section_ctx.all_blocks.index(block)
        except ValueError:
            return -1

    # Score paragraphs with position awareness and context
    scored_para: List[Tuple[float, int, Paragraph]] = [
        (
            score_text(
                p.text,
                section_ctx,
                position_ratio=i / max(1, n_para),
                block_index=find_block_index(p),
            ),
            i,
            p,
        )
        for i, p in enumerate(paragraphs)
    ]
    scored_para.sort(key=lambda x: (-x[0], x[1]))
    
    # Apply redundancy-aware selection
    kept_para_ids: set = set()
    for score, idx, p in scored_para:
        if len(kept_para_ids) >= max_paragraphs:
            break
        # Check redundancy against already-kept content
        redundancy = _redundancy_penalty(p.text, section_ctx.already_kept)
        if redundancy < -3.0:  # Too redundant, skip
            continue
        kept_para_ids.add(id(p))
        section_ctx.mark_kept(p.text)
    
    keep_para = {pid: True for pid in kept_para_ids}

    # Record scores for kept vs dropped paragraphs
    for score, _, p in scored_para:
        preview = p.text[:60].replace("\n", " ") + ("..." if len(p.text) > 60 else "")
        if id(p) in kept_para_ids:
            stats.para_scores.append((score, preview))
        else:
            stats.dropped_para_scores.append((score, preview))

    # Score bullets with context awareness
    scored_bullets: List[Tuple[float, int, Bullet]] = [
        (
            score_text(
                b.text,
                section_ctx,
                position_ratio=i / max(1, n_bullet),
                block_index=find_block_index(b),
            ),
            i,
            b,
        )
        for i, b in enumerate(bullets)
    ]
    scored_bullets.sort(key=lambda x: (-x[0], x[1]))
    
    # Apply redundancy-aware selection for bullets
    kept_bullet_ids: set = set()
    for score, idx, b in scored_bullets:
        if len(kept_bullet_ids) >= max_bullets:
            break
        redundancy = _redundancy_penalty(b.text, section_ctx.already_kept)
        if redundancy < -3.0:
            continue
        kept_bullet_ids.add(id(b))
        section_ctx.mark_kept(b.text)
    
    keep_bullet = {bid: True for bid in kept_bullet_ids}

    # Record scores for kept vs dropped bullets
    for score, _, b in scored_bullets:
        preview = b.text[:60].replace("\n", " ") + ("..." if len(b.text) > 60 else "")
        if id(b) in kept_bullet_ids:
            stats.bullet_scores.append((score, preview))
        else:
            stats.dropped_bullet_scores.append((score, preview))

    # Preserve original ordering for readability
    for b in blocks:
        if isinstance(b, Paragraph):
            if keep_para.get(id(b), False):
                out.append(b)
                stats.kept_paragraphs += 1
        elif isinstance(b, Bullet):
            if keep_bullet.get(id(b), False):
                out.append(b)
                stats.kept_bullets += 1
        elif isinstance(b, CodeBlock):
            if keep_code:
                out.append(b)
                stats.kept_codeblocks += 1

    return out, stats


def build_takeaways(
    sections: List[Tuple[Optional[Heading], List[Block]]],
    max_items: int,
    ctx: ScoringContext,
) -> List[str]:
    """Extract the highest-scoring short statements across the entire document."""
    candidates: List[Tuple[float, str, Optional[str]]] = []  # (score, text, heading)

    all_blocks = [b for _, blks in sections for b in blks]
    n_total = len(all_blocks)

    idx = 0
    for heading, blocks in sections:
        heading_text = heading.text if heading else None
        section_ctx = ctx.with_heading(heading_text)
        
        for b in blocks:
            pos_ratio = idx / max(1, n_total)
            block_idx = idx
            idx += 1
            
            if isinstance(b, Bullet):
                text = b.text.strip()
                if text and len(text) <= 200:
                    score = score_text(text, section_ctx, pos_ratio, block_index=block_idx)
                    candidates.append((score, text, heading_text))
            elif isinstance(b, Paragraph):
                text = b.text.strip()
                # Extract first sentence for takeaway
                first = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0].strip()
                if first and len(first) <= 180:
                    score = score_text(first, section_ctx, pos_ratio, block_index=block_idx)
                    candidates.append((score, first, heading_text))

    # De-duplicate by normalized key
    seen: Set[str] = set()
    deduped: List[Tuple[float, str]] = []
    for score, text, _ in sorted(candidates, key=lambda x: -x[0]):
        key = re.sub(r"\s+", " ", text.lower())
        key = re.sub(r"[^a-z0-9 ]", "", key)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((score, text))
        if len(deduped) >= max_items:
            break

    return [t for _, t in deduped]


def latex_preamble(title: str, author: str, doc_date: str, use_minted: bool = True) -> str:
    """Generate LaTeX preamble with syntax highlighting support.
    
    Uses minted by default for rich syntax highlighting (requires --shell-escape).
    Falls back to listings if minted is unavailable.
    """
    escaped_title = _latex_escape(title)
    escaped_author = _latex_escape(author)
    escaped_date = _latex_escape(doc_date)

    minted_preamble = f"""\\documentclass[11pt]{{article}}
\\usepackage[a4paper,margin=1in]{{geometry}}
\\usepackage[T1]{{fontenc}}
\\usepackage{{lmodern}}
\\usepackage{{xcolor}}
\\usepackage{{hyperref}}
\\usepackage{{enumitem}}
\\usepackage{{minted}}

\\hypersetup{{
    colorlinks=true,
    linkcolor=blue!70!black,
    urlcolor=blue!70!black,
    citecolor=blue!70!black
}}

% Minted configuration for syntax highlighting
\\usemintedstyle{{tango}}
\\setminted{{
    fontsize=\\small,
    frame=single,
    framesep=2mm,
    baselinestretch=1.1,
    linenos=true,
    numbersep=5pt,
    breaklines=true,
    breakanywhere=true,
    tabsize=2
}}

% Custom colors for code background
\\definecolor{{codebg}}{{RGB}}{{248,248,248}}
\\setminted{{bgcolor=codebg}}

\\title{{{escaped_title}}}
\\author{{{escaped_author}}}
\\date{{{escaped_date}}}

\\begin{{document}}
\\maketitle
"""

    listings_preamble = f"""\\documentclass[11pt]{{article}}
\\usepackage[a4paper,margin=1in]{{geometry}}
\\usepackage[T1]{{fontenc}}
\\usepackage{{lmodern}}
\\usepackage{{xcolor}}
\\usepackage{{hyperref}}
\\usepackage{{enumitem}}
\\usepackage{{listings}}

\\hypersetup{{
    colorlinks=true,
    linkcolor=blue!70!black,
    urlcolor=blue!70!black,
    citecolor=blue!70!black
}}

% Enhanced listings colors for syntax highlighting
\\definecolor{{codebg}}{{RGB}}{{248,248,248}}
\\definecolor{{codegreen}}{{RGB}}{{40,160,40}}
\\definecolor{{codepurple}}{{RGB}}{{160,32,240}}
\\definecolor{{codeblue}}{{RGB}}{{0,80,180}}
\\definecolor{{codeorange}}{{RGB}}{{200,80,0}}
\\definecolor{{codegray}}{{RGB}}{{100,100,100}}

\\lstdefinestyle{{notes}}{{
    backgroundcolor=\\color{{codebg}},
    basicstyle=\\ttfamily\\small,
    breaklines=true,
    breakatwhitespace=false,
    frame=single,
    framerule=0.4pt,
    rulecolor=\\color{{black!30}},
    numbers=left,
    numberstyle=\\tiny\\color{{codegray}},
    xleftmargin=2.5em,
    framexleftmargin=2em,
    tabsize=2,
    showstringspaces=false,
    commentstyle=\\color{{codegreen}}\\itshape,
    keywordstyle=\\color{{codeblue}}\\bfseries,
    stringstyle=\\color{{codeorange}},
    emphstyle=\\color{{codepurple}},
    morekeywords={{const,let,var,function,return,if,else,for,while,class,export,import,default,from,async,await,try,catch,throw,new,this,super,extends,static,get,set,of,in}},
    morestring=[b]",
    morestring=[b]',
    morestring=[b]`,
    morecomment=[l]{{//}},
    morecomment=[s]{{/*}}{{*/}},
}}

\\title{{{escaped_title}}}
\\author{{{escaped_author}}}
\\date{{{escaped_date}}}

\\begin{{document}}
\\maketitle
"""

    return minted_preamble if use_minted else listings_preamble


def latex_heading(h: Heading) -> str:
    text = _inline_code_to_latex(_md_links_to_latex(_strip_md_emphasis(h.text)))
    # Map Markdown heading levels to LaTeX sectioning.
    if h.level <= 1:
        return "\\section{" + text + "}"
    if h.level == 2:
        return "\\subsection{" + text + "}"
    if h.level == 3:
        return "\\subsubsection{" + text + "}"
    return "\\paragraph{" + text + "}"


def latex_paragraph(p: Paragraph) -> str:
    text = _strip_md_emphasis(p.text)
    text = _md_links_to_latex(text)
    text = _inline_code_to_latex(text)
    # Preserve line breaks inside paragraphs as LaTeX line breaks
    text = text.replace("\n", "\\\\\n")
    return text + "\n"


def _map_language_for_minted(lang: str) -> str:
    """Map markdown language hints to minted lexer names."""
    l = (lang or "").lower().strip()
    mapping = {
        "js": "javascript",
        "javascript": "javascript",
        "jsx": "jsx",
        "ts": "typescript",
        "typescript": "typescript",
        "tsx": "tsx",
        "html": "html",
        "css": "css",
        "scss": "scss",
        "bash": "bash",
        "sh": "bash",
        "shell": "bash",
        "zsh": "bash",
        "json": "json",
        "python": "python",
        "py": "python",
        "sql": "sql",
        "yaml": "yaml",
        "yml": "yaml",
        "xml": "xml",
        "markdown": "markdown",
        "md": "markdown",
        "c": "c",
        "cpp": "cpp",
        "c++": "cpp",
        "java": "java",
        "go": "go",
        "rust": "rust",
        "ruby": "ruby",
        "php": "php",
    }
    return mapping.get(l, "text")


def sanitize_code_for_listings(code: str) -> str:
    # The LaTeX listings package can be very picky about Unicode, especially
    # box-drawing characters and some punctuation. We keep this conservative
    # and map common offenders to ASCII.
    replacements = {
        "├": "|",
        "└": "|",
        "│": "|",
        "─": "-",
        "→": "->",
        "⇒": "=>",
        "“": '"',
        "”": '"',
        "’": "'",
        "‘": "'",
    }

    for old, new in replacements.items():
        code = code.replace(old, new)

    # Replace remaining non-ASCII chars with a safe placeholder.
    safe_chars: List[str] = []
    for ch in code:
        if ch in ("\n", "\t"):
            safe_chars.append(ch)
            continue
        if 32 <= ord(ch) < 127:
            safe_chars.append(ch)
        else:
            safe_chars.append("?")
    return "".join(safe_chars)


def latex_codeblock(c: CodeBlock, use_minted: bool = True) -> str:
    """Render code block with syntax highlighting."""
    clean_code = sanitize_code_for_listings(c.code)
    
    if use_minted:
        lexer = _map_language_for_minted(c.language)
        return (
            f"\\begin{{minted}}{{{lexer}}}\n"
            + clean_code
            + "\n\\end{minted}\n"
        )
    else:
        # Fallback to listings
        return (
            "\\begin{lstlisting}[style=notes]\n"
            + clean_code
            + "\n\\end{lstlisting}\n"
        )


def latex_from_blocks(blocks: List[Block], use_minted: bool = True) -> str:
    out: List[str] = []

    in_list = False

    def close_list() -> None:
        nonlocal in_list
        if in_list:
            out.append("\\end{itemize}\n")
            in_list = False

    for b in blocks:
        if isinstance(b, Heading):
            close_list()
            out.append(latex_heading(b) + "\n")
        elif isinstance(b, Paragraph):
            close_list()
            out.append(latex_paragraph(b) + "\n")
        elif isinstance(b, CodeBlock):
            close_list()
            out.append(latex_codeblock(b, use_minted=use_minted) + "\n")
        elif isinstance(b, Bullet):
            if not in_list:
                out.append("\\begin{itemize}[leftmargin=*]\n")
                in_list = True
            text = _strip_md_emphasis(b.text)
            text = _md_links_to_latex(text)
            text = _inline_code_to_latex(text)
            out.append("\\item " + text + "\n")

    close_list()
    return "".join(out)


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Convert Markdown notes into a summarized LaTeX document.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s Notes.md --title "My Summary"
  %(prog)s Notes.md --verbose                    # Show scoring statistics
  %(prog)s Notes.md --compile                    # Auto-compile to PDF
  %(prog)s Notes.md --compile --no-minted        # Compile without minted
""",
    )
    parser.add_argument("input", type=str, help="Path to the .md file")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="",
        help="Output folder path (default: '<ParentFolder> - Summary/' next to input)",
    )
    parser.add_argument("--title", type=str, default="", help="Document title (default: derived from folder name)")
    parser.add_argument("--author", type=str, default="")
    parser.add_argument("--date", type=str, default=str(date.today()))
    parser.add_argument("--max-bullets", type=int, default=10, help="Max bullets kept per section")
    parser.add_argument("--max-paragraphs", type=int, default=2, help="Max paragraphs kept per section")
    parser.add_argument("--no-code", action="store_true", help="Drop code blocks from output")
    parser.add_argument("--takeaways", type=int, default=8, help="Number of top takeaways at the top")
    parser.add_argument(
        "--no-minted",
        action="store_true",
        help="Use listings instead of minted for code (no --shell-escape needed)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed scoring statistics to verify summarization",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Auto-compile to PDF after generating .tex (requires pdflatex)",
    )

    args = parser.parse_args(argv)

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 2

    # Derive folder name from parent directory
    parent_folder_name = input_path.parent.name
    summary_folder_name = f"{parent_folder_name} - Summary"

    # Determine output folder
    if args.output:
        output_folder = Path(args.output).resolve()
    else:
        output_folder = input_path.parent / summary_folder_name

    # Create folder structure
    src_folder = output_folder / "src"
    src_folder.mkdir(parents=True, exist_ok=True)

    # Derive title if not provided
    title = args.title if args.title else f"{parent_folder_name} (Summary)"

    md = input_path.read_text(encoding="utf-8")
    blocks = parse_markdown(md)
    sections = split_into_sections(blocks)

    # Build corpus-level scoring context
    all_blocks = [b for _, blks in sections for b in blks]
    ctx = ScoringContext.from_blocks(all_blocks)

    use_minted = not args.no_minted

    summarized: List[Block] = []
    global_stats = SummarizationStats()

    # Add top takeaways section
    takeaways = build_takeaways(sections, max_items=max(0, args.takeaways), ctx=ctx)
    if takeaways:
        summarized.append(Heading(level=2, text="Key Takeaways"))
        for t in takeaways:
            summarized.append(Bullet(indent=0, text=t))

    for heading, sec_blocks in sections:
        section_blocks, section_stats = summarize_section(
            heading,
            sec_blocks,
            max_bullets=max(0, args.max_bullets),
            max_paragraphs=max(0, args.max_paragraphs),
            keep_code=not args.no_code,
            ctx=ctx,
        )
        summarized.extend(section_blocks)
        global_stats.merge(section_stats)

    # Generate .tex content
    tex = latex_preamble(title, args.author, args.date, use_minted=use_minted)
    tex += latex_from_blocks(summarized, use_minted=use_minted)
    tex += "\\end{document}\n"

    # Write .tex to src folder
    tex_filename = f"{summary_folder_name}.tex"
    tex_path = src_folder / tex_filename
    tex_path.write_text(tex, encoding="utf-8")
    print(f"Wrote LaTeX source to: {tex_path}")

    # Print verbose stats if requested
    if args.verbose:
        global_stats.print_summary()

    # Compile to PDF if requested
    if args.compile:
        print(f"\nCompiling PDF...")
        compile_cmd = ["pdflatex", "-interaction=nonstopmode"]
        if use_minted:
            compile_cmd.append("-shell-escape")
        compile_cmd.append(tex_filename)

        # Set environment for minted v3 compatibility
        compile_env = os.environ.copy()
        compile_env["TEXMF_OUTPUT_DIRECTORY"] = str(src_folder)

        # Run pdflatex twice for references
        for run in range(2):
            result = subprocess.run(
                compile_cmd,
                cwd=src_folder,
                capture_output=True,
                text=True,
                env=compile_env,
            )
            if result.returncode != 0 and run == 1:
                print(f"Warning: pdflatex returned non-zero exit code", file=sys.stderr)
                if args.verbose:
                    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)

        # Move PDF to parent folder (so it appears first)
        pdf_name = tex_filename.replace(".tex", ".pdf")
        src_pdf = src_folder / pdf_name
        dest_pdf = output_folder / pdf_name

        if src_pdf.exists():
            shutil.move(str(src_pdf), str(dest_pdf))
            print(f"Output PDF: {dest_pdf}")
        else:
            print(f"Warning: PDF not generated. Check {src_folder / 'Notes_summary.log'} for errors.", file=sys.stderr)

    print(f"\nOutput folder: {output_folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
