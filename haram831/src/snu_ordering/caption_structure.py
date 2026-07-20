"""Rule-based, confidence-aware temporal structure extraction for captions.

The extractor deliberately treats punctuation as weak supervision rather than
forcing every caption into four events.  It preserves the original caption and
returns a variable number of candidate segments plus typed boundaries that can
later be rendered as prompt hints or used for bucketed evaluation.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Literal

BoundaryKind = Literal["NEXT", "BEFORE_AFTER", "OVERLAP", "STRONG", "WEAK"]


@dataclass(frozen=True)
class TemporalBoundary:
    """A relation marker between two non-empty candidate event segments."""

    kind: BoundaryKind
    marker: str
    confidence: float
    char_start: int
    char_end: int


@dataclass(frozen=True)
class CaptionStructure:
    """Original caption with variable-length event candidates and relations."""

    original: str
    segments: tuple[str, ...]
    boundaries: tuple[TemporalBoundary, ...]

    def __post_init__(self) -> None:
        if not self.original.strip():
            raise ValueError("caption must contain non-whitespace text")
        if not self.segments:
            raise ValueError("caption structure must contain at least one segment")
        if len(self.boundaries) != len(self.segments) - 1:
            raise ValueError("boundaries must connect every adjacent segment")

    @property
    def event_count(self) -> int:
        """Candidate count when all boundaries, including bare commas, are used."""

        return len(self.segments)

    def confident_event_count(self, threshold: float = 0.7) -> int:
        """Event count implied only by boundaries at or above ``threshold``."""

        if not 0.0 <= threshold <= 1.0:
            raise ValueError("confidence threshold must be between 0 and 1")
        return 1 + sum(boundary.confidence >= threshold for boundary in self.boundaries)


# Composite punctuation + discourse markers must be matched before bare commas
# and semicolons.  This turns "; then," into one NEXT boundary instead of three
# adjacent markers.
_BOUNDARY_RE = re.compile(
    r"[,;]\s*(?:(?:and\s+)?(?:then|next|afterwards?|afterward|subsequently|"
    r"finally|meanwhile))\s*,?"
    r"|\b(?:followed\s+by|and\s+then|then|next|afterwards?|afterward|"
    r"subsequently|finally|meanwhile|before|after|while|as)\b"
    r"|[;,]",
    flags=re.IGNORECASE,
)
_PUNCTUATION_SPLIT_RE = re.compile(r"[,;]")

_NEXT_RE = re.compile(
    r"\b(?:followed\s+by|then|next|afterwards?|afterward|subsequently|finally)\b",
    flags=re.IGNORECASE,
)
_BEFORE_AFTER_RE = re.compile(r"\b(?:before|after)\b", flags=re.IGNORECASE)
_OVERLAP_RE = re.compile(r"\b(?:while|as|meanwhile)\b", flags=re.IGNORECASE)

_KIND_PRIORITY: dict[BoundaryKind, int] = {
    "WEAK": 0,
    "STRONG": 1,
    "OVERLAP": 2,
    "BEFORE_AFTER": 3,
    "NEXT": 4,
}


def _classify_marker(match: re.Match[str]) -> TemporalBoundary:
    marker = match.group(0)
    if _NEXT_RE.search(marker):
        kind: BoundaryKind = "NEXT"
        confidence = 1.0 if ";" in marker else 0.95
    elif _BEFORE_AFTER_RE.search(marker):
        kind = "BEFORE_AFTER"
        confidence = 0.9
    elif _OVERLAP_RE.search(marker):
        kind = "OVERLAP"
        confidence = 0.8
    elif ";" in marker:
        kind = "STRONG"
        confidence = 0.85
    else:
        kind = "WEAK"
        confidence = 0.35
    return TemporalBoundary(
        kind=kind,
        marker=marker.strip(),
        confidence=confidence,
        char_start=match.start(),
        char_end=match.end(),
    )


def _prefer_boundary(
    current: TemporalBoundary | None,
    candidate: TemporalBoundary,
) -> TemporalBoundary:
    """Collapse adjacent markers, retaining the most informative relation."""

    if current is None:
        return candidate
    current_key = (current.confidence, _KIND_PRIORITY[current.kind])
    candidate_key = (candidate.confidence, _KIND_PRIORITY[candidate.kind])
    return candidate if candidate_key > current_key else current


def _clean_segment(value: str) -> str:
    return value.strip(" \t\r\n,;")


def extract_caption_structure(caption: str) -> CaptionStructure:
    """Extract typed temporal hints without forcing an event count of four.

    Bare commas are retained as low-confidence boundaries.  Explicit temporal
    connectives and semicolons receive higher confidence.  Consecutive markers
    such as ``; then,`` are collapsed and never create empty segments.
    """

    if not isinstance(caption, str) or not caption.strip():
        raise ValueError("caption must contain non-whitespace text")

    segments: list[str] = []
    boundaries: list[TemporalBoundary] = []
    pending: TemporalBoundary | None = None
    cursor = 0

    for match in _BOUNDARY_RE.finditer(caption):
        segment = _clean_segment(caption[cursor : match.start()])
        if segment:
            if segments and pending is not None:
                boundaries.append(pending)
            segments.append(segment)
            pending = None
        pending = _prefer_boundary(pending, _classify_marker(match))
        cursor = match.end()

    tail = _clean_segment(caption[cursor:])
    if tail:
        if segments and pending is not None:
            boundaries.append(pending)
        segments.append(tail)

    # A caption made only of marker-like words should still be auditable rather
    # than producing an invalid zero-segment structure.
    if not segments:
        segments.append(caption.strip())

    return CaptionStructure(
        original=caption,
        segments=tuple(segments),
        boundaries=tuple(boundaries),
    )


def event_count_bucket(count: int) -> str:
    """Return the stable EDA bucket label for a positive event count."""

    if count < 1:
        raise ValueError("event count must be positive")
    return str(count) if count <= 4 else "5+"


def extract_punctuation_candidates(caption: str) -> tuple[str, ...]:
    """Split A1 action candidates only at commas and semicolons.

    Temporal words such as ``then`` and ``while`` remain in their surrounding
    candidate.  This keeps A1 distinct from the relation-aware A2 extractor.
    """

    if not isinstance(caption, str) or not caption.strip():
        raise ValueError("caption must contain non-whitespace text")
    candidates = tuple(
        part.strip()
        for part in _PUNCTUATION_SPLIT_RE.split(caption)
        if part.strip()
    )
    return candidates or (caption.strip(),)


def render_punctuation_hints(caption: str) -> str:
    """Render the A1 caption context while preserving the complete original."""

    candidates = extract_punctuation_candidates(caption)
    lines = [
        "Original caption:",
        caption.strip(),
        "",
        "Approximate event hints:",
    ]
    lines.extend(
        f"[Event {index}] {candidate}"
        for index, candidate in enumerate(candidates, start=1)
    )
    lines.extend(
        [
            "",
            "The event hints are approximate and may not correspond one-to-one "
            "with the four images. Use the original caption when the hints are ambiguous.",
        ]
    )
    return "\n".join(lines)


def _keep_boundary(*, seed: str, boundary_index: int, dropout: float) -> bool:
    """Return a stable dropout decision independent of process hash randomization."""

    digest = hashlib.sha256(f"{seed}:{boundary_index}".encode("utf-8")).digest()
    sample = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return sample >= dropout


def _render_relation_edge(
    left: str, right: str, boundary: TemporalBoundary
) -> str:
    marker = boundary.marker.casefold()
    if boundary.kind == "NEXT":
        return f"[NEXT] {left} -> {right}"
    if boundary.kind == "BEFORE_AFTER":
        if re.search(r"\bafter\b", marker):
            return f"[AFTER] {left} occurs after {right}"
        return f"[BEFORE] {left} -> {right}"
    if boundary.kind == "OVERLAP":
        return f"[OVERLAP] {left} || {right}"
    if boundary.kind == "STRONG":
        return f"[SEQUENCE] {left} -> {right}"
    return f"[WEAK] {left} -> {right}"


def render_relation_hints(
    caption: str,
    *,
    confidence_threshold: float = 0.7,
    boundary_dropout: float = 0.0,
    dropout_seed: str | int | None = None,
) -> str:
    """Render A2 relation edges while always preserving the original caption.

    Dropout is applied only when the caller passes a positive rate. The caller
    disables it for validation and inference. A stable seed makes training
    prompts reproducible across interrupted and resumed runs.
    """

    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1")
    if not 0.0 <= boundary_dropout < 1.0:
        raise ValueError("boundary_dropout must be in [0, 1)")
    if boundary_dropout > 0.0 and dropout_seed is None:
        raise ValueError("dropout_seed is required when boundary_dropout is enabled")

    structure = extract_caption_structure(caption)
    edges: list[str] = []
    for index, boundary in enumerate(structure.boundaries):
        if boundary.confidence < confidence_threshold:
            continue
        if boundary_dropout > 0.0 and not _keep_boundary(
            seed=str(dropout_seed), boundary_index=index, dropout=boundary_dropout
        ):
            continue
        edges.append(
            _render_relation_edge(
                structure.segments[index], structure.segments[index + 1], boundary
            )
        )

    lines = ["Original caption:", structure.original.strip()]
    if edges:
        lines.extend(["", "Approximate action relations:", *edges])
        lines.extend(
            [
                "",
                "The relations are approximate. Use the original caption when "
                "a relation is ambiguous.",
            ]
        )
    return "\n".join(lines)
