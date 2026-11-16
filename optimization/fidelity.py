"""Multi-fidelity source representations for budget-aware evidence selection."""

from __future__ import annotations

import ast
import copy
import io
import tokenize
from dataclasses import dataclass
from typing import Iterable

from optigraph.optimization.models import RankedEvidenceItem
from optigraph.retrieval.models import SourceSnippet


@dataclass(frozen=True)
class FidelityVariant:
    """One representation of an evidence item at a known information level."""

    name: str
    utility_factor: float
    candidate: RankedEvidenceItem


_FIDELITY_LEVELS = (
    ("signature", 0.55),
    ("reduced", 0.85),
    ("full", 1.0),
)


def _strip_comments(source: str) -> str:
    """Remove Python COMMENT tokens without damaging strings or line structure."""
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        kept = (tok for tok in tokens if tok.type != tokenize.COMMENT)
        return tokenize.untokenize(kept).strip()
    except (tokenize.TokenError, IndentationError):
        # For incomplete/non-Python snippets, retain source rather than risk
        # deleting '#' characters that may be meaningful in another language.
        return source.strip()


def _signature_and_docstring(source: str) -> str:
    """Extract the first top-level class/function signature and its docstring."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        lines = source.splitlines()
        declaration = next(
            (i for i, line in enumerate(lines) if line.lstrip().startswith(
                ("class ", "def ", "async def ")
            )),
            None,
        )
        if declaration is None:
            return lines[0].strip() if lines else ""
        return lines[declaration].rstrip()

    target = next(
        (node for node in tree.body if isinstance(
            node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        )),
        None,
    )
    if target is None:
        return next((line.strip() for line in source.splitlines() if line.strip()), "")

    start = min((d.lineno for d in getattr(target, "decorator_list", [])), default=target.lineno)
    body_start = target.body[0].lineno if target.body else target.end_lineno or target.lineno
    lines = source.splitlines()
    signature = "\n".join(lines[start - 1 : max(start, body_start - 1)]).rstrip()

    docstring = ast.get_docstring(target, clean=False)
    if docstring:
        first = target.body[0]
        doc_expr = ast.get_source_segment(source, first)
        if doc_expr:
            signature = f"{signature}\n{doc_expr}".strip()
    return signature


def build_fidelity_variants(candidate: RankedEvidenceItem, estimator) -> list[FidelityVariant]:
    """Return signature/docstring, comment-reduced, and full candidate variants.

    The original candidate is never mutated. Non-source candidates and unresolved
    snippets have a single full-fidelity representation.
    """
    snippet = candidate.snippet
    if snippet is None or snippet.status != "resolved" or not snippet.content:
        return [FidelityVariant("full", 1.0, copy.deepcopy(candidate))]

    source = snippet.content
    signature = _signature_and_docstring(source)
    reduced = _strip_comments(source)
    representations: Iterable[tuple[str, float, str]] = (
        ("signature", 0.55, signature),
        ("reduced", 0.85, reduced),
        ("full", 1.0, source),
    )

    variants: list[FidelityVariant] = []
    seen: set[str] = set()
    for name, factor, content in representations:
        content = content.strip()
        if not content or content in seen:
            continue
        seen.add(content)
        item = copy.deepcopy(candidate)
        item.snippet = SourceSnippet(
            file_path=snippet.file_path,
            start_line=snippet.start_line,
            end_line=snippet.end_line,
            content=content,
            status=snippet.status,
            node_id=snippet.node_id,
        )
        item.provenance = dict(item.provenance)
        item.provenance["snippet_fidelity"] = name
        item.estimated_tokens = estimator.count_snippet_tokens(item.snippet)
        variants.append(FidelityVariant(name, factor, item))

    return variants or [FidelityVariant("full", 1.0, copy.deepcopy(candidate))]
