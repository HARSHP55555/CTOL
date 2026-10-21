"""Source-code snippet extraction and file boundary safety."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Sequence

from optigraph.models.canonical import OptiGraphNode
from optigraph.retrieval.models import RetrievalWarning, RetrievedNode, SourceSnippet


def parse_source_location(location_str: Optional[str]) -> tuple[Optional[int], Optional[int]]:
    """Parse canonical source_location string into 1-indexed (start_line, end_line).

    Supported patterns:
        - "L10-L45", "L10:L45", "L10-45" -> (10, 45)
        - "L10" -> (10, 10)
        - "10-45" -> (10, 45)
        - "10" -> (10, 10)

    Args:
        location_str: Raw location string or None.

    Returns:
        Tuple of (start_line, end_line) or (None, None).
    """
    if not location_str or not isinstance(location_str, str):
        return None, None

    cleaned = location_str.strip()

    # Pattern: L<start>-L<end> or L<start>:L<end> or L<start>-<end> or <start>-<end>
    range_match = re.search(r"L?(\d+)[\:\-]L?(\d+)", cleaned, re.IGNORECASE)
    if range_match:
        try:
            start = int(range_match.group(1))
            end = int(range_match.group(2))
            if start <= end:
                return start, end
            return end, start
        except ValueError:
            pass

    # Pattern: L<start> or <start>
    single_match = re.search(r"L?(\d+)", cleaned, re.IGNORECASE)
    if single_match:
        try:
            val = int(single_match.group(1))
            return val, val
        except ValueError:
            pass

    return None, None


class SourceRetriever:
    """Safely extracts real source text for graph nodes."""

    def __init__(
        self,
        repo_root: Optional[Path | str] = None,
        context_lines: int = 5,
        max_snippet_lines: int = 100,
    ) -> None:
        """Initialize the source snippet retriever.

        Args:
            repo_root: Base project directory for safely resolving relative paths.
            context_lines: Number of surrounding lines to include when only a single line is known.
            max_snippet_lines: Maximum lines allowed in a single snippet.
        """
        self.repo_root = Path(repo_root or ".").resolve()
        self.context_lines = max(0, context_lines)
        self.max_snippet_lines = max(1, max_snippet_lines)

    def resolve_path(self, relative_path: str) -> tuple[Optional[Path], Optional[RetrievalWarning]]:
        """Safely resolve relative source file path against repo_root.

        Args:
            relative_path: Source file path recorded in canonical node.

        Returns:
            Tuple of (resolved_path, warning_if_unsafe_or_missing).
        """
        if not relative_path:
            return None, RetrievalWarning(
                category="empty_source_path",
                message="Node has no source_file recorded",
            )

        candidate = Path(relative_path)
        if candidate.is_absolute():
            resolved = candidate.resolve()
        else:
            resolved = (self.repo_root / candidate).resolve()

        # Path traversal boundary check: ensure resolved path is inside repo_root
        try:
            resolved.relative_to(self.repo_root)
        except ValueError:
            return None, RetrievalWarning(
                category="unsafe_path_traversal",
                message=f"Path '{relative_path}' resolves outside repository root '{self.repo_root}'",
                file_path=relative_path,
            )

        if not resolved.exists():
            return resolved, RetrievalWarning(
                category="missing_source_file",
                message=f"Source file '{relative_path}' does not exist on disk at '{resolved}'",
                file_path=relative_path,
            )

        if not resolved.is_file():
            return resolved, RetrievalWarning(
                category="not_a_file",
                message=f"Source path '{relative_path}' is not a regular file",
                file_path=relative_path,
            )

        return resolved, None

    def retrieve_node_snippet(
        self,
        node: OptiGraphNode,
    ) -> tuple[Optional[SourceSnippet], Optional[RetrievalWarning]]:
        """Retrieve real source code snippet for a single canonical node.

        Args:
            node: Canonical OptiGraphNode.

        Returns:
            Tuple of (SourceSnippet or None, RetrievalWarning or None).
        """
        if not node.source_file:
            return None, RetrievalWarning(
                category="no_source_file",
                message=f"Node '{node.id}' has no source_file metadata",
                node_id=node.id,
            )

        resolved_path, path_warn = self.resolve_path(node.source_file)
        if path_warn is not None and (resolved_path is None or not resolved_path.is_file()):
            path_warn.node_id = node.id
            return (
                SourceSnippet(
                    file_path=node.source_file,
                    start_line=None,
                    end_line=None,
                    content=None,
                    status=path_warn.category,
                    node_id=node.id,
                ),
                path_warn,
            )

        try:
            lines = resolved_path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        except Exception as e:
            warn = RetrievalWarning(
                category="file_read_error",
                message=f"Failed to read file '{resolved_path}': {e}",
                node_id=node.id,
                file_path=node.source_file,
            )
            return (
                SourceSnippet(
                    file_path=node.source_file,
                    start_line=None,
                    end_line=None,
                    content=None,
                    status="read_error",
                    node_id=node.id,
                ),
                warn,
            )

        total_lines = len(lines)
        if total_lines == 0:
            return (
                SourceSnippet(
                    file_path=node.source_file,
                    start_line=1,
                    end_line=1,
                    content="",
                    status="empty_file",
                    node_id=node.id,
                ),
                None,
            )

        start_line, end_line = parse_source_location(node.source_location)

        if start_line is None:
            # If no location is specified on a file node, return initial window
            if node.type == "file":
                end_line = min(total_lines, self.max_snippet_lines)
                content = "".join(lines[0:end_line])
                return (
                    SourceSnippet(
                        file_path=node.source_file,
                        start_line=1,
                        end_line=end_line,
                        content=content,
                        status="resolved_default_file_head",
                        node_id=node.id,
                    ),
                    None,
                )
            warn = RetrievalWarning(
                category="unresolved_location",
                message=f"Node '{node.id}' has no valid source_location: {node.source_location!r}",
                node_id=node.id,
                file_path=node.source_file,
            )
            return (
                SourceSnippet(
                    file_path=node.source_file,
                    start_line=None,
                    end_line=None,
                    content=None,
                    status="unresolved_location",
                    node_id=node.id,
                ),
                warn,
            )

        # Context window expansion if single line is specified
        if start_line == end_line:
            computed_start = max(1, start_line - self.context_lines)
            computed_end = min(total_lines, end_line + self.context_lines)
        else:
            computed_start = max(1, start_line)
            computed_end = min(total_lines, end_line)

        # Out of bounds check
        if computed_start > total_lines:
            warn = RetrievalWarning(
                category="out_of_range",
                message=f"Start line {start_line} exceeds file length ({total_lines} lines)",
                node_id=node.id,
                file_path=node.source_file,
            )
            return (
                SourceSnippet(
                    file_path=node.source_file,
                    start_line=start_line,
                    end_line=end_line,
                    content=None,
                    status="out_of_range",
                    node_id=node.id,
                ),
                warn,
            )

        # Clamp max lines
        if (computed_end - computed_start + 1) > self.max_snippet_lines:
            computed_end = computed_start + self.max_snippet_lines - 1

        content = "".join(lines[computed_start - 1 : computed_end])

        return (
            SourceSnippet(
                file_path=node.source_file,
                start_line=computed_start,
                end_line=computed_end,
                content=content,
                status="resolved",
                node_id=node.id,
            ),
            None,
        )

    def retrieve_snippets_for_nodes(
        self,
        retrieved_nodes: Sequence[RetrievedNode],
        max_snippets: int = 10,
    ) -> tuple[list[SourceSnippet], list[RetrievalWarning]]:
        """Retrieve snippets for multiple retrieved nodes, deduplicating overlapping regions.

        Args:
            retrieved_nodes: Ranked list of RetrievedNode instances.
            max_snippets: Maximum number of snippets to extract.

        Returns:
            Tuple of (snippets, warnings).
        """
        snippets: list[SourceSnippet] = []
        warnings: list[RetrievalWarning] = []
        seen_ranges: set[tuple[str, Optional[int], Optional[int]]] = set()

        for r_node in retrieved_nodes:
            if len(snippets) >= max_snippets:
                break

            # Skip folders and repositories from snippet extraction
            if r_node.node.type in {"folder", "repository"}:
                continue

            snippet, warn = self.retrieve_node_snippet(r_node.node)
            if warn:
                warnings.append(warn)
            if snippet:
                range_key = (snippet.file_path, snippet.start_line, snippet.end_line)
                if range_key not in seen_ranges:
                    seen_ranges.add(range_key)
                    snippets.append(snippet)

        return snippets, warnings
