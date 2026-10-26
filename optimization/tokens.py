"""Token estimation interfaces and approximate code-token estimators.

Provides a decoupled interface for token counting so that exact tokenizers
(e.g., tiktoken, tokenizers) can be plugged in later without altering the
context selection logic.
"""

from __future__ import annotations

import re
from typing import Any, Optional, Protocol, runtime_checkable

from optigraph.retrieval.models import RetrievedNode, RetrievedRelationship, SourceSnippet


@runtime_checkable
class TokenEstimator(Protocol):
    """Protocol for estimating or calculating token counts for context elements."""

    name: str
    is_exact: bool
    accuracy_designation: str

    def count_tokens(self, text: Optional[str]) -> int:
        """Estimate or compute token count for a text string."""
        ...

    def count_snippet_tokens(self, snippet: SourceSnippet) -> int:
        """Estimate token cost for a source snippet including formatting overhead."""
        ...

    def count_node_tokens(self, node: RetrievedNode) -> int:
        """Estimate token cost for a node's structural representation."""
        ...

    def count_relationship_tokens(self, rel: RetrievedRelationship) -> int:
        """Estimate token cost for a graph relationship representation."""
        ...


class ApproximateTokenEstimator:
    """Deterministic, lightweight approximate token estimator for code and text.

    Methodology:
        Modern Byte-Pair Encoding (BPE) tokenizers (such as cl100k_base or Llama)
        split code into:
        - CamelCase, snake_case, and dotted identifier sub-words
        - Individual punctuation and operator symbols
        - Number sequences
        - Whitespace runs and newlines

        This estimator analyzes code and text using sub-token regular expressions
        calibrated against typical BPE code tokenization (~3.5-4.0 characters per token).

    Limitations:
        - This is an APPROXIMATE estimator, not an exact model-specific tokenizer.
        - Vocabulary-specific merging rules (e.g., frequent multi-character keywords)
          may differ slightly from specific LLM tokenizers.
        - Never presents approximate counts as exact counts.
    """

    name: str = "approximate_code_heuristic"
    is_exact: bool = False
    accuracy_designation: str = "approximate"

    # Regex capturing code tokens: sub-words, symbols, numbers, and whitespace runs
    _TOKEN_PATTERN: re.Pattern = re.compile(
        r"[A-Z]{2,}(?![a-z])|[A-Z][a-z]+|[a-z]+|[0-9]+|[^\s\w]|\n|\s+",
        re.UNICODE,
    )

    def __init__(self, overhead_per_item: int = 4) -> None:
        """Initialize the approximate token estimator.

        Args:
            overhead_per_item: Structural markdown/citation overhead (fence marks,
                file path headers) added per context block.
        """
        self.overhead_per_item = max(0, overhead_per_item)

    def count_tokens(self, text: Optional[str]) -> int:
        """Estimate token count for a raw text string.

        Args:
            text: Arbitrary string or None.

        Returns:
            Estimated token count (integer >= 0).
        """
        if not text:
            return 0

        # Pattern match tokens
        tokens = self._TOKEN_PATTERN.findall(text)
        count = len(tokens)

        # Sanity lower and upper bounds based on character length
        char_len = len(text)
        min_tokens = max(1, char_len // 6)
        max_tokens = max(1, int(char_len // 2.5) + 1)

        return min(max(count, min_tokens), max_tokens)

    def count_snippet_tokens(self, snippet: SourceSnippet) -> int:
        """Estimate token cost for an extracted source code snippet.

        Includes header citation tokens (`file_path:start-end`) plus code content.
        """
        if not snippet.content:
            # Missing or unresolved snippet placeholder
            header = f"# {snippet.file_path} (status: {snippet.status})\n"
            return self.count_tokens(header) + self.overhead_per_item

        start = snippet.start_line or 1
        end = snippet.end_line or start
        header = f"### File: {snippet.file_path} (lines {start}-{end})\n```\n"
        footer = "\n```\n"

        content_tokens = self.count_tokens(snippet.content)
        structure_tokens = self.count_tokens(header + footer)

        return content_tokens + structure_tokens + self.overhead_per_item

    def count_node_tokens(self, node: RetrievedNode) -> int:
        """Estimate token cost for a node's structural description."""
        repr_text = (
            f"Node: {node.node.name} (type={node.node.type}, id={node.node.id}, "
            f"file={node.node.source_file}, loc={node.node.source_location})"
        )
        return self.count_tokens(repr_text) + 2

    def count_relationship_tokens(self, rel: RetrievedRelationship) -> int:
        """Estimate token cost for a relationship representation."""
        repr_text = f"Edge: {rel.edge.source} -[{rel.edge.type}]-> {rel.edge.target}"
        return self.count_tokens(repr_text) + 2

    def to_dict(self) -> dict[str, Any]:
        """Serialize estimator metadata."""
        return {
            "name": self.name,
            "is_exact": self.is_exact,
            "accuracy_designation": self.accuracy_designation,
            "overhead_per_item": self.overhead_per_item,
        }
