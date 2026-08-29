"""ROUGE-L validator: semantic integrity check after context compression.

Core principle: better to spend extra tokens than let the agent lose
critical information to an over-aggressive compression.
"""

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from loguru import logger
from rouge_score import rouge_scorer

from tether.context.budget import estimate_tokens


@dataclass
class ValidationResult:
    """Outcome of a single validation run."""

    passed: bool
    rouge_l_score: float
    threshold: float
    original_token_count: int
    compressed_token_count: int
    was_rolled_back: bool
    reason: Optional[str] = None


class RougeValidator:
    """Validates that compressed context preserves original semantics.

    Uses ROUGE-L F1 plus an optional required-keyword check. Scores are
    cached by MD5 hash of the text pair to avoid recomputation.
    """

    def __init__(self, threshold: float = 0.7, cache_size: int = 128) -> None:
        """Store threshold; lower ROUGE-L F1 than this triggers rollback."""
        self.threshold = threshold
        self._scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        self._cache_size = cache_size
        self._cache: Dict[Tuple[str, str], float] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    @staticmethod
    def _text_hash(text: str) -> str:
        """MD5 hex digest of a text, used as cache key."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _cached_rouge_l(self, original_context: str, compressed_context: str) -> float:
        """Compute (or fetch cached) ROUGE-L F1 for a text pair."""
        key = (
            self._text_hash(original_context),
            self._text_hash(compressed_context),
        )
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        self.cache_misses += 1

        if not original_context.strip() or not compressed_context.strip():
            score = 0.0
        else:
            score = self._scorer.score(original_context, compressed_context)["rougeL"].fmeasure

        # Simple size-capped cache: drop everything when full.
        if len(self._cache) >= self._cache_size:
            self._cache.clear()
        self._cache[key] = score
        return score

    def validate(
        self,
        original_context: str,
        compressed_context: str,
        required_keywords: Optional[List[str]] = None,
    ) -> ValidationResult:
        """Check semantic integrity of ``compressed_context`` vs original.

        Passes only if ROUGE-L F1 >= threshold AND every required keyword
        survives in the compressed text; otherwise the result is marked
        rolled back with a human-readable reason.
        """
        rouge_l = self._cached_rouge_l(original_context, compressed_context)
        keywords = required_keywords or []
        missing = [kw for kw in keywords if kw not in compressed_context]

        original_tokens = estimate_tokens(original_context)
        compressed_tokens = estimate_tokens(compressed_context)

        if rouge_l >= self.threshold and not missing:
            logger.debug(
                "ROUGE-L validation passed: {:.3f} (>= {}) tokens {} -> {}",
                rouge_l, self.threshold, original_tokens, compressed_tokens,
            )
            return ValidationResult(
                passed=True,
                rouge_l_score=rouge_l,
                threshold=self.threshold,
                original_token_count=original_tokens,
                compressed_token_count=compressed_tokens,
                was_rolled_back=False,
            )

        reasons = []
        if rouge_l < self.threshold:
            reasons.append(f"ROUGE-L {rouge_l:.3f} < {self.threshold}")
        if missing:
            reasons.append(f"missing keywords: {missing}")
        reason = "; ".join(reasons)
        logger.warning(
            "ROUGE-L validation failed: {}, rolling back to original "
            "(tokens {} -> {})",
            reason, original_tokens, compressed_tokens,
        )
        return ValidationResult(
            passed=False,
            rouge_l_score=rouge_l,
            threshold=self.threshold,
            original_token_count=original_tokens,
            compressed_token_count=compressed_tokens,
            was_rolled_back=True,
            reason=reason,
        )
