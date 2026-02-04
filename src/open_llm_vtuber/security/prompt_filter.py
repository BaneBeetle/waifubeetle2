"""
Prompt Injection Detection and Filtering
Provides pattern-based detection for common LLM jailbreak attempts.
"""

import re
from typing import Tuple
from loguru import logger


# Common prompt injection patterns (case-insensitive)
INJECTION_PATTERNS = [
    # Direct instruction override attempts
    r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?)",
    r"forget\s+(all\s+)?(your\s+)?(instructions?|rules?|training|guidelines?)",
    r"disregard\s+(all\s+)?(previous|prior|your)\s+(instructions?|prompts?)",
    # Role manipulation
    r"you\s+are\s+now\s+(?!going|about)",  # "you are now X" but not "you are now going to"
    r"pretend\s+(you\s+are|to\s+be|you're)",
    r"act\s+as\s+(if\s+you\s+are|a|an)",
    r"roleplay\s+as",
    r"simulate\s+(being|a)",
    # Known jailbreak terms
    r"\bDAN\b",  # Do Anything Now
    r"\bjailbreak\b",
    r"\bdevmode\b",
    r"developer\s+mode",
    # Prompt leaking attempts
    r"(show|reveal|display|print|output)\s+(me\s+)?(your|the)\s+(system\s+)?(prompt|instructions?)",
    r"what\s+(are|is)\s+your\s+(system\s+)?(prompt|instructions?)",
    # Meta instructions
    r"from\s+now\s+on",
    r"new\s+rules?:",
    r"override\s+(the\s+)?(safety|content)\s+filter",
]

# Compile patterns for performance
_compiled_patterns = [
    re.compile(pattern, re.IGNORECASE) for pattern in INJECTION_PATTERNS
]


def detect_prompt_injection(text: str) -> Tuple[bool, str | None]:
    """
    Detect potential prompt injection attempts in user input.

    Args:
        text: User input text to analyze

    Returns:
        Tuple of (is_suspicious, matched_pattern)
        - is_suspicious: True if potential injection detected
        - matched_pattern: The pattern that matched, or None
    """
    if not text:
        return False, None

    for pattern in _compiled_patterns:
        match = pattern.search(text)
        if match:
            logger.warning(
                f"[SECURITY] Potential prompt injection detected: '{match.group()}'"
            )
            return True, match.group()

    return False, None


def sanitize_for_llm(text: str, block_injections: bool = False) -> str:
    """
    Sanitize text before sending to LLM.

    Args:
        text: User input text
        block_injections: If True, raises ValueError on injection detection

    Returns:
        Sanitized text (currently returns as-is, logs warnings)

    Raises:
        ValueError: If block_injections=True and injection detected
    """
    is_suspicious, matched = detect_prompt_injection(text)

    if is_suspicious:
        if block_injections:
            raise ValueError(
                "Input contains potentially harmful content and was blocked"
            )
        # Log but allow - the LLM should handle this with good system prompts
        logger.warning(
            f"[SECURITY] Allowing suspicious input through (blocking disabled)"
        )

    return text
