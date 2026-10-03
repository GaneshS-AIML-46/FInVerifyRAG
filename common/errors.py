"""Failures that paper runs must not swallow."""
from __future__ import annotations


class StrictFailure(RuntimeError):
    """Raised when a strict paper run would otherwise invent a number."""


def print_not_paper_valid_banner(reason: str = "") -> None:
    """Print prominent NOT PAPER-VALID banner when operating outside strict mode."""
    sep = "=" * 72
    lines = [
        "",
        sep,
        "                          NOT PAPER-VALID",
    ]
    if reason:
        lines.append(f"  Reason: {reason}")
    lines.append(sep)
    lines.append("")
    print("\n".join(lines), flush=True)
