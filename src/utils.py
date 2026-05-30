"""
MedQueryAI - Logging Utility
Windows-safe print wrapper that handles Unicode/emoji output.
"""

import sys
import io


def setup_console():
    """Fix Windows console encoding to support Unicode/emoji output."""
    if sys.platform == "win32":
        try:
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer, encoding="utf-8", errors="replace"
            )
            sys.stderr = io.TextIOWrapper(
                sys.stderr.buffer, encoding="utf-8", errors="replace"
            )
        except Exception:
            pass


# Auto-run on import
setup_console()
