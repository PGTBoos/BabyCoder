"""
agent_toolkit.py

Kept so existing code and tests that do `import agent_toolkit` or
`from agent_toolkit import memory` keep working. The code lives in the
babycoder package now; new code should import from there.

This module replaces itself with the package, so agent_toolkit.X and
babycoder.X are the same object. Settings are changed with
configure_model(...) rather than by assigning agent_toolkit.LM_STUDIO_URL,
because the model code reads them from babycoder.settings.
"""

import sys

import babycoder

sys.modules[__name__] = babycoder
