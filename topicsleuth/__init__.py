"""Topic detection and tracking over a stream of documents: link each
item to a topic or open a new one, keep the evidence, and measure how
topics move.

The pieces: ``synthesize`` decides, ``record`` writes, ``interfaces``
names the seams a feeder fills in (store, embedder, judge, decision
model), ``backends`` holds the default implementations, and ``settings``
and ``rubric`` carry what two feeders writing one store must agree on.
"""
from __future__ import annotations

__version__ = "0.2.0"

__all__ = ["__version__"]
