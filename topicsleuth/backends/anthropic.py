"""Claude as the judge. ``pip install "topicsleuth[anthropic]"``."""
from __future__ import annotations

from topicsleuth.interfaces import Completion


class AnthropicJudge:
    """``effort`` is the model's reasoning effort (``low``, ``medium``,
    ``high``, ``xhigh``, ``max``), sent as ``output_config.effort`` when
    set and left to the model's default when None. On a model whose
    thinking is always on (Opus 5.5, Sonnet 5.5) it is the only control
    over how much the judge thinks, and thinking bills as output tokens,
    so set it explicitly and replay before keeping a level.

    The system prompt is the rubric: several thousand tokens, identical
    from one call to the next within a run, so it is marked for prompt
    caching and the rest of the prompt (the item and its candidates) is
    not. Reads of the cached prefix bill at a fraction of input price;
    ``Completion`` reports them apart from the uncached tokens. Pass
    ``cache_system=False`` to send the prompt plain."""

    def __init__(self, model: str = "claude-opus-4-8", client=None, api_key: str | None = None,
                 effort: str | None = None, cache_system: bool = True):
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover
            raise ImportError('AnthropicJudge needs the anthropic package: pip install "topicsleuth[anthropic]"') from e
        self.model = model
        self.effort = effort
        self.cache_system = cache_system
        self._client = client or anthropic.Anthropic(api_key=api_key)

    def complete(self, system: str, user: str, max_tokens: int = 2048) -> Completion:
        request: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": ([{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
                       if self.cache_system else system),
            "messages": [{"role": "user", "content": user}],
        }
        if self.effort:
            request["output_config"] = {"effort": self.effort}
        response = self._client.messages.create(**request)
        usage = getattr(response, "usage", None)
        stop_reason = getattr(response, "stop_reason", None)
        # A refusal (stop_reason "refusal") comes back with no content
        # blocks at all, and a reply may carry non-text blocks ahead of its
        # text, so the text is gathered rather than read from a fixed slot.
        text = "".join(
            block.text for block in (response.content or []) if getattr(block, "type", "text") == "text"
        )
        return Completion(
            text=text,
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            truncated=None if stop_reason is None else stop_reason == "max_tokens",
            refused=None if stop_reason is None else stop_reason == "refusal",
            cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", None),
            cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", None),
        )
