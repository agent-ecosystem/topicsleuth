"""Claude as the judge. ``pip install "topicsleuth[anthropic]"``."""
from __future__ import annotations

from topicsleuth.interfaces import Completion


class AnthropicJudge:
    def __init__(self, model: str = "claude-opus-4-8", client=None, api_key: str | None = None):
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover
            raise ImportError('AnthropicJudge needs the anthropic package: pip install "topicsleuth[anthropic]"') from e
        self.model = model
        self._client = client or anthropic.Anthropic(api_key=api_key)

    def complete(self, system: str, user: str, max_tokens: int = 2048) -> Completion:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
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
        )
