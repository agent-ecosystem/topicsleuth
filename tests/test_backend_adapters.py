"""The thin adapters over each SDK: request shape in, protocol shape out,
with the SDK clients stubbed."""
from __future__ import annotations

import io
import json
import urllib.error

import pytest

from topicsleuth.interfaces import Completion, DecisionModel, Embedder, Judge, PairScore
from topicsleuth.rubric import RELATIONSHIP_LABELS


# ── Voyage ──

class _VoyageResponse:
    def __init__(self, embeddings):
        self.embeddings = embeddings


class _VoyageClient:
    def __init__(self, width=4):
        self.calls = []
        self.width = width

    def embed(self, texts, model, input_type, output_dimension):
        self.calls.append((list(texts), model, input_type, output_dimension))
        return _VoyageResponse([[float(len(t))] * self.width for t in texts])


def test_voyage_embedder_maps_documents_and_queries():
    pytest.importorskip("voyageai")
    from topicsleuth.backends.voyage import VoyageEmbedder

    client = _VoyageClient()
    e = VoyageEmbedder(model="voyage-x", dimensions=4, client=client)
    assert isinstance(e, Embedder) and e.dimensions == 4
    assert e.embed_documents(["ab", "c"]) == [[2.0] * 4, [1.0] * 4]
    assert e.embed_queries(["ab"]) == [[2.0] * 4]
    assert e.embed_documents([]) == [] and len(client.calls) == 2
    assert client.calls[0] == (["ab", "c"], "voyage-x", "document", 4), "the dimensionality is requested, not assumed"
    assert client.calls[1] == (["ab"], "voyage-x", "query", 4)


def test_voyage_embedder_rejects_vectors_of_the_wrong_shape():
    pytest.importorskip("voyageai")
    from topicsleuth.backends.voyage import VoyageEmbedder
    wrong = VoyageEmbedder(model="voyage-x", dimensions=256, client=_VoyageClient(width=1024))
    with pytest.raises(ValueError, match="1024-dimensional vector; this embedder is 256"):
        wrong.embed_documents(["ab"])

    class _Short(_VoyageClient):
        def embed(self, texts, model, input_type, output_dimension):
            return _VoyageResponse([[1.0] * 4])
    with pytest.raises(ValueError, match="returned 1 embeddings for 2 texts"):
        VoyageEmbedder(dimensions=4, client=_Short()).embed_documents(["a", "b"])

    class _Nan(_VoyageClient):
        def embed(self, texts, model, input_type, output_dimension):
            return _VoyageResponse([[float("nan")] * 4])
    with pytest.raises(ValueError, match="non-finite"):
        VoyageEmbedder(dimensions=4, client=_Nan()).embed_documents(["a"])


# ── Anthropic ──

class _Block:
    def __init__(self, text):
        self.text = text


class _Usage:
    input_tokens = 123
    output_tokens = 45


class _Message:
    def __init__(self, text, stop_reason="end_turn"):
        self.content = [_Block(text)]
        self.usage = _Usage()
        self.stop_reason = stop_reason


class _Messages:
    def __init__(self, outer):
        self.outer = outer

    def create(self, **kwargs):
        self.outer.calls.append(kwargs)
        return _Message("{\"decision\": \"skip\"}")


class _AnthropicClient:
    def __init__(self):
        self.calls = []
        self.messages = _Messages(self)


def test_anthropic_judge_maps_request_and_completion():
    pytest.importorskip("anthropic")
    from topicsleuth.backends.anthropic import AnthropicJudge

    client = _AnthropicClient()
    j = AnthropicJudge(model="claude-test", client=client)
    assert isinstance(j, Judge)
    out = j.complete("SYSTEM", "USER", max_tokens=77)
    assert out == Completion(text="{\"decision\": \"skip\"}", input_tokens=123, output_tokens=45, stop_reason="end_turn")
    assert not out.truncated
    call = client.calls[0]
    assert call["model"] == "claude-test" and call["max_tokens"] == 77 and call["system"] == "SYSTEM"
    assert call["messages"] == [{"role": "user", "content": "USER"}]


def test_anthropic_judge_reports_a_reply_cut_off_at_max_tokens():
    pytest.importorskip("anthropic")
    from topicsleuth.backends.anthropic import AnthropicJudge
    from topicsleuth.interfaces import JudgeTruncated

    class _CutMessages(_Messages):
        def create(self, **kwargs):
            return _Message('{"candidate_classifications": [{"topic_id": "a", "candidate_que', stop_reason="max_tokens")

    client = _AnthropicClient()
    client.messages = _CutMessages(client)
    out = AnthropicJudge(model="claude-test", client=client).complete("SYSTEM", "USER", max_tokens=2048)
    assert out.stop_reason == "max_tokens" and out.truncated
    with pytest.raises(JudgeTruncated, match="first-pass decision was truncated at 45 output tokens"):
        out.require_complete("first-pass decision")
    assert Completion(text="{}").require_complete("anything") is not None, "no stop reason means no claim either way"


# ── Jev ──

class _HTTPResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _urlopen_factory(responses, seen):
    def urlopen(req, timeout):
        seen.append((req.full_url, req.get_method(), dict(req.header_items()), json.loads(req.data)))
        r = responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return _HTTPResponse(json.dumps(r).encode())
    return urlopen


def test_jev_decision_model_posts_state_and_questions(monkeypatch):
    from topicsleuth.backends import jev

    seen = []
    monkeypatch.setattr(jev.urllib.request, "urlopen", _urlopen_factory(
        [{"model": "jev-9", "answers": {"relationship": {"choice": "same_question", "probabilities": {"same_question": 0.8}, "confidence": 0.9},
                                        "link": {"noul": 0.85}}, "usage": {"input_tokens": 50}}], seen))
    monkeypatch.setattr(jev.time, "sleep", lambda s: None)
    dm = jev.JevDecisionModel("secret", model="jev-test")
    assert isinstance(dm, DecisionModel)
    out = dm.score_pair({"title": "An item", "summary": "s"}, {"title": "A topic", "summary": "t", "evidence_count": 3}, ["rep one"])
    assert out == PairScore(label="same_question", link_probability=0.85, confidence=0.9,
                            probabilities={"same_question": 0.8}, input_tokens=50, model="jev-9")
    url, method, headers, body = seen[0]
    assert url == jev.API_URL and method == "POST"
    assert headers["Authorization"] == "Bearer secret"
    assert body["model"] == "jev-test" and set(body["questions"]) == {"relationship", "link"}
    assert set(body["questions"]["relationship"]["criteria"]) == set(RELATIONSHIP_LABELS)
    assert body["state"]["candidate_topic"]["representative_observations"] == ["rep one"]
    assert body["state"]["item"]["title"] == "An item"


def test_jev_retries_transient_errors_then_raises(monkeypatch):
    from topicsleuth.backends import jev

    def http_error(code):
        return urllib.error.HTTPError("u", code, "err", {}, io.BytesIO(b"detail"))

    seen = []
    monkeypatch.setattr(jev.time, "sleep", lambda s: None)
    monkeypatch.setattr(jev.urllib.request, "urlopen", _urlopen_factory([http_error(503), {"answers": {"relationship": {"choice": "x"}}}], seen))
    assert jev.JevDecisionModel("k", retries=2).decide({}, {})["answers"]["relationship"]["choice"] == "x"
    assert len(seen) == 2

    monkeypatch.setattr(jev.urllib.request, "urlopen", _urlopen_factory([http_error(401)], seen))
    with pytest.raises(jev.JevError, match="HTTP 401"):
        jev.JevDecisionModel("k").decide({}, {})

    with pytest.raises(ValueError):
        jev.JevDecisionModel("")
