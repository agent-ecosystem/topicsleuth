"""Every CLI command, in process, against the shared fakes selected by
environment: a memory store seeded by the test, a scripted judge, a hashing
embedder. Output goes through capsys; files land in a temp working dir."""
from __future__ import annotations

import json
import sys

import pytest

from topicsleuth import cli
from topicsleuth.backends import fake

# The CLI runs on the packaged settings, whose index dimensionality is 1024.
DIM = 1024
E_A = [1.0] + [0.0] * (DIM - 1)
E_B = [0.0, 1.0] + [0.0] * (DIM - 2)


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TOPICSLEUTH_STORE", "memory")
    monkeypatch.setenv("TOPICSLEUTH_JUDGE", "scripted")
    monkeypatch.setenv("TOPICSLEUTH_EMBEDDER", "hash")
    monkeypatch.setenv("TOPICSLEUTH_DECISION", "scripted")
    monkeypatch.delenv("MONGODB_URI", raising=False)
    fake.reset_shared()
    yield tmp_path
    fake.reset_shared()


def run(capsys, *argv):
    sys.argv = ["topicsleuth", *argv]
    try:
        cli.main()
        code = 0
    except SystemExit as e:
        code = e.code
    out = capsys.readouterr()
    return code, out.out + out.err


def seeded():
    st = fake.shared("store")
    st.create_topic({"topic_id": "a", "title": "Topic A", "summary": "Tracks A.", "embedding": E_A, "tags": ["alpha"]})
    st.create_topic({"topic_id": "b", "title": "Topic B", "summary": "Tracks B.", "embedding": E_B, "tags": ["beta"]})
    for i in range(3):
        st.create_observation({"topic_id": "a", "item_url": f"a{i}", "item_title": f"A {i}", "strength": 0.9})
    st.create_observation({"topic_id": "b", "item_url": "b0", "item_title": "B 0", "strength": 0.7})
    return st


def test_help_lists_commands(capsys, env):
    code, out = run(capsys, "--help")
    assert code == 0
    for cmd in ("seed", "clean", "split-topic", "merge-topics", "suggest-merges", "reprocess-tail",
                "add-topic", "move-observation", "audit", "backfill-embeddings", "tune", "show-prompts"):
        assert cmd in out


def test_settings_file_from_the_environment_reaches_the_prompts(capsys, env, monkeypatch):
    (env / "settings.yaml").write_text(
        "domain: municipal housing policy\n"
        "rubric:\n"
        "  source_kinds: council minutes, local news, planning filings\n"
        "  question_examples:\n"
        "    - question: Are cities loosening parking minimums near transit?\n"
        "      subject: parking\n"
    )
    monkeypatch.setenv("TOPICSLEUTH_SETTINGS", "settings.yaml")
    code, out = run(capsys, "show-prompts", "confirm")
    assert code == 0
    assert "municipal housing policy" in out and "council minutes, local news, planning filings" in out
    assert '"Are cities loosening parking minimums near transit?" is a question a topic can track. "parking" is a subject.' in out
    assert "harness vendors" not in out
    code, out = run(capsys, "show-prompts")
    assert code == 0 and out.count("=" * 70) == 12 and "split_reassign" in out


def test_missing_store_configuration_is_a_clean_error(capsys, env, monkeypatch):
    monkeypatch.delenv("TOPICSLEUTH_STORE")
    code, out = run(capsys, "audit", "--no-save")
    assert "MONGODB_URI is not set" in str(code)


def test_seed_and_audit(capsys, env):
    (env / "topics.yaml").write_text("topics:\n  - title: Topic A\n    summary: Tracks A.\n    tags: [alpha]\n")
    code, out = run(capsys, "seed", "topics.yaml", "--dry-run")
    assert code == 0 and "[DRY RUN] would embed and write 1 topics" in out and "topic-a" in out
    assert fake.shared("store").list_topics() == []
    code, out = run(capsys, "seed", "topics.yaml")
    assert code == 0 and "Seeded 1 topics" in out
    st = fake.shared("store")
    assert len(st.get_topic("topic-a")["embedding"]) == DIM
    code, out = run(capsys, "seed", "topics.yaml")
    assert "already has data" in out
    code, out = run(capsys, "seed", "topics.yaml", "--dry-run")
    assert "already holds 1 topics" in out
    code, out = run(capsys, "audit", "--label", "x")
    assert code == 0 and "Total: 1" in out and (env / "tuning_runs" / "audit-x.md").exists()
    (env / "empty.yaml").write_text("topics: []\n")
    assert "No seed topics" in run(capsys, "seed", "empty.yaml")[1]


def test_clean_removes_recent_store_output(capsys, env):
    seeded()
    code, out = run(capsys, "clean", "--hours", "1", "--dry-run")
    assert code == 0 and "[DRY RUN]" in out and "topics_deleted           2" in out and "observations_deleted     4" in out
    assert len(fake.shared("store").list_topics()) == 2
    code, out = run(capsys, "clean", "--hours", "1")
    assert code == 0 and "topics_deleted" in out and fake.shared("store").list_topics() == []


def test_split_topic_dry_run_and_real(capsys, env):
    seeded()
    subs = {"sub_topics": [{"topic_id": "a-one", "title": "A one", "summary": "s1", "tags": [], "suggested_directions": []},
                           {"topic_id": "a-two", "title": "A two", "summary": "s2", "tags": [], "suggested_directions": []}]}
    reassign = [{"item_url": f"a{i}", "new_topic_id": "a-one" if i < 2 else "a-two", "new_strength": 0.8, "new_connection_type": "supports", "reasoning": "r"} for i in range(3)]
    fake.shared("judge").queue(subs, reassign, subs, reassign)
    code, out = run(capsys, "split-topic", "a", "--dry-run")
    assert code == 0 and "[DRY RUN]" in out and "a-one: 2 observations" in out and fake.shared("store").get_topic("a")["status"] == "emerging"
    code, out = run(capsys, "split-topic", "a")
    assert code == 0 and "Split complete" in out and fake.shared("store").get_topic("a")["status"] == "retired"


def test_merge_topics_and_suggest(capsys, env):
    seeded()
    merged = {"title": "AB", "summary": "Tracks AB.", "tags": ["alpha", "beta"], "suggested_directions": [], "rationale": "same"}
    fake.shared("judge").queue(merged, merged)
    code, out = run(capsys, "merge-topics", "b", "a", "--dry-run")
    assert code == 0 and "[DRY RUN] Proposed merge" in out and "merged title:   AB" in out
    code, out = run(capsys, "merge-topics", "b", "a")
    assert code == 0 and "Merge complete" in out and fake.shared("store").get_topic("b")["merged_into"] == "a"
    code, out = run(capsys, "suggest-merges", "--threshold", "0.5")
    assert code == 0 and "No pairs above" in out
    st = fake.shared("store")
    st.create_topic({"topic_id": "a2", "title": "A again", "embedding": st.get_topic("a")["embedding"], "tags": ["alpha"]})
    code, out = run(capsys, "suggest-merges", "--threshold", "0.9")
    assert "topicsleuth merge-topics a2 a --dry-run" in out


def test_reprocess_tail_writes_urls_for_the_feeder(capsys, env):
    seeded()
    code, out = run(capsys, "reprocess-tail", "--max-obs", "1", "--dry-run")
    assert code == 0 and "[DRY RUN]" in out and "- b" in out and fake.shared("store").get_topic("b")
    code, out = run(capsys, "reprocess-tail", "--max-obs", "1", "--urls-out", "requeue.urls")
    assert code == 0 and "Items to re-queue:    1" in out
    assert (env / "requeue.urls").read_text() == "b0\n" and fake.shared("store").get_topic("b") is None
    assert "Nothing to do" in run(capsys, "reprocess-tail", "--max-obs", "0")[1]


def test_add_topic_and_move_observation(capsys, env):
    seeded()
    (env / "d.yaml").write_text("title: Topic D\nsummary: Tracks D.\n")
    code, out = run(capsys, "add-topic", "d.yaml", "--link", "a0", "--link", "nope", "--dry-run")
    assert code == 0 and "[DRY RUN]" in out and "no observation for nope" in out and fake.shared("store").get_topic("topic-d") is None
    code, out = run(capsys, "add-topic", "d.yaml", "--link", "a0")
    assert code == 0 and "moved 1 observations" in out and fake.shared("store").count_observations_for_topic("topic-d") == 1
    code, out = run(capsys, "move-observation", "b0", "topic-d", "--dry-run")
    assert code == 0 and "left empty, will be retired" in out
    code, out = run(capsys, "move-observation", "b0", "topic-d")
    assert code == 0 and "Retired: b" in out and fake.shared("store").get_topic("b")["status"] == "retired"


def test_backfill_embeddings(capsys, env):
    st = seeded()
    st.update_topic("b", {"embedding": [1.0, 0.0]})       # wrong dimensions: needs a seed
    for i in range(5):
        st.create_observation({"topic_id": "a", "item_url": f"x{i}", "strength": 0.9})
        st.save_item_embedding(f"x{i}", E_A)
    code, out = run(capsys, "backfill-embeddings", "--dry-run")
    assert code == 0 and "would seed: b" in out and "would refresh: a" in out
    code, out = run(capsys, "backfill-embeddings")
    assert code == 0 and "seeded=1/1" in out and "refreshed=1/1" in out
    assert len(st.get_topic("b")["embedding"]) == DIM


def test_tune_replays_a_prior_report(capsys, env):
    st = seeded()
    st.items["a0"] = {"url": "a0", "title": "A 0", "summary": "s", "embedding": E_A}
    (env / "prior.json").write_text(json.dumps({"decisions": [{"item_url": "a0"}]}))
    fake.shared("judge").queue({"candidate_classifications": [{"topic_id": "a", "label": "same_question"}], "decision": "link_existing",
                                "links": [{"topic_id": "a", "connection_type": "supports",
                                           "fit_questions": {"q1_scope_expansion": True, "q2_representative_citation": True, "q3_distinguishing_evidence": True}}],
                                "reasoning_trace": "r"})
    code, out = run(capsys, "tune", "synthesize", "--replay", "prior.json", "--label", "t", "--limit", "3", "--model", "m2")
    assert code == 0 and "link_existing     1" in out and "report:" in out
    report = json.loads((env / "tuning_runs" / "synthesize-t-replay.json").read_text())
    assert report["config"]["model"] == "m2" and report["config"]["candidates"]["vector_limit"] == 3 and "leave_one_out" not in report
    code, out = run(capsys, "tune", "synthesize")
    assert code != 0 and "needs --replay" in str(code)
