"""The rubric: everything a model is told, and how its answers become numbers.

This is the one place to read or change what the library asks of a model.
The judge prompts for the decision, the proposal reviewer, merge and split
are assembled here from shared pieces, so a definition appears once and
every prompt that needs it renders the same text. A decision model gets the
same relationship criteria as data. The maps from a model's answers to the
strengths the store records live here too, since they are the other half
of the questions.

Read top to bottom:

  1. what a topic is, and what counts as right-sized
  2. how an item relates to a candidate topic: the relationship labels
  3. the decision rules
  4. the fit questions, and the strength they map to
  5. the connection types
  6. the prompts, assembled from the above

Two kinds of text live here. The structure (the labels, the decision
rules, the fit questions, the connection types, the response formats) is
fixed in code, because the parser, the strength map and the store depend
on it. The prose that describes a particular field (what kinds of source
arrive, how each kind bears on a question, what a right-sized question
looks like there, the counter-examples) is ``RubricConfig``: it ships
with defaults and a store sets its own through ``settings.rubric``. The
prompts take a ``Settings`` and render both.

Change the rubric and replay it against a baseline before trusting the
change (see "Keep the judge honest" in the README): wording moves the
link-or-propose boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from topicsleuth.settings import Settings


# ── 1. What a topic is ───────────────────────────────────────────────

DEFAULT_COUNTER_EXAMPLES = (
    "An above-threshold paper proposing a benchmark for prompt engineering, with "
    "no candidates rated `same_question` or `subset_question`, where the "
    "candidates are all adjacent: the right answer is `skip` if no genuinely new "
    "right-sized topic is in the paper, not `propose_new`.",

    "A paper on agentic post-training, judged against an existing topic "
    "\"Training Paradigms for Agentic Code Models\" rated `subset_question`: the "
    "right answer is `link_existing`. Even if the paper has its own narrative "
    "angle, it is a specific instance of the candidate's question. The strength "
    "questions then determine score; typically Q1=no (summary covers it), Q2=yes "
    "(citable), Q3 depends on whether it adjudicates anything.",

    "A paper proposing a new failure detection method, judged against \"Agent "
    "Failure Mode Detection\" rated `same_question`: link_existing. Q1 likely no, "
    "Q2 yes, Q3 yes if the method is genuinely novel rather than incremental; "
    "strength ends up in the upper band.",

    "A paper on RL training that briefly observes some agent failures, judged "
    "against \"Agent Failure Mode Detection\" rated `surface_overlap` (the "
    "question is RL training, failures are incidental): do not link, even at low "
    "strength. Either link to a training-topic candidate or skip.",

    "A changelog entry from a harness vendor adding sandboxed tool execution, "
    "judged against \"Are harness vendors sandboxing agent tool execution by "
    "default?\" rated `same_question`: link_existing with `supports`, or `shifts` "
    "if it is the first vendor to make it the default. The item being a release "
    "rather than a paper changes nothing about the link.",

    "A product announcement that merely mentions agents, with candidates all "
    "rated `surface_overlap`, and no right-sized question of its own: skip. A "
    "topic per product is a catch-all in waiting.",
)


@dataclass
class RubricConfig:
    """The prose in the prompts that describes the field a store follows.

    Everything here is rendered into the judge prompts next to the fixed
    structure. The defaults describe a feed of papers, release notes and
    posts about a technical field; a store that follows something else
    sets its own under ``rubric:`` in its settings file. Each field:

    source_kinds: the kinds of source that arrive, as a list in prose.
    evidence_kinds: the vocabulary of the ``evidence`` field a feeder may
        put on an item (what the item does: proposes, measures, ships...).
    reading_by_kind: one sentence fragment telling the judge how each kind
        of source bears on a question, so a release and a paper on the same
        question land on the same topic.
    topic_size: how much evidence a right-sized topic could gather, as a
        phrase; sets the judge's sense of how narrow a question should be.
    question_examples: pairs of a right-sized question and the subject it
        would be wrong to track instead, in the store's own field. Two or
        three are enough; the judge generalizes from them.
    subset_example: one worked example of the ``subset_question`` label,
        the label that most often separates a link from a duplicate topic.
    counter_examples: decisions the judge is likely to get wrong in this
        field, each stated as the case and the right answer, using the
        labels and the Q1/Q2/Q3 fit questions by name. Write these from
        the mistakes you see in replays.
    """
    source_kinds: str = "papers, release notes, blog posts, discussion threads, tutorials"
    evidence_kinds: list[str] = field(
        default_factory=lambda: ["proposes", "measures", "argues", "ships", "uses"]
    )
    reading_by_kind: str = (
        "a paper bears on a question by proposing or measuring something, a "
        "release note by showing that something shipped, a tutorial or usage "
        "report by showing something in practice, an opinion piece by arguing a "
        "position"
    )
    topic_size: str = "10-50 observations over a year"
    question_examples: list[dict[str, str]] = field(default_factory=lambda: [
        {"question": "How do agents fail in ways static analysis misses?",
         "subject": "Agent failures"},
        {"question": "Are harness vendors sandboxing agent tool execution by default?",
         "subject": "sandboxing"},
    ])
    subset_example: str = (
        "The candidate asks how agents fail in ways static analysis misses; the "
        "item proposes a detection method for one failure class, or ships a tool "
        "that catches one. The item is a subset, not a different question."
    )
    counter_examples: list[str] = field(default_factory=lambda: list(DEFAULT_COUNTER_EXAMPLES))


def right_sized_topic(rubric: RubricConfig) -> str:
    examples = ""
    if rubric.question_examples:
        parts = []
        for i, ex in enumerate(rubric.question_examples):
            q, s = ex["question"], ex["subject"]
            if i == 0:
                parts.append(f"\"{q}\" is a question a topic can track. \"{s}\" is a subject.")
            else:
                parts.append(f"\"{q}\" is a question; \"{s}\" is a subject.")
        examples = " (" + " ".join(parts) + ")"
    return (
        "A topic tracks one question over time, not a subject area. A right-sized topic:\n"
        "- Names a single specifiable question that distinct items can each bear on "
        f"from different angles.{examples}\n"
        f"- Could plausibly accumulate {rubric.topic_size} without becoming incoherent.\n"
        "- Is narrower than a subfield but broader than a single technique, product, "
        "or release."
    )


def reading_by_kind(rubric: RubricConfig) -> str:
    kinds = ", ".join(f"`{k}`" for k in rubric.evidence_kinds)
    return (
        f"The item carries its source type and may carry an evidence kind ({kinds}) "
        f"and tags from its feeder. Read the item for what it is: {rubric.reading_by_kind}. "
        "The question is the same across kinds. Two items of different kinds on the "
        "same question belong on the same topic; that convergence is the point of "
        "the store. Kind never decides the link: do not propose a separate topic for "
        "the practice side of a question an existing topic already tracks, and do "
        "not fold a different question into a topic because both mention the same "
        "product, organization or name."
    )


SYMMETRIC_FAILURES = (
    "A topic that absorbs every loosely-related item becomes a catch-all. A topic "
    "spawned for every novel angle creates redundant siblings. The two failures are "
    "symmetric; both are wrong."
)


# ── 2. How an item relates to a candidate topic ──────────────────────

RELATIONSHIP_LABELS = (
    "same_question",
    "subset_question",
    "adjacent_question",
    "surface_overlap",
)

RELATIONSHIP_INSTRUCTIONS = (
    "Name the question the item bears on and the question the candidate topic "
    "tracks, then pick the structural relationship between them. The labels are "
    "descriptive, not evaluative: they name a structural relationship, not a "
    "quality judgment. Weigh the topic's representative observations alongside "
    "its one-sentence summary."
)

# One entry per label. ``what`` is the definition; ``tie_break`` and
# ``not_for`` sharpen the boundary with the neighbouring label. The judge
# prompts render these as prose, with the store's own worked example of
# subset_question (RubricConfig.subset_example); a decision model gets the
# dict as it stands.
RELATIONSHIP_CRITERIA = {
    "same_question": {
        "what": (
            "The candidate's question is the question this item bears on. The "
            "item's primary contribution is an answer, a partial answer, or a new "
            "piece of evidence for that question."
        ),
        "not_for": (
            "An item that only touches the candidate's subject, or that asks a "
            "narrower sub-question (that is subset_question)."
        ),
    },
    "subset_question": {
        "what": (
            "The item bears on a specific instance of the candidate's question; the "
            "candidate's question subsumes what the item is about. A different "
            "method, artifact type, or subdomain inside the same question is still "
            "a subset."
        ),
        "not_for": (
            "An item whose question the candidate's summary, read fairly, does not "
            "cover (that is adjacent_question)."
        ),
    },
    "adjacent_question": {
        "what": (
            "The item bears on a different question in the same domain or subfield "
            "as the candidate. The candidate's summary does not actually cover what "
            "the item is about, even though they share keywords."
        ),
        "tie_break": (
            "If broadening the candidate's summary by one clause would cover the "
            "item, prefer subset_question."
        ),
        "not_for": (
            "Pairs that share only tags, model names or subfield with no shared "
            "direction (that is surface_overlap)."
        ),
    },
    "surface_overlap": {
        "what": (
            "The item shares tags, model names, products, or subfield with the "
            "candidate, but its question lies elsewhere. A literature review of the "
            "candidate's topic would not cite this item."
        ),
        "not_for": (
            "Pairs in the same area with a genuinely neighbouring question (that is "
            "adjacent_question)."
        ),
    },
}

LINK_CRITERION = (
    "This item belongs as evidence under this topic: the topic's question is "
    "the same as, or subsumes, the item's question."
)


def relationship_rubric_text(rubric: RubricConfig) -> str:
    """The relationship criteria as the prose the judge prompts carry."""
    examples = {"subset_question": rubric.subset_example}
    blocks = []
    for label in RELATIONSHIP_LABELS:
        c = RELATIONSHIP_CRITERIA[label]
        text = f"- `{label}`: {c['what']}"
        if examples.get(label):
            text += f" (Example: {examples[label]})"
        if "tie_break" in c:
            text += f" {c['tie_break']}"
        text += f" Not for: {c['not_for']}"
        blocks.append(text)
    return "\n\n".join(blocks)


REPRESENTATIVE_OBSERVATIONS_NOTE = (
    "When candidates include `representative observations`, use those to ground "
    "your classification. The candidate's one-sentence summary describes the topic "
    "abstractly; the representative observations show what the topic actually "
    "contains. If the new item resembles those observations in question, not just "
    "subject, that is strong evidence for `same_question` or `subset_question`, "
    "even if the summary is phrased more generally than the item's specific "
    "contribution, and even if the observations and the item are different kinds "
    "of source."
)


def pair_view(item: dict, topic: dict, representative_titles: list[str]) -> dict:
    """What a decision model is shown for one item-topic pair: the item's
    title, summary and an excerpt of its text; the candidate's title,
    summary, evidence count and representative observations. A backend
    renders this into whatever request shape its API takes."""
    return {
        "item": {
            "title": item.get("title", ""),
            "summary": item.get("summary", ""),
            "abstract_excerpt": (item.get("content_raw") or "")[:1800],
        },
        "candidate_topic": {
            "title": topic.get("title", ""),
            "summary": topic.get("summary", ""),
            "observations_so_far": topic.get("evidence_count"),
            "representative_observations": list(representative_titles),
        },
    }


# ── 3. The decision rules ────────────────────────────────────────────

DECISIONS = ("link_existing", "propose_new", "skip")

# The first two rules are also enforced in code after the judge answers
# (synthesize._enforce_link_gate): a model that proposes over a same- or
# subset-question candidate is overridden to link.
DECISION_RULES = (
    "Apply these rules in order. Do not skip ahead.\n\n"
    "1. If any candidate is `same_question`, link_existing to it. If multiple, "
    "pick the one with the most evidence (convergence beats fragmentation).\n"
    "2. Else if any candidate is `subset_question`, link_existing to it. "
    "Subset_question is the right home for items that are specific instances of "
    "an existing direction; this is not a reason to spawn a more specific "
    "sibling.\n"
    "3. Else (all candidates are `adjacent_question` or `surface_overlap`):\n"
    "   - If the item bears on a question that is itself right-sized (per the "
    "rubric above) and is genuinely distinct from every adjacent candidate's "
    "question, propose_new.\n"
    "   - Otherwise skip. An above-threshold item that does not link and does "
    "not warrant a new topic is a valid outcome, not a failure mode."
)

SURVEYS = (
    "A common failure mode is to treat a survey paper, a \"state of X\" post, or a "
    "launch roundup as broader than every candidate and propose a \"comprehensive "
    "X\" meta-topic. Do not do this. A survey or overview is a subset of every "
    "topic it covers, not a superset. The question of a survey is whichever "
    "subset_question candidate the survey most centrally treats. If multiple "
    "candidates are subset matches, link to the one with the most evidence (the "
    "survey is then a strong piece of representative evidence for that topic). "
    "Meta-topics named \"comprehensive X\" or \"everything about Y\" produce "
    "catch-alls and should be avoided."
)


def counter_examples_text(rubric: RubricConfig) -> str:
    return "\n\n".join(f"- {text}" for text in rubric.counter_examples)


# ── 4. The fit questions and the strength they map to ────────────────

# Three yes/no questions about an item-topic link, in the order the model
# answers them. The model never emits a strength; derive_strength maps the
# answers to one.
FIT_QUESTIONS = {
    "q1_scope_expansion": (
        "Would the topic's one-sentence summary need to expand or be rewritten to "
        "incorporate this item's contribution? (yes if the item broadens what the "
        "topic covers; no if the existing summary already describes what this "
        "item investigates.)"
    ),
    "q2_representative_citation": (
        "If someone were writing a one-page summary of this topic and citing 5 "
        "representative items, would this item plausibly be in that list? (yes if "
        "the item is among the kind of work that characterizes the topic; no if "
        "it is one of many similar items that wouldn't make the cut.)"
    ),
    "q3_distinguishing_evidence": (
        "Does the item contribute evidence that distinguishes the topic's claim "
        "from alternative explanations or competing positions? (yes if the item "
        "helps adjudicate a question about the topic; no if it adds another data "
        "point consistent with what's already established.)"
    ),
}

FIT_INSTRUCTIONS = (
    "Do not produce a strength number directly. Answer three independent yes/no "
    "questions about the item-topic relationship. The questions describe "
    "structural properties, not quality."
)

FIT_CLOSING = (
    "The strength score is computed from these answers downstream; you do not "
    "emit it. Answer the three questions honestly; do not work backward from a "
    "target strength."
)

# Q2 (representative citation) is the load-bearing question; the others
# refine within bands.
STRENGTH_BANDS = {
    "all_three_yes": 0.95,          # scope-expanding, citable, distinguishing
    "citable_plus_one": 0.80,       # Q2 yes and one of Q1/Q3 yes
    "citable_only": 0.65,           # Q2 yes, neither expanding nor distinguishing
    "not_citable_some": 0.50,       # Q2 no, but Q1 or Q3 (or both) yes
    "all_three_no": 0.30,           # link forced by the gate; the item is marginal
}


def derive_strength(fit_questions: dict | None) -> float | None:
    """Map the three fit answers to a strength. Returns None if the answers
    are missing or malformed, in which case callers fall back."""
    if not isinstance(fit_questions, dict):
        return None
    try:
        answers = [bool(fit_questions[q]) for q in FIT_QUESTIONS]
    except (KeyError, TypeError):
        return None
    q2 = answers[1]
    yes_count = sum(answers)
    if yes_count == 3:
        return STRENGTH_BANDS["all_three_yes"]
    if q2 and yes_count == 2:
        return STRENGTH_BANDS["citable_plus_one"]
    if q2 and yes_count == 1:
        return STRENGTH_BANDS["citable_only"]
    if not q2 and yes_count >= 1:
        return STRENGTH_BANDS["not_citable_some"]
    return STRENGTH_BANDS["all_three_no"]


# Strength assigned to every link made by the proposal reviewer, by either
# backend. The reviewer is not asked the fit questions: having just argued
# that the item belongs, it answered yes to all three nearly every time
# (0.95) in tuning runs, which is agreement with itself, not calibration.
# What these links are is known by construction: an item the first pass
# judged distinct enough to want its own topic, pulled back as an edge
# extension. That is real scope evidence, so it stays above the 0.7 minor
# floor for mature topics (record.STRENGTH_FLOORS), but it should not
# outrank a topic's core observations (0.80/0.95) when representative
# examples are chosen.
CONFIRM_LINK_STRENGTH = 0.70

# Strength for a link the gate forces over a propose_new answer: the model
# did not answer the fit questions for the candidate it is now linked to.
GATE_OVERRIDE_DEFAULT_STRENGTH = 0.65


def fit_questions_text() -> str:
    return "\n\n".join(f"- `{key}`: {text}" for key, text in FIT_QUESTIONS.items())


# ── 5. The connection types ──────────────────────────────────────────

# How an observation relates to what its topic already holds. The judge
# labels each link; the library keeps the label for readers and does not
# act on it.
CONNECTION_TYPES = {
    "supports": (
        "adds evidence consistent with the topic's claim. This is the default "
        "for most links."
    ),
    "extends": (
        "broadens the topic's scope (use only when q1_scope_expansion is yes)."
    ),
    "contradicts": "provides evidence against the topic's claim.",
    "shifts": "marks a change in the topic's trajectory or status.",
}

DEFAULT_CONNECTION_TYPE = "supports"

# Every link the proposal reviewer makes is, by construction, an extension.
REVIEWER_CONNECTION_TYPE = "extends"


def connection_types_text() -> str:
    return "\n".join(f"- `{name}`: {text}" for name, text in CONNECTION_TYPES.items())


# ── 6. The prompts ───────────────────────────────────────────────────

def _enum(values) -> str:
    return " | ".join(f'"{v}"' for v in values)


def _fit_questions_json(indent: str = "    ") -> str:
    return ",\n".join(f'{indent}"{q}": <bool>' for q in FIT_QUESTIONS)


DECISION_RESPONSE_FORMAT = (
    "Respond with ONLY a JSON object (no markdown fencing):\n"
    "{\n"
    '  "candidate_classifications": [\n'
    "    {\n"
    '      "topic_id": "<id>",\n'
    '      "candidate_question": "<one phrase: the question this candidate tracks>",\n'
    '      "item_question": "<one phrase: the question this item bears on>",\n'
    f'      "label": {_enum(RELATIONSHIP_LABELS)}\n'
    "    }\n"
    "  ],\n"
    f'  "decision": {_enum(DECISIONS)},\n'
    '  "topic_id": "<existing topic_id if link_existing, null otherwise>",\n'
    f'  "connection_type": {_enum(CONNECTION_TYPES)} | null,\n'
    '  "fit_questions": {\n'
    f"{_fit_questions_json()}\n"
    "  },\n"
    '  "reasoning_trace": "<short explanation of the decision>",\n'
    '  "new_topic": {\n'
    '    "title": "<if propose_new>",\n'
    '    "summary": "<one sentence naming what is being investigated>",\n'
    '    "topic_id": "<slug>",\n'
    '    "tags": ["<...>"],\n'
    '    "suggested_directions": ["<...>"],\n'
    '    "scope_test_question": "<the single question this topic would track>",\n'
    '    "differentiation_from_closest": "<closest candidate topic_id and the concrete claim that makes this distinct, or \'none\' if no candidates>"\n'
    "  }\n"
    "}\n\n"
    "Include all evaluated candidates in candidate_classifications. Provide "
    "fit_questions only when decision is link_existing; omit or set to null "
    "otherwise. Provide new_topic only when decision is propose_new."
)


def _decision_steps(rubric: RubricConfig, step1_heading: str, step1_intro: str, with_notes: bool) -> str:
    """Steps 1 to 4, shared by the first-pass and cold-start prompts."""
    step1 = f"## {step1_heading}\n\n{step1_intro}\n\n{relationship_rubric_text(rubric)}\n\n"
    step1 += (
        "For each candidate, state the candidate's question and the item's "
        "question in one phrase each, then pick the label."
    )
    if with_notes:
        step1 += f"\n\n{REPRESENTATIVE_OBSERVATIONS_NOTE}"
    return (
        f"{step1}\n\n"
        f"## Step 2: Decide\n\n{DECISION_RULES}\n\n"
        f"## Step 3: Strength assessment (link_existing only)\n\n"
        f"{FIT_INSTRUCTIONS}\n\n{fit_questions_text()}\n\n{FIT_CLOSING}\n\n"
        f"## Step 4: Connection type (link_existing only)\n\n{connection_types_text()}"
    )


def synthesize_system_prompt(settings: Settings) -> str:
    """The first pass: classify every candidate, then link, propose or skip."""
    r = settings.rubric
    return (
        f"You are the analyst who maintains a topic store that follows {settings.domain}. "
        f"Items arrive from every kind of source: {r.source_kinds}. Given a new item "
        "and candidate topics it might relate to, decide how to connect them.\n\n"
        f"## What a right-sized topic is\n\n{right_sized_topic(r)}\n\n"
        f"## Reading the item by its kind\n\n{reading_by_kind(r)}\n\n{SYMMETRIC_FAILURES}\n\n"
        + _decision_steps(
            r,
            "Step 1: Classify each candidate",
            "For each candidate topic provided, choose the relationship label that "
            "best describes how the item relates to that topic. " + RELATIONSHIP_INSTRUCTIONS,
            with_notes=True,
        )
        + f"\n\n## Survey papers and comprehensive overviews\n\n{SURVEYS}\n\n"
        f"## Counter-examples worth holding in mind\n\n{counter_examples_text(r)}\n\n"
        f"## Response format\n\n{DECISION_RESPONSE_FORMAT}\n"
    )


def cold_start_system_prompt(settings: Settings) -> str:
    """The first pass on an empty store: only this batch's proposals to
    classify against."""
    r = settings.rubric
    return (
        f"You are the analyst who maintains a topic store that follows {settings.domain}. "
        "There are currently no established topics. Evaluate whether this item "
        "suggests a question worth tracking over time.\n\n"
        "If recently proposed topics from this batch are listed below, classify the "
        "item against each using the labels in Step 1 and apply the decision rules in "
        "Step 2. Otherwise, the choice is between propose_new and skip.\n\n"
        f"## What a right-sized topic is\n\n{right_sized_topic(r)}\n\n"
        + _decision_steps(
            r,
            "Step 1: Classify each batch-proposed candidate",
            "For each candidate (if any), choose the relationship label that best "
            "describes how the item relates to it. " + RELATIONSHIP_INSTRUCTIONS,
            with_notes=False,
        )
        + f"\n\n## Response format\n\n{DECISION_RESPONSE_FORMAT}\n"
    )


def confirm_system_prompt(settings: Settings) -> str:
    """The second, independent review of a propose_new decision."""
    r = settings.rubric
    return (
        "You are a second, independent reviewer for a topic store that follows "
        f"{settings.domain}. Items are {r.source_kinds}; a topic tracks one question "
        "across all of them. A first-pass analyst decided that a new item does not "
        "fit any existing topic and proposed a new topic for it. Your job is to "
        "check that call. You see the item, the closest existing topics, and the "
        "proposed topic.\n\n"
        "## Why this check exists\n\n"
        "The two possible mistakes are not symmetric. Linking an item to an existing "
        "topic that only partly covers it costs one slightly noisy observation, and "
        "the topic's summary can later be broadened. Creating a topic that should "
        "have been a facet of an existing one creates a permanent near-duplicate that "
        "splits future evidence, and most topics opened by mistake never receive a "
        "second observation. So the bar for a new topic is higher than \"the item's "
        "question differs from every candidate's question\".\n\n"
        "## The test\n\n"
        "A new topic is warranted only if BOTH hold:\n\n"
        "1. No candidate is the same *question* viewed from a different facet. Ask, "
        "for the closest candidate: if that topic's one-sentence summary were "
        "broadened by a clause, would it naturally cover this item? If yes, the item "
        "is an extension of that topic, not a new question. Different method, "
        "different artifact type, different source kind, different subdomain, or a "
        "narrower instance are all facets, not new questions.\n\n"
        "2. The proposed topic is right-sized on its own. "
        f"{right_sized_topic(r)} A topic that restates this one item's contribution, "
        "or that follows one product or one source rather than a question, fails "
        "this test.\n\n"
        "If (1) fails: `link_existing` to that candidate with connection_type "
        f"`{REVIEWER_CONNECTION_TYPE}`. If (1) holds but (2) fails: `skip`. Skip is a "
        "normal outcome for an above-threshold item that is interesting but not a "
        "direction. If both hold: `confirm_new`.\n\n"
        "You do not assess link strength. A link made here is by construction an "
        "edge extension of the topic and is scored as such downstream.\n\n"
        "## Response format\n\n"
        "Respond with ONLY a JSON object (no markdown fencing):\n"
        "{\n"
        '  "verdict": "confirm_new" | "link_existing" | "skip",\n'
        '  "topic_id": "<candidate topic_id if link_existing, else null>",\n'
        f'  "connection_type": {_enum((REVIEWER_CONNECTION_TYPE, DEFAULT_CONNECTION_TYPE))} | null,\n'
        '  "closest_candidate": "<topic_id of the closest candidate regardless of verdict>",\n'
        '  "broadened_summary": "<if link_existing: the candidate summary with the clause that covers this item; else null>",\n'
        '  "reasoning": "<two or three sentences>"\n'
        "}\n"
    )


def merge_system_prompt(settings: Settings) -> str:
    """Rewrite the survivor of a merge."""
    return (
        f"You are the analyst who maintains a topic store that follows {settings.domain}. Two "
        "topics have been identified as near-duplicates and are being merged into one "
        "surviving topic. You will see both topics (titles, summaries, representative "
        "observations) and your job is to write a unified title, summary, tag set, "
        "and suggested directions for the merged topic.\n\n"
        f"## What a right-sized topic is\n\n{right_sized_topic(settings.rubric)}\n\n"
        "## The merged topic\n\n"
        "The merged topic should:\n"
        "- Track the single specifiable question that both source topics were "
        "following from slightly different angles.\n"
        "- Be no broader than the union of the two source topics. Do not pull in "
        "adjacent territory.\n"
        "- Have a one-sentence summary that names what is being investigated, not "
        "just the subject area.\n"
        "- Have a tag set that is the union of both source tag sets, deduplicated.\n"
        "- Have suggested directions that capture the most useful open questions "
        "from both source topics, deduplicated.\n\n"
        "## Response format\n\n"
        "Respond with ONLY a JSON object (no markdown fencing):\n"
        "{\n"
        '  "title": "<concise title for the merged topic>",\n'
        '  "summary": "<one sentence naming what is being investigated>",\n'
        '  "tags": ["<deduplicated union of both source tag sets>"],\n'
        '  "suggested_directions": ["<deduplicated useful directions from both>"],\n'
        '  "rationale": "<one or two sentences explaining what the merged topic captures and what was preserved>"\n'
        "}\n\n"
        "The topic_id of the surviving topic is fixed by the caller; do not propose "
        "a new topic_id.\n"
    )


def split_propose_system_prompt(settings: Settings) -> str:
    """Propose the sub-topics a broad topic splits into."""
    return (
        f"You are the analyst who maintains a topic store that follows {settings.domain}. A "
        "topic has grown too broad and needs to be split into more specific "
        "sub-topics. You will be given the topic and a summary of its observations. "
        "Propose 2-4 more specific sub-topics that collectively cover the same space "
        "at a finer granularity.\n\n"
        "Each sub-topic must be right-sized on its own, a distinct question within "
        "the broader space rather than a catch-all.\n\n"
        f"## What a right-sized topic is\n\n{right_sized_topic(settings.rubric)}\n\n"
        "## Response format\n\n"
        "Respond with ONLY a JSON object (no markdown fencing):\n"
        "{\n"
        '  "sub_topics": [\n'
        "    {\n"
        '      "topic_id": "<slug, lowercase with hyphens>",\n'
        '      "title": "<concise title>",\n'
        '      "summary": "<2-3 sentence description of the specific question tracked>",\n'
        '      "tags": ["<relevant tags>"],\n'
        '      "suggested_directions": ["<specific open questions>"]\n'
        "    }\n"
        "  ]\n"
        "}\n"
    )


def split_reassign_system_prompt(settings: Settings) -> str:
    """Reassign a split topic's observations to its sub-topics."""
    return (
        f"You are the analyst who maintains a topic store that follows {settings.domain}. "
        "Observations are being reassigned from a broad parent topic to more specific "
        "sub-topics. For each observation, decide which sub-topic it best fits, then "
        "assess its fit to that narrower topic.\n\n"
        f"## Strength assessment\n\n{FIT_INSTRUCTIONS}\n\n{fit_questions_text()}\n\n"
        f"{FIT_CLOSING}\n\n"
        f"## Connection type\n\n{connection_types_text()}\n\n"
        "## Response format\n\n"
        "Respond with ONLY a JSON array (no markdown fencing) containing one entry "
        "per observation:\n"
        "[\n"
        "  {\n"
        '    "item_url": "<url of the original item>",\n'
        '    "new_topic_id": "<topic_id of the best-fitting sub-topic>",\n'
        '    "fit_questions": {\n'
        f"{_fit_questions_json(indent='      ')}\n"
        "    },\n"
        f'    "new_connection_type": {_enum(CONNECTION_TYPES)},\n'
        '    "reasoning": "<brief explanation>"\n'
        "  }\n"
        "]\n\n"
        "Every observation must appear exactly once. Do not drop any.\n"
    )


# Every prompt by name, for `topicsleuth show-prompts` and for tests.
PROMPTS = {
    "synthesize": synthesize_system_prompt,
    "cold_start": cold_start_system_prompt,
    "confirm": confirm_system_prompt,
    "merge": merge_system_prompt,
    "split_propose": split_propose_system_prompt,
    "split_reassign": split_reassign_system_prompt,
}
