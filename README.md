# agent-loop-benchmark

ALMM measures runtime memory quality as conversation histories accumulate.
This checkout supplies the **RuntimeAdapter 1.0 contract**, deterministic fixture
generator, and core execution harness. Scoring and participating runtimes are
separate tasks. Offline smoke responses are not scored benchmark answers or a
claim of model accuracy.

## Frozen scorer calibration content

[`calibration/scorer-v1.0.0`](calibration/scorer-v1.0.0) contains the BCH-014
calibration corpus, independently authored from the fixture generator's topic
and lifecycle patterns, not sampled from canonical fixtures:

- `probes.json`: 100 manually agent-labeled candidates, 20 per primary ability.
  Match types: 40 semantic, 16 exact, 16 numeric, 8 ordered-list, 20 abstain.
- `review.json`: second-pass review of all 100 labels, coverage counts, and
  canonical-separation evidence. Author and reviewer are the same agent; this
  is not a claim of independent human annotation.
- `manifest.json`: frozen file SHA-256 digests and scorer-version association.
  **Scorer 1.0.0 is reserved for BCH-007**, not an implemented or calibrated
  scorer. BCH-007 owns the actual agreement gate; no 95% agreement is claimed here.

Each calibration row includes a question, candidate answer, expected record,
gold judgment, justification, edge-case tags, and an independently authored
miniature evidence fixture (`evidence`, `afterSessionIndex`, `answerAsOf`).
Evidence facts have local IDs, introduction session indices, text, and lifecycle
states. They remain available across sessions; expired/superseded facts support
historical questions but not current answers. These calibration envelopes are
not canonical replay fixtures and must not be passed to `SchemaValidator` or a
runtime adapter. Generator 1.0.0 emits only exact, ordered-list, and abstain;
the independently authored numeric/semantic records extend its core expected
fields to cover the initiative's complete match taxonomy.

Expected-record rules:

- `requiredFactIds` and `forbiddenFactIds` resolve within the row's evidence.
- Exact answers trim surrounding whitespace only; case and punctuation matter.
- Ordered-list answers are JSON arrays; all items must match in the declared order,
  without missing or extra items. `acceptedAnswers` retains the fixture format
  of JSON-array strings; `items` is the decoded comparison target.
- Numeric answers are single decimal numbers in the question's units.
  `targetNumber` and `tolerance` are decimal strings. `absolute-inclusive` means
  `abs(candidate - target) <= tolerance`, including both tolerance edges.
- Semantic answers must convey every `requiredClaims` entry and assert no
  `disallowedContradictions`; the versioned `rubric` states the matching policy.
- Gold `abstain` means the candidate declines without a substantive answer,
  **not** that the answer is necessarily correct. With `answerability: false`
  it is a correct abstention; with evidence it is a false abstention. A hedged
  factual guess without evidence is `fail`. Preserve the tri-state labels when
  measuring judge agreement; report abstention correctness separately.

The set includes 8 each of semantic near misses, correct paraphrases,
contradictions, false abstentions, and numeric tolerance boundaries; 10 each of
correct abstentions and unsupported answers; and 2 each of list permutations,
partial lists, extra-item lists, and correct lists. A second pass confirmed every
label and justification, including independent decimal arithmetic for boundaries.
Comparison against all 11,989 possible generator-1.0.0 question strings establishes
zero identical canonical probes for any public or held-out seed, without reading
private fixtures or seeds. Recheck separation when generator templates change.

Do not rewrite these files when changing the scorer. Retain the frozen labels
with scorer 1.0.0; later scorers reference the same set. A label correction requires
an additive calibration-set version, retaining the previous files and association.
The manifest hashes detect edits; they do not provide filesystem write protection.

Reproduce the offline content smoke check from the checkout (no model or API key):

```sh
python3 - <<'PY'
from collections import Counter
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from almm_fixture.topics import TopicPool

root = Path("calibration/scorer-v1.0.0")
manifest = json.loads((root / "manifest.json").read_text())
for name, digest in manifest["files"].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
data = json.loads((root / "probes.json").read_text())
review = json.loads((root / "review.json").read_text())
probes = data["probes"]
assert data["frozen"] and len(probes) == 100
assert len({p["probeId"] for p in probes}) == 100
assert len(review["records"]) == 100 and review["result"] == "passed"
abilities = Counter(p["ability"] for p in probes)
matches = Counter(p["expected"]["matchType"] for p in probes)
edges = Counter(tag for p in probes for tag in p["edgeCases"])
assert len(abilities) == 5 and min(abilities.values()) >= 10
assert set(matches) == {"exact", "numeric", "ordered-list", "semantic", "abstain"}
assert matches["semantic"] >= 0.30 * len(probes)
for tag in ("near-miss", "paraphrased-correct", "contradictory",
            "correct-abstention", "false-abstention", "unsupported-answer",
            "numeric-tolerance-boundary", "ordered-list-permutation",
            "ordered-list-partial", "ordered-list-extra"):
    assert edges[tag] >= 2, tag
for p, checked in zip(probes, review["records"]):
    assert checked["probeId"] == p["probeId"]
    assert checked["confirmedGoldJudgment"] == p["goldJudgment"]
    assert checked["justificationConsistent"] and checked["evidenceConsistent"]
    assert all(p[k] for k in ("question", "candidateAnswer", "expected",
                              "goldJudgment", "justification"))
    e = p["expected"]
    ids = {f["factId"] for f in p["evidence"]}
    assert set(e["requiredFactIds"] + e["forbiddenFactIds"]) <= ids
    assert not set(e["requiredFactIds"]) & set(e["forbiddenFactIds"])
    assert all(f["sessionIndex"] < p["afterSessionIndex"] for f in p["evidence"])
    if e["matchType"] == "numeric":
        good = abs(Decimal(p["candidateAnswer"]) - Decimal(e["targetNumber"])) <= Decimal(e["tolerance"])
        assert p["goldJudgment"] == ("pass" if good else "fail")
    elif e["matchType"] == "exact":
        good = p["candidateAnswer"].strip() in e["acceptedAnswers"]
        assert p["goldJudgment"] == ("pass" if good else "fail")
    elif e["matchType"] == "ordered-list":
        good = json.loads(p["candidateAnswer"]) == e["items"]
        assert p["goldJudgment"] == ("pass" if good else "fail")
    elif e["matchType"] == "abstain":
        assert not p["answerability"] and not ids
        good = p["candidateAnswer"] in e["acceptedAnswers"]
        assert p["goldJudgment"] == ("abstain" if good else "fail")
    else:
        assert e["rubric"] and e["requiredClaims"] and e["disallowedContradictions"]
questions = {"What is my emergency verification phrase? I have not shared it."}
topics = TopicPool().topics
for t in topics:
    questions.update((f"What is the {t.slot_name} for {t.label}?",
                      f"Before it expired, what was the {t.slot_name} for {t.label}?",
                      f"What is the current {t.slot_name} for {t.label}, after the change?"))
    for other in topics:
        questions.add(f"List the values for {t.label} and {other.label}, in that order.")
assert not questions & {p["question"] for p in probes}
print("PASS:", len(probes), "probes;", dict(abilities), dict(matches))
print("Frozen hashes, review coverage, atomic labels, edge coverage and canonical separation: PASS")
PY
```

This smoke check corroborates structure and deterministic labels; semantic label
quality still depends on reviewing the evidence, rubrics, candidates, and
justifications. It does not call or calibrate a semantic judge.

## Quick start

Python 3.11 or later. Adapter smoke and fixture generation use only the standard library.
From the checkout:

```sh
python3 -m almm_adapter --smoke
python3 -m almm_adapter conformance
```

Smoke runs ten sequential sessions and twenty model requests using a deterministic
offline proxy. It verifies that the reference adapter still includes session-1
context at session 10. Its character-count tokenizer is smoke-only, not a model
tokenizer. No API key or cloud service is used.

To install the CLI and make the package available outside the checkout:

```sh
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -e .
almm-adapter conformance
```

For development checks in the activated environment:

```sh
python3 -m pip install -e '.[dev,model]'
ruff check almm_fixture almm_adapter almm_harness tests templates/runtime-adapter
coverage run -m unittest discover -s tests -v
coverage run -m almm_adapter conformance
coverage run -m almm_adapter --smoke
coverage combine
coverage report -m
python3 -m build
```

Coverage includes subprocess servers and enforces an 80% threshold. The
repository lint configuration uses Ruff's default correctness rules.

## Deterministic fixture generation

Generator **1.0.0**, schema **1.0**: CPU-only, no model calls or API keys.
The public development set is in [`fixtures/v1.0.0`](fixtures/v1.0.0), with
seed 42 at 10/100/500/1000 sessions. Each session contains ten distinct topics.
The pool has 108 curated topics across people, places, preferences, events,
decisions, and quantities. At least 20% of visited topics recur across sessions.
Follow-up turns reference earlier facts without restating their values.

```sh
python3 -m almm_fixture generate --sessions 10 --seed 42 --output /tmp/almm-dev-10.json
python3 -m almm_fixture validate /tmp/almm-dev-10.json
# Installed entry point: almm-fixture generate ...
```

Outputs are immutable: choose a new path rather than overwriting an old fixture.
All randomness comes from the seed. JSON uses sorted keys, compact separators,
UTF-8, and a trailing newline. `contentHash` is SHA-256 of that serialization
with the `contentHash` field omitted. The fixture records `generatorVersion`;
the harness copies it into its run manifest. Output-changing updates
require a new generator version and additive fixture directories.

`FactGraph` retains active, superseded, and expired facts. `invalidatedAt`
identifies the turn at which a value stopped being current. Probe answers are
assembled against their checkpoint, not the final ledger alone. Historical
probes may retrieve expired/superseded values; current-value probes cannot.
`answerAsOf` distinguishes those cases. Knowledge-update records forbid the
old supersession chain. Cross-session ordered-list answers use values from
different introduction sessions, in required-fact ID order. Abstention probes
have `answerability: false`, no gold factual evidence, and an abstention answer.
The validator rejects malformed IDs/references, unresolved supersession,
premature probes, and ledger-inconsistent answers before any output is created.

Cross-session probes cover distances 10, 100, 500, and 999. The maximum
prior-session distance within exactly 1000 sessions is 999: a fact introduced
in session 1 and probed after session 1000. No extra prelude session or
distance relabeling is used.

### Held-out isolation

Development seeds are nonnegative integers below 1,000,000,000; the reserved
evaluation range starts at 1,000,000,000. Actual evaluation seeds are supplied
privately and must never be committed. Values used by automated isolation
tests are synthetic test-only seeds, not evaluation-set seeds.

Provide a private JSON file with one integer `seed` field. Both its parent
directory and the output parent must already exist with 0700 permissions;
the seed file must have 0600 permissions. Paths inside a public checkout are
rejected, including paths resolving through symlinks into it.

```sh
python3 -m almm_fixture generate --sessions 1000 \
  --held-out-seed-file /private/almm-evaluation/seed.json \
  --output /private/almm-evaluation/v1.0.0-run
```

The new 0700 output directory contains two 0600 files:
- `adapter.json`: only session IDs/timestamps, user turn IDs/roles/text, and
  probe IDs/checkpoints/questions. No seed, fact ledger, expected answers,
  match types, or gold fact IDs.
- `scorer.json`: complete validated fixture, including seed, version, hash,
  expected-answer records, and gold evidence.

Only `adapter.json` crosses the harness input boundary. For RuntimeAdapter 1.0,
send turns as their allowlisted turn objects and probes as
`{probeId, question}`; the checkpoint is harness scheduling metadata.
Filesystem permissions are not an adapter sandbox: run untrusted adapters
without access to the private scorer directory.

## Core harness

`FixtureRunner(manifest, adapter, proxy, artifact_dir).run(fixture, resume=False)`
executes validated fixtures sequentially, inserting probes after their declared
sessions. It initializes a clean adapter for each fixture/run pair and passes
only allowlisted turns and probe questions. Scoring remains a separate pipeline;
`eligibleForAccuracy` identifies answered probes, not correct answers.

Install `.[dev,model]` for the pinned tiktoken loader and tokenizer tests. This
offline example exercises the generated ten-session fixture without an API key:

```sh
python3 - <<'PY'
from importlib.metadata import version
from tempfile import TemporaryDirectory
from almm_adapter.reference import ReferenceAdapter
from almm_fixture.engine import generate
from almm_harness.budget import BudgetVerifier
from almm_harness.proxy import ModelProxy
from almm_harness.runner import FixtureRunner
from almm_harness.tokenizers import load_tokenizer

manifest = {
    'runId': 'offline-example',
    'adapter': {'name': 'reference', 'revision': '1.0.0', 'contractVersion': '1.0'},
    'model': {'provider': 'offline', 'name': 'smoke', 'version': '1',
              'decoding': {'temperature': 0}, 'seed': 42},
    'tokenizer': {'name': 'tiktoken:cl100k_base', 'version': version('tiktoken')},
    'stablePrefix': ['Benchmark contract.', 'SOUL.', 'Agent.', 'User.'],
    'seed': 42, 'scorerVersion': 'unscored', 'rateLimitRpm': 1000000,
    'probeTimeoutSeconds': 10,
}
count = load_tokenizer(manifest['tokenizer'])
proxy = ModelProxy(manifest, BudgetVerifier(manifest, count),
                   lambda payload: {'answer': 'Offline smoke answer'})
with TemporaryDirectory() as directory:
    adapter = ReferenceAdapter(proxy, count)
    result = FixtureRunner(manifest, adapter, proxy, directory).run(generate(42, 10))
    assert result['results']['answeredCount'] == 5
    print(result['results'])
PY
```

For real calls, inject `OpenAICompatibleProvider(endpoint=..., api_key_env=...)`.
Credentials come only from the named environment variable. The transport sends
`model.version` as the wire model, the pinned decoding settings and optional
model seed, with exactly concatenated segment content in one user message.
Declare `tokenizer.requestOverheadTokens` for that provider's chat framing.
The budget verifier retokenizes the complete wire content plus this overhead;
adapter counts cannot bypass the 25,000-token ceiling. Segment and per-tier
counts are recorded independently, without per-tier quotas. The immutable stable
prefix must lead every request; its content hash is recorded.

Each proxy has an independent, thread-safe token bucket (default 60 requests/min,
initial burst equal to bucket capacity). Only 429/503 errors retry: three retries
with 1/2/4-second backoff, at most 30 seconds of retry backoff and quota waiting.
Initial throttling is not capped by the retry wait limit. Identical scored
requests always reach the provider. Provider token usage and version are recorded
when supplied; version drift is rejected.

Artifacts live under a run-ID/configuration-hash directory: immutable
`manifest.json`, `requests.jsonl`, `probes.jsonl`, `events.jsonl`, and
`dead-letters.jsonl`. The manifest includes actual fixture/harness hashes,
adapter/runtime revision, pinned model/tokenizer/decoding, seeds, timestamp,
generator/scorer version, stable-prefix hash, and every request's tier totals.
Probe records retain raw answers, request telemetry, latency and optional source
IDs. Dead letters preserve redacted request context, errors and retry history;
they are not re-scored. Final artifacts cannot be overwritten: use a new run ID.

Atomic `checkpoint.json` records the last fully completed session and stream byte
offsets. After interruption, a fresh runner/adapter with the same manifest and
fixture calls `run(fixture, resume=True)`. Partial-session records are truncated;
completed sessions reconstruct adapter memory from verified archived answers,
without new provider calls or scoring. Execution then starts at the next
session. This recovery-only transcript replay is not response caching during
scored execution. Invalid checkpoints or unverifiable reconstruction restart
at session 1; configuration/fixture mismatch rejects resume.

Budget, provider, adapter and timeout failures are reported separately and never
count as incorrect answers. Accuracy denominators include answered probes only.
More than 5% provider/adapter/timeout failures flags investigation without
invalidating the run; budget failures remain a separate operational metric.
Hard probe timeout terminates the fixture and forbids reuse of its adapter/proxy,
because Python cannot safely kill arbitrary adapter threads. Create fresh
instances and a new run ID after timeout; unattempted probes are reported
separately. Only trusted adapters may run in-process.

## Contract and reference adapter

[`almm_adapter/contract.py`](almm_adapter/contract.py) defines JSON-compatible
types and validators. The four required methods are:

| Method | Input | Result |
| --- | --- | --- |
| `initialize(runManifest)` | Run configuration | `None` / JSON `null`; fully reset run state |
| `handleTurn(turn)` | User turn | `response` string and `requests` list |
| `answerProbe(probe)` | Probe ID and question only | `answer` string and `requests` list |
| `getRequestTelemetry()` | None | Independent snapshot of all requests since initialization |

The manifest declares `adapter: {name, revision, contractVersion: "1.0"}` and
includes `runId`, `model`, `tokenizer`, and `stablePrefix: list[str]`.
Model and tokenizer objects carry the harness's pinned configurations; the core
harness also requires a seed and scorer version.
Missing and incompatible contract versions are rejected before run initialization.
[`sample_manifest()`](almm_adapter/conformance.py) supplies offline examples.

Each model request is `{requestId, segments: [...]}` with unique IDs within a run.
Segments preserve model-request order and have:

- `tier`: `stable`, `semi-stable`, or `unstable`.
- `content`: the complete segment text, including role framing.
- `tokenCount`: nonnegative integer, calculated using the declared tokenizer.
- Optional `sourceIds`: fixture turn/fact IDs for provenance.

Every turn/probe returns all model requests made for that operation. The telemetry
method includes every assembled attempt, even if its proxy call fails.
The stable prefix must remain unchanged within a run. Not every request needs
all three tiers; there are no per-tier quotas.

[`ReferenceAdapter`](almm_adapter/reference.py) accepts an injected
`model_proxy(request) -> str` and `count_tokens(content) -> int`. It retains full
conversation history in memory and puts it in the unstable tier; it has no
summary tier. This is an implementation example, not a scalable baseline.
Actual runs must inject the harness proxy and declared tokenizer. The harness,
not this adapter, enforces the 25,000-token ceiling on the complete request.
Unbounded full-history requests will eventually fail that budget.

Adapter input is an allowlisted projection: turns contain `turnId`, optional
`sessionId`, `role: "user"`, and `text`; probes contain `probeId` and `question`.
Never send an entire fixture, expected answers, match types, or gold evidence IDs.
Boundary validators reject extra fields rather than silently forwarding scorer data.

## Separate-process HTTP

Start the reference adapter:

```sh
python3 -m almm_adapter serve --port 8765
```

In another terminal:

```sh
python3 -m almm_adapter conformance --url http://127.0.0.1:8765
python3 -m almm_adapter --smoke --url http://127.0.0.1:8765
```

Only trusted adapters may run in-process. For an untrusted adapter use
`HttpAdapter(url)`; the harness must never import its implementation.
The server binds to loopback and processes operations sequentially. The HTTP
client has a 30-second default timeout, configurable in its constructor.
HTTP is the scaffold's chosen HTTP/gRPC transport; gRPC is not implemented.

### Wire contract

Use `POST` with `Content-Type: application/json`:

| Path | JSON body | JSON response |
| --- | --- | --- |
| `/v1/initialize` | Run manifest | `null` |
| `/v1/handleTurn` | Turn | Turn result |
| `/v1/answerProbe` | Probe | Probe result |
| `/v1/getRequestTelemetry` | `null` | Request list |

The body/result uses the same contract as in-process adapters. Errors are
`{"error": "diagnostic"}`: 400 for invalid input/contract violations, 404 for an
unknown method, 501 for unimplemented template methods, and 500 for adapter
execution errors. No dynamic method names outside the four allowed methods are
dispatched. Incoming request bodies are limited to 16 MiB. Telemetry responses
include the full archive and have no size cap; size the adapter/client process
memory limits for the selected run.

Process separation is not a security sandbox. Launch untrusted adapter processes
under OS/container controls with no access to scorer files, a fresh run-scoped
storage directory, and network access limited to the model proxy. The loopback
development server is unauthenticated; do not expose it as a public service.
Provider credentials must not appear in manifests or conversation telemetry.

## Add a runtime

Copy [`templates/runtime-adapter`](templates/runtime-adapter), follow its
[README](templates/runtime-adapter/README.md), and implement all four methods.
The template deliberately fails conformance until implemented, with a diagnostic
identifying the missing method.

Run `python3 -m almm_adapter conformance --factory your_module:create_adapter`
for a trusted local factory, or the HTTP command above for a separate process.
The reusable `check_adapter(adapter)` suite checks method callability/signatures,
version rejection, chronological lifecycle, every segment's tier and token
metadata, complete request telemetry, stable-prefix consistency, and observable
run reset. Failure diagnostics include the method/check and segment field path.

The generic suite checks that prior-run sentinel facts and source IDs do not
reappear in fresh answers or assembled requests. It cannot prove deletion of
private storage that an adapter does not expose. Runtime authors must also test
reset of their own facts, memory indexes, summaries, caches, and session state.

https://github.com/Iron-Flank/driftless-agent
