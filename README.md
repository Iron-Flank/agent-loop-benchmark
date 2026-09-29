# agent-loop-benchmark

ALMM measures runtime memory quality as conversation histories accumulate.
This checkout supplies the **RuntimeAdapter 1.0 contract** and a deterministic
fixture generator. The full benchmark harness, scoring, and participating
runtimes are separate tasks. The offline adapter smoke command below exercises
the adapter lifecycle, not a scored benchmark or a claim of model accuracy.

## Quick start

Python 3.11 or later; no runtime dependencies outside the standard library.
From the checkout:

```sh
python3 -m almm_adapter --smoke
python3 -m almm_adapter conformance
python3 -m unittest discover -s tests -v
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
python3 -m pip install -e '.[dev]'
ruff check almm_fixture almm_adapter tests templates/runtime-adapter
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
the future harness must copy it into its run manifest. Output-changing updates
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

**BCH-013 ac-2 is blocked by the specified bound:** in exactly 1000 sessions,
the maximum prior-session distance is 999, not 1000. The generator honestly
covers distances 10, 100, 500, and 999; it does not relabel 999 as 1000 or add
a hidden session. A human must authorize a 1001-session fixture, a separate
prelude, or a corrected distance before the task can pass this criterion.

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
Model and tokenizer objects carry the harness's pinned configurations; this
adapter scaffold does not define the complete future harness manifest schema.
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
