# Runtime adapter template

Copy this entire directory; it is a project, not a working adapter. All four
methods deliberately raise `NotImplementedError` until implemented.
Python 3.11 or later is required. The scaffold has no runtime dependencies
outside the standard library.

## Start

From the benchmark checkout, use an absolute path for the scaffold dependency:

```sh
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -e .
cp -R templates/runtime-adapter /tmp/my-runtime-adapter
cd /tmp/my-runtime-adapter
python3 -m pip install --no-deps -e .
python3 conformance.py
```

The last command must exit 1 with:

```text
FAIL: contract version: Implement initialize(runManifest): validate contractVersion and reset ALL run state
```

After implementation, the same command must pass. Rename the project in
`pyproject.toml` before distributing it. `almm-adapter==1.0.0` must be installed
from the benchmark checkout; this package is not assumed to exist on PyPI.

## Implement each method

Use `almm_adapter.contract` for types and boundary validators. See
`almm_adapter.reference.ReferenceAdapter` for a complete, minimal example.

- **initialize(runManifest)**: call `validate_manifest`; reject a missing or
  incompatible `adapter.contractVersion` before accepting a run. The supported
  version is `1.0`. Receive pinned model/tokenizer configuration and stable
  instructions from the manifest. Reset all prior facts, memories, summaries,
  caches, session IDs, counters, persistent run storage, and request telemetry.
  Return `None`. Storage may be unbounded, but must be isolated by run.
- **handleTurn(turn)**: call `validate_input(turn, 'turn')`. Receive only
  `turnId`, optional `sessionId`, `role: 'user'`, and `text`. Assemble the entire
  model request, including stable instructions, selected memory, recent history,
  and the current turn. Send it through the harness ModelProxy, not directly to
  a provider. Return `{'response': str, 'requests': [ModelRequest, ...]}`.
- **answerProbe(probe)**: call `validate_input(probe, 'probe')`. Receive only
  `probeId` and `question`. Return `{'answer': str, 'requests': [...]}`. Never
  accept expected answers, match types, gold IDs, or a full scorer fixture.
- **getRequestTelemetry()**: return an independent snapshot of every assembled
  model request in chronological order since initialization, including attempts
  rejected by the model proxy. Return an empty list immediately after reset.
  A single turn or probe may generate multiple model requests; report all of them.

Each request has a run-unique `requestId` and a nonempty ordered `segments` list.
Every segment has `tier`, string `content`, nonnegative integer `tokenCount`, and
optional `sourceIds: list[str]`. Tiers are `stable`, `semi-stable`, and `unstable`.
Use the manifest's declared tokenizer; do not reuse the reference smoke
character counter for real runs. Segment content must include role framing and
all instructions that reach the model, not just remembered facts. The harness
verifies the complete request against its 25,000-token budget; no tier quotas
are imposed. Source provenance is optional, not a scoring prerequisite.

## Separate-process HTTP

From this project, using the same activated environment:

```sh
python3 -m almm_adapter serve --factory adapter:create_adapter --port 8765
```

In another terminal with the environment activated:

```sh
python3 conformance.py --url http://127.0.0.1:8765
```

The same conformance checks run locally and over HTTP. The HTTP client does not
import your adapter. Adapters written in other languages can implement the
HTTP protocol documented in the benchmark root README. HTTP is the selected
transport; gRPC is not implemented.

## Reset verification

The suite checks cleared telemetry and absence of prior-run sentinel facts and
source IDs in new answers/requests. This cannot prove that inaccessible storage
was physically deleted. Add runtime-specific tests for all memory stores and
session state. An adapter retaining old data but not exposing it during the
check is still nonconforming.

## Isolation

Run untrusted adapters as separate processes under OS/container restrictions:
no network except the harness model proxy, no access to scorer data, and a
run-scoped writable storage directory. HTTP separates processes; it is not an
OS security sandbox. Never load an untrusted factory into the harness process.
