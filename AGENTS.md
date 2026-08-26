# AGENTS.md — agent-loop-benchmark

Instructions for AI coding agents working in this repository. Follow these conventions for all changes.

## Project Overview

**agent-loop-benchmark** is a standardized benchmark that measures agent runtime quality and degradation at scale.

The benchmark runs scenario suites across escalating conversation-session counts — **10, 100, 500, and 1000 sessions** — to measure where agent quality breaks down as context and reference volume grow. It is designed to support multiple agent runtimes, including **OpenClaw, Hermes, Nanobot, and Driftless**.

The project is **MIT licensed**. See [LICENSE](./LICENSE).

## Repository Structure

> The repository is in its initial scaffolding stage. The directories below describe the intended layout once scaffolded.

- **Scenario fixtures** — declarative scenario definitions that agents execute against. New scenarios **must accumulate references across sessions** (see [Contributing](#contributing)) to exercise agent memory at scale.
- **Runner** — orchestrates scenario execution across the session counts above for each supported runtime.
- **Results collector** — gathers runtime outputs, quality signals, and degradation metrics from each run.
- **Reporting** — aggregates and renders benchmark results for comparison across runtimes and session scales.

## Git Workflow

1. **Push to a branch**: Create a feature branch from `main`, push it to the remote, and open a PR against `main`.
2. **Enable auto-merge**: Enable auto-merge on the PR so it merges to `main` automatically once any checks pass. No manual merge step is required.
3. **CI checks**: CI checks are not configured yet but may be added in the future. When they are, auto-merge will gate on them automatically — no workflow change needed.

## Contributing

1. **Fork** the repository and create a feature branch from `main`.
2. Make your changes, keeping **PRs focused and small**.
3. **Open a PR** against `main` describing the change.
4. **Enable auto-merge** on the PR so it merges to `main` automatically once checks pass — no manual reviewer-merge step (see [Git Workflow](#git-workflow)).

### Code Style & PR Expectations

- Match the existing style of any file you touch; do not reformat unrelated code in the same PR.
- Keep PRs focused and small — one concern per PR makes review faster and safer.
- New scenarios **must accumulate references across sessions** so the benchmark can test agent memory at scale. A scenario that resets or discards accumulated references does not meet this requirement.

## CI/CD

**No CI/CD pipeline is configured yet.** There is no automated build, test, or lint pipeline on this repository. Auto-merge is enabled on PRs; when CI checks are added in the future, auto-merge will gate on them automatically. Until then, ensure your change builds and passes locally before opening a PR.

## License

This project is licensed under the **MIT License** (see [LICENSE](./LICENSE)). By contributing, you agree that your contributions are licensed under the MIT license.