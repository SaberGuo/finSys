<!--
Sync Impact Report
Version change: template -> 1.0.0
Modified principles:
- Template Principle 1 -> I. Test-First Delivery (NON-NEGOTIABLE)
- Template Principle 2 -> II. Executable Planning Before Coding
- Template Principle 3 -> III. Traceable Requirements and Decomposition
- Template Principle 4 -> IV. Risk-Driven Integration Verification
- Template Principle 5 -> V. Simplicity with Explicit Trade-offs
Added sections:
- Engineering Quality Gates
- Delivery Workflow
Removed sections:
- None
Templates requiring updates:
- ✅ .specify/templates/plan-template.md
- ✅ .specify/templates/spec-template.md
- ✅ .specify/templates/tasks-template.md
- ⚠ pending: .specify/templates/commands/*.md (directory not present)
Follow-up TODOs:
- None
-->

# finSys Constitution

## Core Principles

### I. Test-First Delivery (NON-NEGOTIABLE)
All implementation work MUST begin with automated tests that express the expected
behavior before production code is written. For each requirement slice, teams MUST
show failing tests first, then implement, then refactor while keeping tests green.
At minimum, each feature MUST include unit tests plus integration or contract tests
for cross-boundary behavior. Rationale: defect prevention is cheaper than late
stabilization, and test-first flow keeps scope honest.

### II. Executable Planning Before Coding
Every feature MUST start with a decomposition that is directly executable: scoped
phases, task dependencies, parallel markers, file-level targets, and measurable
done criteria. Plans MUST break work into increments that can be validated in
isolation and delivered independently. Rationale: executable plans reduce ambiguity,
enable parallel execution, and improve delivery predictability.

### III. Traceable Requirements and Decomposition
Each requirement MUST map to at least one acceptance scenario, one validation path,
and one implementation task. Story priorities (P1, P2, P3...) MUST be explicit,
with each story independently testable as an MVP slice. Rationale: end-to-end
traceability prevents orphan work and makes review objective.

### IV. Risk-Driven Integration Verification
Integration risks MUST be identified during planning and covered by explicit tests
for interfaces, data contracts, failure paths, and dependency interactions. Any
contract or integration behavior change MUST include regression coverage before
merge. Rationale: most production incidents occur at boundaries, not isolated units.

### V. Simplicity with Explicit Trade-offs
Design and implementation MUST prefer the simplest solution that satisfies current
requirements. Added complexity (framework, abstraction, protocol, extra service)
MUST be justified in the plan with rejected alternatives and operational impact.
Rationale: simplicity improves maintainability and shortens feedback loops.

## Engineering Quality Gates

- A feature MUST NOT enter implementation until Constitution Check gates pass.
- Pull requests MUST include evidence of test execution and requirement traceability.
- Critical-path changes MUST include deterministic reproduction or verification
	steps in documentation.
- No placeholder tokens are permitted in approved spec, plan, or task artifacts.

## Delivery Workflow

1. Clarify scope and assumptions; record unresolved items explicitly.
2. Produce a plan with executable decomposition and constitution gates.
3. Define user stories and acceptance scenarios with independent test strategy.
4. Generate dependency-ordered tasks that start with tests and enforce failure-first.
5. Implement in priority order, validating each story independently before expansion.

## Governance

This constitution overrides conflicting local workflow guidance for planning,
testing, and delivery quality. Amendments require:

1. A documented proposal describing intent, scope, and migration impact.
2. Review and approval by project maintainers.
3. Synchronization updates to impacted templates and command guidance.

Versioning policy:

- MAJOR: backward-incompatible governance or principle removals/redefinitions.
- MINOR: new principle/section or materially expanded mandatory guidance.
- PATCH: wording clarifications, typo fixes, and non-semantic edits.

Compliance review expectations:

- Every plan MUST pass Constitution Check before Phase 0 research begins.
- Every task list MUST reflect executable decomposition and required testing tasks.
- Reviewers MUST block merges that violate non-negotiable principles.

**Version**: 1.0.0 | **Ratified**: 2026-04-28 | **Last Amended**: 2026-04-28
