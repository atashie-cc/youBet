# ADR 001: Initial Architecture

## Status
Accepted

## Context
Building a general-purpose prediction/betting framework, starting with NCAA March Madness 2026 as the first workflow. Need architecture that supports multiple prediction domains without over-engineering.

## Decision
- **Monorepo with core library + workflow directories**: `src/youbet/core/` provides domain-agnostic components. `workflows/{domain}/` contains domain-specific pipelines.
- **Config-driven workflows**: Each workflow uses `config.yaml` for all parameters. No magic numbers in code.
- **XGBoost on stat differentials as baseline**: Proven by multiple independent sources (77-90% accuracy).
- **Log loss as primary metric**: Calibration > accuracy for betting ROI.
- **Isotonic regression calibration**: Standard post-processing step.
- **60/20/20 temporal split**: Prevents data leakage from future seasons.
- **Kelly Criterion for bet sizing**: Quarter Kelly as default to reduce variance.

## Consequences
- New prediction domains follow the template workflow structure
- Core library stays domain-agnostic; domain-specific logic lives in workflows
- Research findings are documented and evolve with the project
