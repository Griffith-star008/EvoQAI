# ADR 0003: Public API Governance & Stability Guarantee

**Status:** Accepted
**Date:** 2026-07-09

## Context and Problem Statement
As AQIP transitions from a research prototype to a global production framework, enterprise adopters and external researchers require strict guarantees regarding API backward compatibility. If internal compiler passes change arbitrarily, users' custom quantum-classical workflows will break.

## Decision
We adopt a strict 4-Tier API Governance Model enforced via Semantic Versioning (SemVer 2.0.0):
1. **Stable API (`aqip.core.*`):** Guaranteed backward compatibility within major version lifecycles. Any breaking changes require a 12-month deprecation warning.
2. **Experimental API (`aqip.experimental.*`):** Bleeding-edge algorithms (e.g., new self-evolving heuristics). No backward compatibility guaranteed.
3. **Internal API (`_aqip_internal.*`):** Hidden from public documentation. For framework developers only.
4. **Deprecated API:** Supported but throws `DeprecationWarning`.

## Alternatives Considered
- **No strict governance (Ad-hoc breaking changes):** Typical in early-stage research code, but completely unacceptable for a "Production" or "World-Class" standard.

## Consequences
- **Positive:** Enterprise confidence. Researchers can confidently build long-term projects on top of UAIR.
- **Negative:** Increased maintenance overhead for core maintainers; requires maintaining legacy wrappers during the deprecation period.
