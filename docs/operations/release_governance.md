# Release Governance

To ensure stability across the ecosystem, AQIP adheres to strict release governance.

## 1. Semantic Versioning
AQIP uses strictly adhered Semantic Versioning (`MAJOR.MINOR.PATCH`).
- **MAJOR:** Incompatible API changes (e.g., modifying the UAIR graph schema).
- **MINOR:** Backwards-compatible functionality additions (e.g., adding a new backend simulator).
- **PATCH:** Backwards-compatible bug fixes (e.g., fixing an SMT solver timeout bug).

## 2. Release Schedule
- **Patch Releases:** Bi-weekly (every other Tuesday).
- **Minor Releases:** Quarterly (March, June, September, December).
- **Major Releases:** Annually (as dictated by major theoretical advancements).

## 3. Compatibility Policy
- The `aqip.core` API is guaranteed stable across Minor releases.
- Compiled UAIR graphs (`.uair` files) are guaranteed compatible across the entire Major version lifecycle.
- Checkpoints from version `X.Y` can always be loaded in version `X.(Y+1)`.

## 4. Deprecation Policy
1. An API is marked as `@deprecated` in code and logs a warning at runtime.
2. The deprecation is announced in the `CHANGELOG.md` of a Minor release.
3. The API remains fully functional for at least **two Minor releases** (6 months).
4. The API is removed in the next Major release.

## 5. Artifact Verification
Every release must include:
1. `SBOM.json` (Software Bill of Materials).
2. Signed Docker images published to the official registry.
3. Checksums (`SHA256`) for all binary artifacts.
