# Security & Safety Policy — Resetting World

## Philosophy

Resetting World builds software, decentralized communication protocols, and open-hardware designs intended for high-stress disconnection scenarios, emergencies, and off-grid communities. Security bugs, cryptographic flaws, unsafe electrical specs, or dangerous medical/technical inaccuracies can put real people at risk.

## Reporting a Vulnerability

**Please DO NOT report security vulnerabilities through public GitHub issues.**

Report suspected vulnerabilities privately to the maintainers:
- **GitHub Security Advisories**: Open a private draft advisory under the repository's **Security -> Advisories** tab.
- **Details to include**: Affected repository (`rw-knowledge`, `rw-protocol`, `rw-hardware`, or `smart-local`), reproduction steps, threat model/impact, and suggested remediation if known.

We aim to acknowledge reports within **48 hours** and provide a coordinated remediation timeline within **7 days**.

## Scope by Pillar

1. **Knowledge (`rw-knowledge` / `smart-local`)**:
   - Arbitrary file read/write or path traversal during `.zip` / `.gpk` capsule import.
   - SQL injection or malformed vector/payload deserialization in local RAG APIs.
   - Localhost API binding exposure when serving on untrusted ad-hoc networks.
2. **Communication (`rw-protocol`)**:
   - Ed25519 signature bypass, replay attacks, or CBOR parser panics/DoS over LoRa/BLE.
   - DTN routing amplification loops or mesh flooding vulnerabilities.
3. **Hardware (`rw-hardware`)**:
   - Electrical fire/short-circuit hazards, missing fuse/BMS protections in LiFePO4 schematics, or unsafe thermal limits.

## Safety & Acceptable Use Guardrails

Resetting World strictly prohibits contributing or distributing:
- Actionable blueprints or guides for manufacturing weapons, explosives, or harmful agents.
- Unvetted medical advice presented as clinical fact without primary institutional sources.
- Telemetry, hidden cloud callbacks, or remote kill-switches (all code must uphold the **100% offline-first** guarantee).
