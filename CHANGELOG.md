# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Root package metadata and a reproducible development lockfile.
- Public CI lanes for the shared runner and the three built-in task packages.
- Project license, contribution, security, conduct, and citation guidance.
- Draft experiment contracts for all three built-in tasks.
- Integrity verification and trust documentation for the external G12D joblib
  model artifact.
- Automatic manifest-declared dependency preflight before non-mock runs.

### Changed

- Clarified that v0.1 is a release candidate and that built-in campaign
  contracts remain drafts pending qualification evidence.
- Removed the G12D joblib binary from Git while retaining its checksum and
  provenance metadata.
- Consolidated project-specific technical guides under `docs/` while retaining
  conventional public project and governance files at the repository root.
- Removed the protein inverse-folding task until its public contract and
  dependencies are ready.
- Corrected stale module paths and local-only artifact metadata.

[Unreleased]: https://github.com/yzailab/Large-Discovery-Models/commits/ldm_engine
