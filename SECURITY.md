# Security Policy

## Reporting a vulnerability

Do not open a public issue for an undisclosed vulnerability. Use the
repository's **Security** tab and GitHub private vulnerability reporting to
send the maintainers a private report. Include the affected version or commit,
reproduction steps, impact, and any proposed mitigation.

The maintainers will acknowledge the report through that private channel,
investigate it, and coordinate disclosure after a fix or mitigation is ready.
Please avoid publishing exploit details until that process is complete.

## Artifact trust

Some scientific workflows load model files through pickle-compatible formats
such as joblib. These formats can execute code during deserialization. Only
load artifacts from trusted sources, verify published checksums, and review the
artifact's provenance and redistribution terms.
