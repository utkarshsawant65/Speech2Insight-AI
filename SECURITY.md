# Security Policy

## Reporting a Vulnerability

This is a solo/academic project. There is no formal security team, but vulnerabilities
are taken seriously.

**Please do not open a public GitHub issue for security vulnerabilities.**

Instead, report via [GitHub's private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing/privately-reporting-a-security-vulnerability)
if enabled for this repository.

Include:
- A clear description of the vulnerability
- Steps to reproduce
- Potential impact

You can expect an acknowledgment within 7 days. This project does not offer a bug bounty.

## Scope

Relevant areas include:
- Arbitrary file read/write via uploaded audio files
- Dependency vulnerabilities (monitored via Dependabot)
- Docker image security

## Supported Versions

Only the latest commit on `main` is actively maintained.
