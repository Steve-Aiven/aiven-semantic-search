# Security

## Secrets

- **Never commit** `.env` files, service account JSON keys, or Aiven connection strings that include passwords.
- This repository **gitignores** `.env`, `part-1/.env`, and common credential filenames. Use `part-1/.env.example` as a template only.
- If a password or key was ever **pasted in chat, a screenshot, or a public log**, **rotate it** in the Aiven console and Google Cloud (as applicable) before you rely on that credential.

## Local development

- Prefer **`gcloud auth application-default login`** for Vertex AI (Application Default Credentials) over downloading long-lived key files to your machine.
- **Do not** set `verify_certs=False` for OpenSearch in production. This codebase keeps certificate verification on.

## GitHub

- Before `git push`, run `git status` and confirm you do not see `.env` or any `*.json` key files listed as staged.
- If you use CI, inject secrets from **GitHub Actions secrets** or a secret manager, not from the repo.

## Reporting

If you find a security issue in this tutorial repo, open a private report with the repository maintainers (or your org’s process).
