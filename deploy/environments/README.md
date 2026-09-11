# Per-environment configuration

Three templates — `development.env.example`, `staging.env.example`,
`production.env.example` — one per stage of the `development → staging → production`
promotion path (`docs/deployment.md`). Each documents the **non-secret** settings that
legitimately differ by environment (log format, reload, rate limits, resource-adjacent
timeouts, ...).

**Every secret-shaped field in these files is intentionally left blank**, with a comment
explaining where its real value comes from. That is not an oversight to fix by filling one
in — copying a real secret into a file under version control is exactly what this phase's
"never place secrets in the GitHub repository" requirement forbids. See
[`../../docs/deployment.md`](../../docs/deployment.md) §Secrets management for where each
blank field's real value is actually supplied (GitHub Environment secrets for CI/CD itself;
an env file placed on the target host outside this repo, or a cloud secrets manager, for the
running application).

## Usage

These are templates, not files any process reads directly:

- **Local development**: copy `../../.env.example` (not these) to `../../.env` — see the
  root README's "Install" section. `development.env.example` here documents what a
  *deployed* (containerized, not `ad-mri serve` on a laptop) development environment's
  settings look like, for parity with staging/production.
- **Staging / production**: the target host's `/etc/ad-mri/<environment>.env` (or
  equivalent secret-manager-injected file) is built by applying the real secret values on
  top of the non-secret template here — never by committing a filled-in copy of the
  template to this repository.
