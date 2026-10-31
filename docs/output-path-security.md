# Local output paths and trust boundaries

`scripts/capture_reference.py --output` and
`scripts/prepare_test_images.py --output` accept a filesystem location selected
by the local caller. This is intentional: users may place artifacts in a
directory they control outside the repository. A caller with permission to run
these commands must therefore treat the path as trusted.

## SonarCloud S8707 findings

SonarCloud reported `pythonsecurity:S8707` for both commands because the CLI
argument flows to filesystem writes. In the current repository, the scripts
are local developer utilities; no HTTP endpoint or agent wrapper is implemented
for them. The findings nevertheless identify a real risk **if** an integration
passes an untrusted request, retrieved document, model response or other
attacker-controlled value as the output path.

The existing behavior is not a security boundary against a caller who controls
the destination. In particular, exclusive file creation (`xb`) prevents
replacing an existing final file but does not confine the destination
directory. The fixture generator preflights existing names including dangling
symlinks and uses exclusive creation, but this is not a complete defense
against directory symlink changes, races or a malicious caller.

## Required boundary for future integrations

Do not forward an untrusted path to these commands or call their write functions
with one. Before connecting them to an agent or service:

1. Designate an output root through trusted operator configuration, independent
   of requests and model-generated arguments; restrict process permissions.
2. Accept only an appropriate identifier or filename from an untrusted caller,
   reject absolute paths and traversal, and enforce containment using
   platform-appropriate filesystem primitives.
3. Address symlinks and time-of-check/time-of-use races in the implementation;
   a simple string-prefix check or `resolve()` alone is not sufficient against
   concurrent filesystem changes.
4. Add integration tests for traversal, absolute paths, symlinks, races and
   overwrite behavior. Reassess the SonarCloud findings against the resulting
   actual data flow.

These scripts remain trusted local tools and have not been converted into
sandboxed services. Do not mark the findings resolved solely because
`xb` or a preflight check exists. Record any SonarCloud disposition against
the actual deployment and revisit it if the trust boundary changes.

Spanish: [es/output-path-security.md](../es/output-path-security.md).
