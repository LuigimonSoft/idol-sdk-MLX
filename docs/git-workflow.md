# Git workflow and clean checkout

## Development conventions

Use [Conventional Commits](https://www.conventionalcommits.org/) for commit
messages, for example `docs: document model access` or `test: cover tensor
conversion`. Work on a task branch such as `codex/issue-4-clean-checkout`
and open a pull request against the default branch. Use the pull request
template: what, why, how, a Mermaid diagram, how to test, changed files and
objectives achieved. Keep English first and add the Spanish section at the end.
Spanish documentation belongs in `es/`.

## Working around AppleDouble metadata

The original checkout on the external volume showed Git errors mentioning a
`._pack-*.idx` file in `.git/objects/pack`. These `._*` entries are
AppleDouble metadata; an ignore rule affects the working tree only and does not
repair metadata inside `.git`.

Use a fresh checkout on a local filesystem when the external volume introduces
these files. Choose an appropriate location outside an existing checkout:

```sh
git clone https://github.com/LuigimonSoft/idol-sdk-MLX.git
cd idol-sdk-MLX
git status --short --branch
git fsck --no-reflogs --connectivity-only
```

The existing checkout may contain uncommitted changes: inspect and preserve them
before moving work. Do not delete or modify another checkout's `.git` metadata
as part of this procedure. The fresh checkout's `.gitignore` excludes
working-tree AppleDouble files, local virtual environments and Python caches;
it does not prevent a filesystem from creating files inside `.git`.

## Validation

A separate clone on the local temporary filesystem passed `git status` and
`git fsck --no-reflogs --connectivity-only`. This checks Git object
connectivity, not all possible filesystem or storage failures. A temporary
checkout is not durable storage; clone into a permanent local directory for
ongoing development.

Spanish: [es/git-workflow.md](../es/git-workflow.md).
