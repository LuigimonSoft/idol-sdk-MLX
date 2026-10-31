# Pull request checks and SonarQube Cloud

The GitHub Actions workflow runs unit tests on pull requests, pushes to
`master` and manual runs. It checks out the public upstream reference source at
commit `4702a7f00850e34b6e83a2170cc123c3e9570744`, installs the locked
reference dependencies with hash verification, and sets
`IDOL_REFERENCE_SOURCE` so the reference tests execute rather than being
skipped.

The runner is GitHub-hosted macOS ARM64. Tests use CPU code and do not require
Hugging Face credentials or model weights. They do not validate Metal, gated
model downloads or full inference.

## SonarQube Cloud automatic analysis

SonarQube Cloud analysis is **separate from the GitHub Actions workflow**.
With a GitHub-bound project and automatic analysis enabled, SonarQube Cloud can
analyze changes to PRs and the default branch using its GitHub integration.

1. Import or bind `LuigimonSoft/idol-sdk-MLX` to the global SonarQube Cloud
   project with organization `luigimonsoft` and project key
   `LuigimonSoft_idol-sdk-MLX`.
2. In the SonarQube Cloud project's **Administration > Analysis Method**, enable
   automatic analysis if the project supports it.
3. Open or update a PR and verify the analysis appears in SonarQube Cloud and
   GitHub. Branch protection and required checks are configured separately in
   GitHub; this change does not enable them.

Automatic analysis does not import the unit-test coverage report. The workflow
does not invoke SonarScanner, wait for a Sonar quality gate, or pass a
SonarCloud token to PR code. Do not interpret a passing unit-test job as proof
that SonarCloud analysis succeeded. Verify both checks independently.

References: [SonarQube Cloud GitHub setup](https://docs.sonarsource.com/sonarqube-cloud/getting-started/github),
[automatic analysis](https://docs.sonarsource.com/sonarqube-cloud/discovering-sonarcloud/what-sonarcloud-can-do)
and [coverage limitations](https://docs.sonarsource.com/sonarqube-cloud/analyzing-source-code/test-coverage/overview).

## Local tests

With the reference environment and pinned upstream checkout described in
[environments.md](environments.md), run:

```sh
IDOL_REFERENCE_SOURCE=../idol-sdk-reference .venv-reference/bin/python -m unittest discover -s tests -v
```

Never commit credentials, private images or model weights. The workflow does
not need a `SONAR_TOKEN` secret for automatic analysis.

Spanish: [es/ci-sonarcloud.md](../es/ci-sonarcloud.md).
