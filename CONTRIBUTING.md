# Contributing to LibRPA

See [Development Tips](docs/develop/develop_tips.md) for the pull request
workflow, the risk tiers, the evidence standards, and the coding rules.

See [REVIEW.md](REVIEW.md) for the review criteria.

## Local agent instructions

In LibRPA 0.9.0 and earlier, `.gitignore` ignored `CLAUDE.md` and `AGENTS.md`.
Git now tracks them. If you have local copies of these files, rename them before
you pull or rebase:

```bash
mv CLAUDE.md CLAUDE.local.md
mv AGENTS.md AGENTS.local.md
```

Otherwise git stops the pull, or replaces your copy with no message. Git ignores
`CLAUDE.local.md` and `AGENTS.local.md`, and Claude Code also reads
`CLAUDE.local.md`.
