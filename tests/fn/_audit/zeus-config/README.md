# zeus council configuration (snapshot 2026-07-26)

Copies of the live config on zeus, kept here so the audit's double-check
apparatus is reproducible and reviewable from the repo.

## The roster

    members       minimax-m3:cloud, minimax-m2.5:cloud,
                  gemma4:31b-cloud, gpt-oss:120b-cloud
    orchestrator  minimax-m3:cloud     (reconciles, surfaces disagreements)
    final         Opus 5               (advisory only -- the book outranks both)

Two orchestrators by design, so no single model both generates an answer and
blesses it.

## Dead models removed

`qwen3-coder:480b-cloud` returns **HTTP 410 Gone** from the cloud
passthrough. `deepseek-v3.1:671b-cloud` is not among the 47 installed models.
Both were in MoA's file defaults and `qwen3-coder` headed HELM's
`_DEFAULT_PREF`, so **HELM was selecting a dead model by default on every
fresh session**.

Removed rather than demoted: a retired endpoint is not a fallback. A model
that 410s inside a council silently reduces a four-member panel to three
without saying so, which is the failure mode the council exists to prevent.

## Durability

MoA had **no systemd unit** -- it had been running detached for two days, and
its file defaults still named the dead models, so any restart would have
resurrected the bug. It now has `~/.config/systemd/user/moa.service` with the
roster pinned in `Environment=`, `Restart=on-failure`, and `loginctl
enable-linger perseus` so it survives logout and reboot. HELM already had a
unit.

## Gotchas worth keeping

- MoA binds `MOA_HOST=127.0.0.1`, NOT the tailnet IP. Probing
  `100.97.162.67:8081` hangs and looks like a broken service. It is not.
- Read the *process* environment (`/proc/<pid>/environ`), not the file
  defaults, before concluding a service is misconfigured. I got this wrong
  once and reported MoA broken when it was healthy.
- `tmux` is NOT installed on zeus. `setsid ... &` detaches fine.
  To add it: `sudo apt install -y tmux`.
