# Registration

`miner.yaml` in this directory is a stub. Three things are blocking it from being real, all
waiting on responses in the Telegraph Protocol Discord:

**1. Which Intent this hackathon actually accepts.** The general Telegraph docs list
`AI_DETECTION` as a canonical Intent, but this hackathon uses a separate, curated Intent list
that hasn't been published yet. It's possible `AI_DETECTION` isn't on it. If that happens,
the classifier itself doesn't need to change — it's just a one-line swap in `miner.yaml` to
`TEXT_AUTHENTICITY_CHECK` or `CONTENT_VERIFICATION`, the two closest fits on the general
list, and re-registering. We're deliberately not asking about this directly in Discord and
just waiting for the announcement, since it's expected to land on its own.

**2. Node address and internal secret.** YAML validation (`/miner-dispatcher/validate`)
needs both, and neither has been requested yet. This needs asking for early, since it depends
on someone responding — better to find out it's missing now than at registration time.

**3. Where to get testnet MACHINA.** The 100 MACHINA registration bond is testnet-only (no
real money involved), but unlike testnet ETH, there's no known faucet for it yet. Also
unasked.

A couple of smaller open questions that don't block registration outright but affect how
Veritarach is built around it:

- Whether `miner.yaml` needs an explicit Intent version suffix (`AI_DETECTION@v1.0`) or
  defaults to latest — unconfirmed, needs checking before Phase 4.
- Which response field carries the per-signal Explorer hash back to the caller — a partial
  answer exists in Discord but not a definitive one. Worth rechecking once the Explorer
  feature ships, since it affects how Veritarach should log its own requests/responses.

Once the Discord answers land, update `miner.yaml` directly and remove the corresponding
comment — no other file should need to change.
