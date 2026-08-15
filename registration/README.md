# Registration

**Status: registered and active as of 2026-08-13.** `miner.yaml` follows the real schema
published at `docs.telegraphprotocol.com/miners/yaml-config.md`. `miner.public.yaml` is the
comment-free copy that actually got pinned to IPFS and registered — the YAML is public once
submitted, so the internal working notes stay out of it.

What actually happened, in order:

1. Deployed the service to a DigitalOcean droplet (Singapore) behind Caddy for automatic
   HTTPS — `base_url` in `miner.yaml` points at it, verified live (`/health` and `/predict`
   both confirmed working over the public URL).
2. Picked a random 6-digit `id` (708425) — no lookup endpoint exists to check availability in
   advance, it's register-and-see.
3. Went through **integrate.telegraphprotocol.com**'s 3-step flow: connected a wallet,
   imported `miner.public.yaml`, it sandbox-tested the live `/predict` endpoint (passed),
   pinned to IPFS via Pinata, then signed the `registerMiner` transaction on Base Sepolia.
4. Cost was gas only — no bond, no stake, no MACHINA required to register, confirmed in both
   the registration doc and `protocol/tokenomics.md` ("miners receive zero MACHINA from
   emissions"). A few cents of Base Sepolia testnet ETH from a public faucet covered it.
5. **Verified independently, not just the UI's success message** — checked the registering
   wallet's transaction nonce incremented (proof a tx was actually included), then queried
   the node's own `/miner-dispatcher/integrations` endpoint directly, which confirmed
   `activation_status: "active"` for `veritarach-ai-text-detector`. Already active, ahead of
   the "activates at next epoch boundary" the UI implied.

`AI_TEXT_DETECTION` validated and registered without issue, despite `yaml-config.md`'s
general canonical Intent list showing `AI_DETECTION` instead — that doc table just hadn't
caught up with the hackathon-specific intents; no fallback to `TEXT_AUTHENTICITY_CHECK` was
needed.

There's no update function once registered — changing anything means
`deregisterMiner(registrationId)` then `registerMiner(...)` again with a new YAML — but since
there's no bond, that's cheap to do if it's ever needed.

**2026-08-15 note:** the model backing `base_url` has been retrained and redeployed since
this registration. `miner.yaml`/`miner.public.yaml` still say 99.65% test F1 because that's
what was actually pinned to IPFS and registered on-chain on 2026-08-13 — left as-is rather
than edited, since these two files are meant to be an exact record of what got submitted, not
a running status page. The live service's real current figure is 99.86% (see the main
README's Status section), which is better on its own training distribution but has a
documented generalization gap the earlier figure didn't have visibility into either. Worth a
re-registration with updated YAML if that discrepancy matters for how the miner gets
represented on-chain — not done here since it's a real transaction, not a docs update.
