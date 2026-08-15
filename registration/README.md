# Registration

**Status: registered and active as of 2026-08-15 (registrationId 85).** `miner.yaml` follows the real schema
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

**2026-08-15: re-registered after the retrain.** The model backing `base_url` was retrained
and redeployed (see the main README's Status section — real F1 went from an unverified 99.65%
claim to an independently-confirmed 99.86%, with a documented generalization gap the earlier
figure had no visibility into either way). Re-registering to get that onto the on-chain record
turned into more than a docs update:

1. **Deregistered the old entry first.** `updateMiner` exists in the protocol (atomically
   deregisters and re-registers in one transaction) but isn't exposed anywhere in
   integrate.telegraphprotocol.com's UI, and the registry contract isn't verified on BaseScan,
   so there's no point-and-click way to call it either. Went with the two-step path instead:
   found the original registration's on-chain `registrationId` (77 — not the same as this
   file's `id: 708425`, which the contract never even sees) by querying `MinerRegistered`
   event logs directly via `eth_getLogs`, filtered by our registering address, since neither
   BaseScan nor the node API surface that number anywhere on their own. Deregistered it with
   `cast send ... deregisterMiner(uint256) 77`.
2. **The website's own registration flow turned out to require an extra "wallet linked to
   account" step** (on top of having both a wallet connected and an email/password or
   magic-link account signed in) that isn't documented anywhere and doesn't work — the
   "Profile" page it points you to 404s, and the in-app "Profile" button doesn't navigate
   anywhere either. Disconnecting and reconnecting the wallet while already logged in didn't
   help — the site's "Disconnect" button only clears its own local state, not the wallet's
   actual site permission, so MetaMask just silently reauthorized on the next render.
3. **Registered directly instead, bypassing the broken step.** Used the website as far as it
   would go (Import & Upload still worked fine, and produced the same IPFS pin as before:
   `QmStgms5GJXxTpd1gqsgQAKYwm4KZWfBe5fwYcpECLjMBB`, same content, same hash), then called
   `registerMiner` on the Diamond contract directly with `cast send`, using that hash and IPFS
   gateway URL. New `registrationId`: **85**.
4. **Verified three independent ways** before calling it done: a direct `eth_call` to
   `getMiner(85)` confirming `active == true`, the same call for the old registration (77)
   confirming `active == false`, and the node's own `/api/miners` catalog showing the new
   description live. Not just the transaction succeeding — actual on-chain and node-side state
   checked directly, same standard as the original registration.

The API-key-requirement toggle on the Import & Upload step has its own quirk worth noting for
next time: leaving "Requires API key" off (correct for this service, since `/predict` has no
auth) made the site's own `/api/validate` call fail with an opaque 400 before it even reached
Veritarach's endpoint. Toggling the requirement on and entering a placeholder value routed
around it cleanly — clearly a bug in how the site handles the keyless case, not anything on
Veritarach's side.
