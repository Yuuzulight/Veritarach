# Brings the existing veritarach-miner DigitalOcean droplet under Terraform management via
# `terraform import`. This droplet is a LIVE, registered Telegraph Miner serving real
# traffic -- these arguments are written to match its actual current state exactly (see
# droplet-snapshot.json, gitignored, produced by `doctl compute droplet get 592141473 -o
# json`), not the state it was originally created with. No new resource is provisioned by
# this configuration; it only takes ownership of what already exists.

resource "digitalocean_droplet" "veritarach" {
  name     = "veritarach-miner"
  region   = "sgp1"
  size     = "s-2vcpu-4gb"
  image    = "ubuntu-24-04-x64"
  vpc_uuid = "c72b0a52-3ff1-4101-ab99-b2c7da690015"
  tags     = ["veritarach"]

  backups    = false
  monitoring = false # confirmed via `terraform state show` post-import -- the
                      # "droplet_agent" feature turned out NOT to mean monitoring=true

  # ssh_keys is intentionally omitted. The DigitalOcean droplet GET API does not return
  # ssh_keys after creation (SSH keys are a create-time-only concept from the API's
  # perspective -- confirmed by directly querying the raw API), and `terraform state
  # show digitalocean_droplet.veritarach` after import confirmed the attribute recorded
  # as null. Setting a value here that doesn't match state risks a ForceNew
  # destroy/recreate on a live, registered Miner -- leaving it unset matches the actual
  # imported state exactly and produces no diff. (For reference: the droplet was
  # originally created with SSH key id 58472804, "veritarach-deploy".)
}
