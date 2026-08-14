variable "do_token" {
  description = "DigitalOcean API token. Sourced from the TF_VAR_do_token environment variable at plan/apply time -- never hardcode this."
  type        = string
  sensitive   = true
}
