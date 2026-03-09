# Instructor Handoff – MedConnect on Testnet

This doc summarizes VPS setup so the service **keeps running without the student**, and lists **commands** used.

## What runs on the VPS

- **medconnect-grpc** (systemd): gRPC backend, port 50052 (localhost only).
- **snetd-medconnect** (systemd): daemon, port 10000 (HTTPS) — shared with CKD when 10001 is not open.

Both are enabled on boot.

## Paths (VPS: devtraining4.deep-lab.ai, user aj)

- App: `/home/aj/medconnect_service/`
- Daemon config: `/home/aj/medconnect_service/snetd.medconnect.config.json`
- snetd unit: `/etc/systemd/system/snetd-medconnect.service`
- SSL: `/etc/letsencrypt/live/devtraining4.deep-lab.ai/`
- Endpoint: `https://devtraining4.deep-lab.ai:10000` (or `:10001` when firewall allows)

## VPS commands

**Install snetd (if not already):**
```bash
curl -LJO https://github.com/singnet/snet-daemon/releases/download/v6.2.1/snetd-linux-amd64-v6.2.1
chmod +x snetd-linux-amd64-v6.2.1
sudo mv snetd-linux-amd64-v6.2.1 /usr/local/bin/snetd
snetd --version
```

**Config:** Repo has `snetd.medconnect.config.example.json` only (no secrets). On VPS use `snetd.medconnect.config.json`; fill org/service IDs, Alchemy, ETCD, SSL paths, and `private_key_for_free_calls`. Edit on VPS: `nano ~/medconnect_service/snetd.medconnect.config.json`.

**Install snetd-medconnect service.** From local (repo root = medconnect_ai):
`scp snetd-medconnect.service aj@devtraining4.deep-lab.ai:/tmp/`

On VPS:
```bash
sudo cp /tmp/snetd-medconnect.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable snetd-medconnect
sudo systemctl start snetd-medconnect
sudo systemctl status snetd-medconnect
```

**Edit config on VPS (e.g. blockchain_enabled, free_calls_per_address):**
```bash
nano ~/medconnect_service/snetd.medconnect.config.json
sudo systemctl restart snetd-medconnect
sudo systemctl status snetd-medconnect
```

**Why User=root in snetd-medconnect.service:** SSL certs are in /etc/letsencrypt/ (root-only).

**gRPC:** `sudo systemctl status medconnect-grpc` | `start` | `journalctl -u medconnect-grpc -n 30 -f`

**Check daemon:** `sudo systemctl status snetd-medconnect`; `sudo ss -tlnp | grep 10000`; `curl -k https://localhost:10000/`

**Logs:** `sudo journalctl -u snetd-medconnect -n 50 -f`

## Free calls

- Set `"blockchain_enabled": false` in `snetd.medconnect.config.json` for free calls.
- Free-call signer address must be registered on the publisher portal for the medconnect service.
- Optional: `free_calls_per_address` in config to allowlist addresses with custom limits.

## ETCD (for paid calls later)

When `blockchain_enabled: true`, daemon needs etcd. Start with:
`docker start docker-etcd-node-1`
Optional on boot: `docker update --restart unless-stopped docker-etcd-node-1`

## Published service (testnet)

- Org ID: AJ_dev_outreach_test_1
- Service ID: medconnect
- Endpoint: https://devtraining4.deep-lab.ai:10000

## Summary

Both services are systemd, enabled on boot. snetd-medconnect runs as root for SSL. Config: edit on VPS with nano, then `sudo systemctl restart snetd-medconnect`. Do not commit real `snetd.medconnect.config.json` (it contains keys); use `snetd.medconnect.config.example.json` as template.
