#!/usr/bin/env bash
# Firewall for the LLM API host. Allowlist, not LAN-wide.
#
# Order matters: SSH rules must exist BEFORE `ufw enable`, or enabling the
# firewall cuts the session you are running this from.
#
# Edit the four addresses below for your network, then run as root.
set -euo pipefail

LAN_CIDR=192.168.3.0/24        # subnet allowed to reach SSH
VPN_CIDR=10.96.71.0/24         # ZeroTier subnet allowed to reach SSH
API_CLIENTS=(192.168.3.98 10.96.71.67)   # hosts allowed to reach the API

ufw allow from "$LAN_CIDR" to any port 22 proto tcp comment 'SSH from LAN'
ufw allow from "$VPN_CIDR" to any port 22 proto tcp comment 'SSH over ZeroTier'
# ZeroTier needs inbound from arbitrary peer addresses for NAT traversal.
ufw allow 9993/udp comment 'ZeroTier'
for ip in "${API_CLIENTS[@]}"; do
  ufw allow from "$ip" to any port 8080 proto tcp comment "LLM API client"
done

ufw default deny incoming
ufw default allow outgoing
ufw logging low
ufw --force enable

# A pre-existing world-open SSH rule is common; remove it after enabling.
ufw --force delete allow 22/tcp || true
ufw status numbered verbose
