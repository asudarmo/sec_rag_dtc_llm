# Cloud deployment (bonus)

Optional stretch goal, per the rubric's cloud-deployment bonus (2 pts): lift-and-shift
the exact `docker-compose.yml` stack onto a free-tier VM, rather than splitting
services across managed offerings — no code changes, just VM/firewall setup.

**Status**: full stack deployed and reachable — app at `http://35.196.253.183:8501`,
Grafana at `:3000`, both fully working (including the dashboard's Postgres connection
— see the bug fix below). Stress-tested under the heaviest realistic load (see below);
it holds up on memory but is slow. **Decision**: staying on the free-tier `e2-micro`
rather than spending bonus credits on a larger instance — this is an optional bonus
item, and the goal was demonstrating a successful deployment, not optimizing its
performance, so the latency finding is documented as a known trade-off rather than
something to fix.

## Provider & sizing decision

**GCP**, using the existing account/billing setup. Chose **`e2-micro`** — the only
instance type covered by GCP's "Always Free" tier — over Oracle's more generous
(24GB RAM) ARM tier, trading memory headroom for x86 compatibility (no need to
cross-build the Docker image for arm64) and using an already-verified account.

**A real constraint worth knowing**: profiling the stack locally showed a single
retrieve+rerank call alone can peak around ~400MB (sentence-transformers + torch CPU
inference), and `e2-micro` only has **~969MB total RAM** (confirmed via `free -h` on
the actual VM: ~618MB available at idle, before anything is deployed). This is tight
— comfortably running app + Postgres + Grafana simultaneously, especially under
"Compare both" mode (two concurrent pipelines), may need a swap file as a safety net.
This will be measured for real once the stack is deployed, not just estimated.

**Free-tier region constraint**: GCP's Always Free `e2-micro` only applies in
`us-west1`, `us-central1`, or `us-east1` — any other region bills against the
account's billing/credits instead. Used **`us-east1`** (lower latency from the UK
than `us-central1`, no other technical reason to prefer one over another).

## What was set up

| Resource | Value |
|---|---|
| Project | `dtc-llm-2026` |
| Billing | already linked to the project before this deployment started |
| VM name | `sec-rag-app` |
| Zone | `us-east1-b` |
| Machine type | `e2-micro` (Always Free eligible) |
| OS | Debian 12 (bookworm) |
| Boot disk | 30GB standard persistent disk (within free-tier allowance) |
| External IP | `35.196.253.183` |
| Firewall | `allow-sec-rag-ports` — TCP 8501 (app) and 3000 (Grafana) open to `0.0.0.0/0`; port 5432 (Postgres) deliberately **not** exposed, stays internal to the VM's Docker network |

## How it was provisioned

Two equivalent paths exist — the Console (manual, no local install) and `gcloud` CLI
(used here, since it was already installed and authenticated). The CLI commands, for
reference/reproducibility:

```bash
# One-time setup (already done for this project)
gcloud init                                    # auth + default project
gcloud billing projects link PROJECT_ID --billing-account=BILLING_ACCOUNT_ID
gcloud services enable compute.googleapis.com

# VM
gcloud compute instances create sec-rag-app \
  --zone=us-east1-b \
  --machine-type=e2-micro \
  --image-family=debian-12 \
  --image-project=debian-cloud \
  --boot-disk-size=30GB \
  --boot-disk-type=pd-standard \
  --tags=sec-rag-app

# Firewall — app + Grafana only, not Postgres
gcloud compute firewall-rules create allow-sec-rag-ports \
  --allow=tcp:8501,tcp:3000 \
  --target-tags=sec-rag-app \
  --source-ranges=0.0.0.0/0

# SSH — gcloud generates and registers an SSH keypair automatically on first use,
# no manual key setup needed
gcloud compute ssh sec-rag-app --zone=us-east1-b
```

The Console (manual) equivalent: create/select a project → **Billing** → link a
billing account → **APIs & Services → Library** → enable "Compute Engine API" →
**Compute Engine → VM instances → Create Instance** (e2-micro, a free-tier-eligible
region, Debian 12, up to 30GB disk) → **VPC network → Firewall → Create Firewall
Rule** (TCP 8501, 3000) → use the **SSH** button next to the instance in the console
(browser-based terminal, keys handled automatically).

## Deployment steps (done)

1. Added a 2GB swap file (`/swapfile`, persisted via `/etc/fstab`) as a safety net
   against the ~969MB RAM ceiling, before installing anything else.
2. Installed Docker Engine + Compose plugin via Docker's official apt repo for Debian.
3. Got the repo onto the VM via `git archive HEAD | ...` piped through `scp` (no
   GitHub remote configured yet, so this reproduces exactly what a real `git clone`
   would contain, without needing one).
4. Set up `.env` on the VM with the real `GEMINI_API_KEY` and a freshly-generated
   random `POSTGRES_PASSWORD` — deliberately **not** the repo's default
   `postgres`/`postgres`, since Postgres's port is reachable on the VM's own network
   interface (just not from the internet, per the firewall rule above).
5. `docker compose up -d --build` — a genuine from-scratch build (no layer cache),
   took several minutes on `e2-micro`'s limited CPU. Confirmed `torch==2.14.0+cpu`
   installed (not the CUDA build) — the NVIDIA/triton package downloads visible in the
   build log are just uv's dependency-resolution graph computation, not things that
   actually get installed.
6. Confirmed the app and Grafana are reachable at `http://35.196.253.183:8501` and
   `:3000` from outside the VM (HTTP 200 both).

## Memory & latency stress test (done)

Ran the single heaviest realistic scenario: **"Compare both" mode**, traditional side
using `method=hybrid` + reranking + query rewriting (the most expensive combination —
both dense and sparse retrieval, a cross-encoder pass, and an extra Gemini call to
split the query), running **concurrently** with the agentic side, for "Compare the
main AI-related risk factors between tech and banking companies." Memory was sampled
every 2 seconds throughout via `free -m` and `docker stats`.

**Result: it completed successfully, no crash, no OOM kill** (`dmesg` confirmed
nothing), and all 3 containers were still healthy afterward. But it was tight and
slow:

| Metric | Value |
|---|---|
| Peak system memory used | 804MB / 969MB (~83%) — only ~165MB "available" at the tightest sample |
| Peak `app` container | 433MB |
| Peak `postgres` container | 44MB |
| Peak `grafana` container | 104MB |
| Traditional side latency | 176.9s |
| Agentic side latency | 214.0s |
| Total wall time (concurrent) | 214.4s |

**The real finding here is latency, not memory.** The same question profile takes
roughly **5s (traditional) / 17s (agentic)** locally on a normal development machine
(see [agentic-rag.md](agentic-rag.md)) — on `e2-micro`'s minimal shared/burstable
vCPU allocation, the identical work takes **35-45x longer**. Memory held up (barely)
with the swap file in place; CPU is the actual bottleneck for this instance size, and
a live reviewer waiting 3-4 minutes for the heaviest-case answer is a poor demo
experience even though nothing technically breaks. See [PLAN.md](../PLAN.md) for the
open decision on how to address this (e.g. defaulting the cloud instance to a lighter
config, or accepting the latency as a known trade-off of the free tier).

## Bug found via this deployment: Grafana password authentication failure

Visiting the live Grafana dashboard showed "password authentication failed for user
postgres" on every panel — a real, reusable bug, not specific to this VM (full
technical detail in [CHANGELOG.md](../CHANGELOG.md)). Short version: the datasource
provisioning file hardcoded the default `postgres`/`postgres` credentials, which
happened to match every local `.env` used so far, but not the freshly-generated
random password deliberately used for this deployment. Fixed by driving the
datasource config from the actual `.env` values (Grafana's `$__env{VARNAME}`
provisioning syntax) instead of hardcoding them, and adding the missing `env_file`
wiring the `grafana` service needed to see those variables at all. Verified fixed
both locally and on the VM via Grafana's `/api/ds/query` endpoint returning real data.

## Teardown (if abandoning this or done demoing)

```bash
gcloud compute instances delete sec-rag-app --zone=us-east1-b
gcloud compute firewall-rules delete allow-sec-rag-ports
```

Deleting the instance stops all billing for it — the boot disk is deleted with it by
default (not left behind as an orphaned charge).
