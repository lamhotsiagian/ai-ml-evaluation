# Orbital Security and Access Control

## Authentication

Orbital supports three credential types. **Publishable keys** (prefix `pk_`) may be
embedded in client applications and can only write events. **Secret keys** (prefix
`sk_`) grant full API access and must never leave a server. **Session tokens** are
issued by the dashboard login flow and expire after 12 hours of inactivity.

Secret keys can be scoped to a subset of permissions at creation time. A scoped key
cannot be widened later; the documented procedure is to create a new key and revoke the
old one. Key rotation is expected every 90 days, and the dashboard shows a warning
banner for keys older than 120 days.

## Roles

Four roles exist: `owner`, `admin`, `analyst`, and `integrator`.

- `owner` — full access, including billing, key management and lifting circuit breakers.
  A workspace must have at least one owner; the last owner cannot be removed.
- `admin` — everything except billing changes and workspace deletion.
- `analyst` — read-only access to reports and invoices. Cannot see secret keys.
- `integrator` — can write events and read ingestion diagnostics. Cannot read invoices.

Role changes take effect on the next token refresh, which can lag by up to 12 hours for
active sessions. To revoke access immediately, revoke the session explicitly from the
Sessions page.

## Audit log

Every privileged action writes an audit record containing the actor, the action, the
target resource, the source IP, and a monotonic sequence number. Audit records are
immutable and retained for 24 months on all plans. The audit log is exportable as JSONL
by any `owner` or `admin`.

Audit export is rate limited to one export per workspace per hour. Exports larger than
500 MB are delivered as a signed download link valid for 1 hour rather than inline.

## Encryption

Data is encrypted at rest with AES-256-GCM using keys managed in a hardware security
module. Customer-managed keys are available on Scale only. In transit, Orbital requires
TLS 1.2 or higher; TLS 1.0 and 1.1 were disabled in March 2024.

## Incident response

Orbital commits to notifying affected workspace owners within 72 hours of confirming a
security incident that involves customer data. The status page is updated within 30
minutes of an incident being declared. Post-incident reviews are published for any
incident rated Severity 1 or Severity 2.

Severity levels: **S1** — customer data exposure or total ingestion outage. **S2** —
partial ingestion outage or invoice correctness defect. **S3** — degraded reporting.
**S4** — cosmetic or documentation defect.
