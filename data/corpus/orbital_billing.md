# Orbital Billing Platform — Operations Handbook

Orbital is a usage-based billing platform. This handbook is the corpus used by the
retrieval labs. It is deliberately specific: retrieval evaluation needs documents that
contain exact identifiers, numbers and policy statements, because those are what
distinguish a system that retrieves from a system that paraphrases plausibly.

## Plans and limits

Orbital sells three plans. The **Starter** plan costs 49 USD per month and includes
100,000 metered events. The **Growth** plan costs 249 USD per month and includes
2,000,000 metered events. The **Scale** plan is quoted individually and starts at
1,500 USD per month with a 20,000,000 event allowance.

Overage on Starter is billed at 0.0009 USD per event. Overage on Growth is billed at
0.0004 USD per event. Scale customers negotiate overage in their order form; the
default contract rate is 0.00018 USD per event.

Every plan includes a hard cap called the *circuit breaker*. When a workspace exceeds
300% of its included event allowance within a single billing period, ingestion is
paused and an `INGEST_PAUSED_CIRCUIT_BREAKER` event is emitted. Only a workspace owner
can lift the pause, and the lift is recorded in the audit log.

## Rate limits

The ingestion API accepts 1,000 events per second per workspace on Starter, 5,000 on
Growth, and 25,000 on Scale. Bursts up to twice the sustained rate are absorbed for at
most 10 seconds. Beyond that, the API returns HTTP 429 with the header
`X-Orbital-Retry-After` carrying a value in milliseconds.

The reporting API is limited to 60 requests per minute regardless of plan. Reporting
429s do not carry `X-Orbital-Retry-After`; clients should back off exponentially with
full jitter starting at 500 ms.

## Error codes

- `ORB-1001` — malformed event envelope. The event is rejected and not retried.
- `ORB-1002` — unknown metric key. The event is stored in the dead-letter queue for
  7 days and can be replayed once the metric is defined.
- `ORB-1003` — duplicate idempotency key within the 24-hour dedupe window. The event
  is acknowledged but not double-counted.
- `ORB-2001` — workspace suspended for non-payment. Ingestion returns 402.
- `ORB-2002` — circuit breaker engaged. Ingestion returns 429 until an owner lifts it.
- `ORB-3001` — invoice finalisation failed because a usage record arrived after the
  close window. The invoice is placed in `pending_review`.

## Invoicing

Billing periods close at 00:00 UTC on the first day of each month. Usage records that
arrive more than 6 hours after close are excluded from that invoice and roll into the
next period. Orbital does not issue retroactive credits for late-arriving usage; the
documented remedy is a manual credit note raised by support.

Invoices move through four states: `draft`, `pending_review`, `finalised`, `paid`. Only
a `draft` invoice can be edited. A `finalised` invoice can be voided within 14 days,
which produces a credit note and a replacement draft.

## Data retention

Raw events are retained for 90 days on Starter, 400 days on Growth, and 24 months on
Scale. Aggregated daily rollups are retained for 7 years on all plans. Deletion
requests under GDPR Article 17 are executed within 30 days and cover raw events and
rollups, but not the immutable invoice ledger, which is retained for statutory
accounting reasons.

## Webhooks

Orbital signs every webhook with an HMAC-SHA256 signature in the `X-Orbital-Signature`
header, computed over the raw request body using the workspace webhook secret. The
signature is prefixed with `t=<unix_timestamp>,v1=`. Reject any webhook whose timestamp
is more than 300 seconds old to prevent replay.

Failed webhook deliveries are retried 8 times with exponential backoff over roughly
14 hours. After the eighth failure the endpoint is marked `degraded` and delivery is
suspended until it is manually re-enabled.
