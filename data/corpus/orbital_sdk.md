# Orbital SDK Integration Guide

## Installing

The Python SDK is published as `orbital-sdk` and requires Python 3.10 or newer. The
Node SDK is `@orbital/sdk` and requires Node 18 or newer. Both SDKs read the secret key
from the `ORBITAL_SECRET_KEY` environment variable if no key is passed explicitly.

## Sending events

An event envelope has five required fields: `metric_key`, `workspace_id`, `timestamp`
in RFC 3339 format, `quantity` as a non-negative number, and `idempotency_key`. An
optional `properties` object may carry up to 32 string keys with values no longer than
512 characters each. Exceeding either limit rejects the event with `ORB-1001`.

The SDK batches events in memory and flushes every 2 seconds or every 500 events,
whichever comes first. Call `client.flush()` before process exit; the SDK registers an
`atexit` hook, but that hook does not run on `SIGKILL`.

## Idempotency

Idempotency keys are deduplicated for 24 hours. Reusing a key inside that window
returns HTTP 200 with `"deduplicated": true` and does not increment usage. Reusing a key
after 24 hours creates a second billable event, which is the single most common source
of double-billing incidents reported to support.

The recommended key construction is `sha256(workspace_id + metric_key + source_event_id)`
truncated to 32 hex characters.

## Backfills

Historical events can be submitted with a `timestamp` up to 35 days in the past using
the `/v1/events:backfill` endpoint. Backfill requests are limited to 50,000 events per
call and 10 calls per hour. Events dated before the current billing period's close are
excluded from the closed invoice and produce `ORB-3001` in the ingestion diagnostics.

## Testing

Every workspace has a paired sandbox workspace with the suffix `_sandbox`. Sandbox
workspaces accept the same API surface, never generate invoices, and reset all data
every 30 days. Sandbox secret keys use the prefix `sk_test_`.

The SDK exposes `client.simulate(event)` which validates an envelope locally against
the schema without sending it. Use this in unit tests; it performs no network I/O and
raises `EnvelopeError` with a field-level message on failure.

## Migration from v1 to v2

The v2 API renamed `event_name` to `metric_key` and made `idempotency_key` mandatory.
The v1 endpoints remain available until 30 June 2026 and return the deprecation header
`X-Orbital-Deprecation: sunset=2026-06-30`. The SDK's `LegacyAdapter` translates v1
envelopes to v2 and fills a deterministic idempotency key derived from the v1 payload.
