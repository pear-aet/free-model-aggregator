# free-model-aggregator-01

One server that serves an OpenAI- and Anthropic-compatible API over ten free
models, built for the make.fun bounty "The best free AI model aggregator".

## Run

```sh
node main.ts
```

`node main.ts --list` prints the catalog document and exits 0.

The trial sets `MAKEFUN_ENV_URL` and `MAKEFUN_PROMPT_FILE`; the server
probes the providers, registers its port and catalog, serves the workload,
and exits 0 when the env says the workload is done.

## How it routes

- Quality first: every free model meeting a request's requirements is ranked
  by the quality table's `rating_milli`, best first.
- Spread: each earlier call to a provider charges 12 milli-points against
  that provider's candidates, so no single provider's per-trial budget is
  drained first.
- Failover: on a 402/429/5xx the request is retried on the next free model
  that meets it; a provider that fails twice in a row is cooled down for
  three requests.
- All three upstream wires are spoken: OpenAI, Anthropic and Gemini, in
  both directions, preserving system text, last user text, image, tool
  names, max_tokens, stop sequences and JSON mode. A streamed OpenAI
  answer always carries the final usage chunk, with the provider's counts
  recomputed from the request and the answer text when the provider's
  stream does not report them.

Local practice scores (kit's sealed scorer, process driver): 99 / 99 / 99
on practice trials 0, 1 and 2 (quality 99, served 100, wire 100,
failover 100, catalog 100). The kit starter scores 38 on trial 0.

## License

MIT
