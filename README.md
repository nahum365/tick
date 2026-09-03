# Tick runtime

Tick is an always-on agent runtime for a Robinhood Agentic account. This
repository is the part that runs on **your** machine or your own cloud box:
the spec model and validator, the rule engine and the cage, the broker
adapters, the append-only hash-chained ledger, the scheduler and kill switch,
the box API, and the direct-connection tunnel the Tick app dials.

It is published from Tick's private monorepo by an export script; every
commit here is one export and names the source revision it came from.

Nothing in this repository holds or transmits your credentials to anyone but
your broker and your model provider: broker tokens and provider logins live
only under `TICK_HOME` on the box that runs this code.

```sh
uv sync
uv run tick --help
uv run pytest -q
```
