# News by Equation

Every night this repo takes yesterday's most-read English Wikipedia articles
and analyses them with a set of open-source tools, then writes the result to
[`output/nbe.json`](output/nbe.json). The animated story graph and terminal at
the top of [sapientronic.ai](https://sapientronic.ai) replay that file: real
output, rerun every day.

## What it does

| Step | Tool | What it says about today's stories |
|---|---|---|
| Check the data | [llm-feedback-control](https://pypi.org/project/llm-feedback-control/) | Each view series is checked before anything is claimed about it. Bot-inflated pages are refused with the reason; weak claims (a decay fitted too soon, a curve that doesn't fit) are refused or corrected. |
| When did it break? | [temporal-neural-networks](https://github.com/pcoz/temporal-neural-networks) | Self-tuning leaky neurons flag the day a story broke and how far it jumped, in units of its own noise. |
| What shape is it? | [timeseries-formula-finder](https://pypi.org/project/timeseries-formula-finder/) | A readable formula for each story's attention curve, e.g. `v(t) ≈ 98k·e^(−0.27t) + 88k`, its half-life, and tomorrow's forecast. |
| What happens to spikes like this? | history in [`state/`](state) | Fresh spikes are compared with thousands of past spikes: most halve in under a day, whatever their size. |
| How do stories connect? | [holant-tools](https://pypi.org/project/holant-tools/), [structural-computing](https://pypi.org/project/structural-computing/), [numeric-rule-finder](https://pypi.org/project/numeric-rule-finder/) | Story clusters, the article that bridges them, how long a thread has been running, and how attention moved within it (rose together, fell together, one replacing another). |
| How long will it last? | [petra-nn](https://pypi.org/project/petra-nn/) | An order-aware model of a story's life in the top 100, giving each story's chance that today is its last day there, shown next to a plain count-based estimate. |

## Keeping score honestly

Every forecast is committed to [`state/`](state) by the nightly job before
the data it predicts exists, and graded on the next run. Each forecast is always shown
next to the naive one ("same as today"), whichever wins. Fitted formulas only
issue a forecast when they've been tracking the last three days within 25%.
That rule was adopted on 13 October 2026 after looking at results, so the
panel reports the unguarded numbers alongside it, and grades from before that
date are marked as post-hoc ([`state/policy.json`](state/policy.json)).

## Run it yourself

```bash
pip install -r requirements.txt
python -m nbe.run --date 2026-10-12      # default: yesterday, UTC
python tests/validate_nbe.py output/nbe.json
```

The first run fetches a few thousand pages of data from the Wikimedia APIs
(politely, with caching in `cache/`); later runs only fetch what's new.

Data: [Wikimedia pageviews API](https://wikimedia.org/api/rest_v1/) and the
[MediaWiki API](https://www.mediawiki.org/wiki/API:Main_page). Wikipedia text
and data are available under CC BY-SA.

## Licence

Public domain under [CC0 1.0](LICENSE). Attribution is requested, not
required: please credit Edward Chalk (edward.chalk@sapientronic.ai). See
[NOTICE](NOTICE).
