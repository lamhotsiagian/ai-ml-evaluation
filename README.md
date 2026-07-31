# AI & ML Evaluation Lab

The runnable companion to the ebook **Cracking AI & ML Evaluation System Design Interviews**.

`evalcore` is a production-grade evaluation library — statistics, metrics, LLM judges, RAG and
agent evaluation, safety probes, drift detection, and a CI regression gate — plus a Streamlit
app with one hands-on lab per chapter.

Repository: <https://github.com/lamhotsiagian/ai-ml-evaluation>

---

## Table of contents

- [What this is for](#what-this-is-for)
- [Install and run in 60 seconds](#install-and-run-in-60-seconds)
- [Your first five minutes](#your-first-five-minutes)
- [How the project is organised](#how-the-project-is-organised)
- [The ten labs](#the-ten-labs)
- [Using `evalcore` as a library](#using-evalcore-as-a-library)
- [Configuration](#configuration)
- [Golden datasets](#golden-datasets)
- [Testing](#testing)
- [CI integration](#ci-integration)
- [Cost and rate limits](#cost-and-rate-limits)
- [Troubleshooting](#troubleshooting)
- [Design decisions worth knowing](#design-decisions-worth-knowing)

---

## What this is for

Most evaluation code answers "what score did it get?". This answers the four questions that
actually decide whether something ships:

| Question | Where it is answered |
| --- | --- |
| Is this difference real, or is it noise? | `evalcore.stats` — intervals, paired tests, power |
| *Which part* of my system broke? | `evalcore.rag`, `evalcore.agents` — stage and trajectory attribution |
| Can I trust the thing doing the measuring? | `evalcore.judge.calibration` — judge vs human agreement |
| Should this build ship? | `evalcore.report` — a gate with three exit codes |

Everything runs on the Gemini free tier, and roughly half of it runs with no API key at all.

**You should use this if** you are preparing for AI evaluation interviews, building an
evaluation system at work and want a reference implementation, or reading the book and want
to run what it describes.

**This is not** a hosted platform or a drop-in replacement for LangSmith / Phoenix / DeepEval.
It is deliberately small enough to read in an afternoon — that is the point. Chapter 7 of the
book explains which parts you should own yourself and which you should rent.

---

## Install and run in 60 seconds

```bash
git clone https://github.com/lamhotsiagian/ai-ml-evaluation.git
cd ai-ml-evaluation

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # then paste a free Gemini key into it
streamlit run app.py
```

Get a free key at <https://aistudio.google.com/app/apikey>. It takes about a minute and needs
no billing details.

**No key yet?** Run it anyway. Pages 1, 2, 8, 9 and 10 are fully offline; the rest tell you
what is missing instead of crashing.

Verify the install without touching the network:

```bash
pytest          # 78 tests, ~20 seconds, no API calls
```

---

## Your first five minutes

Three things to try, in order. Each one takes about ninety seconds and each is designed to
change how you think about a number you have probably reported before.

### 1. Find out whether your suite can settle your argument

Open **1 — Evaluation Foundations → Power & sample size**. Set the baseline pass rate to
`0.80` and the improvement to detect to `0.05`.

> **What you'll see:** you need **906 cases per arm**. Now drag "paired correlation" to `0.8`
> — the requirement drops to **182**. Same evidence, one fifth of the cost, purely from
> running both systems on the same items.

Then read the MDE chart. At 100 cases and an 80% baseline, the smallest difference the suite
can reliably detect is **about 16 points**. If your team is arguing about a 3-point
regression on a 100-case suite, the suite cannot settle it — and no amount of staring at the
dashboard will change that.

### 2. Watch a metric lie to you

Open **2 — Classical ML Evaluation → Slices & fairness** and choose *Accuracy (averaged per
item)*.

> **What you'll see:** every slice within a couple of points of the aggregate, nothing
> flagged. Looks like a uniformly healthy model.

Now switch to *ROC-AUC (recomputed per slice)*.

> **What you'll see:** `segment:mobile-apac` collapses to **≈0.44 against an overall 0.82** —
> worse than a coin toss — and is flagged after FDR control. The model is nearly random on
> that segment and accuracy could never have told you, because at 6% prevalence accuracy is
> pinned near the base rate no matter what the model does.

Note also that `channel:mobile` and `region:apac` *individually* barely move. The failure
lives on the interaction, and each marginal is diluted by the segments around it.

### 3. See a gate refuse to answer

Open **9 — Evaluation Infrastructure → Regression gate** (run the harness on page 7 twice
first, so there is something to compare).

> **What you'll see:** the gate reports `pass`, `fail`, or `incomparable` — and names the
> specific case IDs that flipped. Change `EVAL_JUDGE_MODEL` in `.env`, restart, run again, and
> compare across the change: the gate returns **exit code 2** and refuses to produce a delta,
> because the ruler moved.

---

## How the project is organised

The mental model is a pipeline with four layers. You will spend most of your time in the
middle two.

```
     data              measurement            execution            decision
  ┌──────────┐       ┌──────────────┐       ┌───────────┐       ┌──────────┐
  │ datasets │──────▶│   metrics    │──────▶│  runner   │──────▶│  report  │
  │  splits  │       │ judge / rag  │       │   store   │       │   gate   │
  │ labdata  │       │ agents/safety│       │   cache   │       │          │
  └──────────┘       └──────────────┘       └───────────┘       └──────────┘
   what you           how you score          how it runs         what you do
   measure on          each item           at scale, safely      about it
```

<details>
<summary><strong>Full directory tree</strong> (click to expand)</summary>

```
ai-ml-evaluation/
├── app.py                       Streamlit entry point + environment check
├── pages/                       one lab page per chapter (Streamlit multipage)
├── ui/components.py             shared widgets: headers, tables, API-key gate
│
├── evalcore/
│   ├── config.py                settings singleton + run fingerprint
│   ├── llm.py                   Gemini factories, token bucket, SQLite cache
│   ├── labdata.py               seeded synthetic datasets for the offline labs
│   │
│   ├── stats.py             [1] intervals, paired tests, effect sizes, power, kappa
│   ├── splits.py            [1] stratified/grouped/temporal splits, leakage audit
│   ├── datasets.py        [1,9] content-addressed golden dataset registry
│   │
│   ├── metrics/
│   │   ├── classification.py[2] thresholds, ROC/PR, MCC, cost-optimal cuts
│   │   ├── regression.py    [2] MAE/RMSE/tails, residual bins, pinball loss
│   │   ├── ranking.py     [2,4] recall@k, MRR, MAP, NDCG, reciprocal rank fusion
│   │   ├── calibration.py   [2] ECE/MCE/Brier, temperature + isotonic scaling
│   │   └── slicing.py       [2] slice eval with FDR control, fairness, metamorphic
│   │
│   ├── judge/
│   │   ├── rubric.py        [3] anchored, weighted, versioned rubrics
│   │   ├── judge.py         [3] structured verdicts, position swap, Bradley-Terry
│   │   ├── calibration.py   [3] judge-vs-human meta-evaluation, bias probes
│   │   └── validators.py    [3] deterministic checks that must never reach a judge
│   │
│   ├── rag/
│   │   ├── index.py         [4] sentence chunking, Chroma dense, BM25, hybrid RRF
│   │   ├── pipeline.py      [4] traced retrieve → rerank → assemble → generate
│   │   └── metrics.py       [4] context precision/recall, faithfulness, citations
│   │
│   ├── agents/
│   │   ├── graph.py         [5] instrumented LangGraph ReAct agent + tools
│   │   └── trajectory.py    [5] trajectory scoring, efficiency, recovery, diffing
│   │
│   ├── safety/
│   │   ├── probes.py        [6] canary injection probes + over-refusal controls
│   │   └── pii.py           [6] PII scanning, memorisation, contamination
│   │
│   ├── harness.py         [7,8] Metric / Suite / Assertion — the framework abstraction
│   │
│   ├── production/
│   │   ├── drift.py         [8] PSI, KS, chi-square, embedding MMD, prompt hashing
│   │   ├── online.py        [8] A/B, mSPRT, canary controller, shadow comparison
│   │   └── cost.py          [8] cost model, latency percentiles, Pareto frontier
│   │
│   ├── runner/
│   │   ├── runner.py        [9] async runner: concurrency, retries, partial failure
│   │   └── store.py         [9] SQLite experiment store
│   │
│   └── report/
│       └── regression_gate.py [9,10] the build decision
│
├── data/
│   ├── corpus/                  Orbital docs (~1,400 words) used by the RAG labs
│   └── golden/                  versioned golden datasets (JSONL)
├── scripts/ci_gate.py           exit-code regression gate for CI
├── tests/                       78 offline tests
└── artifacts/                   generated: cache, experiment store, Chroma index
```

`[n]` marks the chapter each module belongs to.

</details>

---

## The ten labs

| # | Lab | Needs key | Try this | What you should notice |
| --- | --- | :---: | --- | --- |
| 1 | Evaluation Foundations | no | Set baseline 0.80, effect 0.05, correlation 0 → then 0.8 | 906 → 182 cases per arm; pairing is the cheapest win in evaluation |
| 1 | ↳ Leakage audit | no | Keep the defaults | One exact and one near-duplicate leak; drop the threshold to 0.4 and watch false positives arrive |
| 2 | Classical ML | no | Drag the threshold from 0.5 to 0.9 | Accuracy barely moves while recall collapses — the imbalance trap in one gesture |
| 2 | ↳ Calibration | no | Compare ECE before/after temperature scaling | ECE falls, ROC-AUC is bit-identical (asserted in the tests) |
| 2 | ↳ Slices | no | Switch from Accuracy to ROC-AUC | A segment at 0.44 vs 0.82 overall that accuracy slicing cannot see |
| 3 | LLM Judge | yes | Judge the default answer, then set samples to 3 | Agreement below 100% marks the items where your rubric is ambiguous |
| 3 | ↳ Pairwise | yes | Compare two similar answers | `position_bias = YES`; the result is forced to a tie rather than a coin flip |
| 4 | RAG Evaluation | yes | Query an error code like `ORB-1002` in all three modes | BM25 finds it, dense retrieval often does not — why hybrid is the default |
| 4 | ↳ Full suite | yes | Run 8–12 cases | Faithfulness is usually bimodal; the mean hides "mostly 1.0 with a few 0.2s" |
| 5 | Agent Evaluation | yes | Add `calculator` to forbidden tools and rerun | Composite drops to exactly 0.0 while outcome stays 1.0 — safety is a gate, not a weight |
| 6 | Safety Evaluation | yes | Run the probes with the boundary defence on, then off | The difference between the two attack success rates *is* the defence's measured value |
| 7 | Evaluation Harness | yes | Run a suite, then run it again | Second run is near-instant — that is the cache making per-commit evaluation affordable |
| 8 | Production Monitoring | no | Sequential tab, set true lift to exactly 0 | Count how often a naive daily p-value fires under a true null. That is the cost of peeking |
| 8 | ↳ Pareto frontier | no | Read which configurations are dominated | Everything off the frontier can be discarded without argument |
| 9 | Infrastructure | no | Change `EVAL_JUDGE_MODEL`, rerun, compare | Gate returns `incomparable` / exit 2 rather than a meaningless delta |
| 10 | Dashboard | no | Raise the quality floor slowly | The gate flips at the CI *lower bound*, not the point estimate |

---

## Using `evalcore` as a library

Everything in the UI is a thin wrapper over these calls. All snippets below run as-is.

### Size a suite before you build it

```python
from evalcore.stats import required_n_for_proportion, minimum_detectable_effect

# "We want to detect a 5-point improvement on an 80% baseline."
print(required_n_for_proportion(0.80, 0.05).as_text())
# two-proportion z-test: n>=906 per arm to detect +0.050 at alpha=0.05, power=80%

# Same question, but both systems run on the SAME items (rho = 0.8):
print(required_n_for_proportion(0.80, 0.05, paired_correlation=0.8).as_text())
# two-proportion z-test: n>=182 per arm ...

# Inverse question: what can 100 cases actually resolve?
print(round(minimum_detectable_effect(100, 0.80), 3))   # 0.158
```

### Decide whether B beats A

```python
from evalcore.stats import mcnemar_test, paired_bootstrap_test

baseline  = [1, 1, 0, 1, 0, 1, 1, 1, 0, 1] * 20   # pass/fail per item
candidate = [1, 1, 1, 1, 0, 1, 1, 1, 1, 1] * 20   # same items, same order

print(mcnemar_test(baseline, candidate).as_text())
# McNemar (exact binomial): delta=+0.2000, p=1.819e-12, odds ratio (fixed/broken)=+inf, n=200

result = paired_bootstrap_test(baseline, candidate)
print(result.delta_interval.as_text())
# 0.2000 [0.1450, 0.2600] (95% CI, n=200)   <- the improvements consistent with the data
```

Use McNemar for pass/fail, the paired bootstrap for continuous scores. Both require the two
score vectors to be **item-aligned** — same cases, same order.

### Audit a split for leakage

```python
from evalcore.splits import detect_leakage, grouped_split

report = detect_leakage(
    train_texts=["The Growth plan costs 249 USD per month and includes 2,000,000 events"],
    test_texts=["the growth plan costs 249 usd per month and includes 2,000,000 events"],
    near_duplicate_threshold=0.6,
    shingle_k=4,             # lower k for short texts; 5 is right for prose
)
print(report.summary())
# LEAKAGE: 1 exact, 0 near-duplicate, ... (100.0% of test contaminated).
print(report.clean)          # False -> fail the build

# Keep every row of a conversation on one side of the split
split = grouped_split([f"conv-{i // 12}" for i in range(600)])
print(split.sizes)           # {'train': 468, 'validation': 60, 'test': 72}
```

> `shingle_k` is the near-duplicate comparison window in words. The default of 5 is tuned for
> prose; short titles or one-line queries need 3–4, or every text collapses to a single
> shingle and nothing is ever near-matched.

### Score a classifier honestly

```python
from evalcore.labdata import make_fraud_like_dataset
from evalcore.metrics.classification import evaluate_binary, threshold_for_min_cost

data = make_fraud_like_dataset(n_samples=4000, positive_rate=0.06)

report = evaluate_binary(data.y_true, data.y_score, threshold=0.5)
print(report.as_row())
# accuracy 0.9407 looks great; recall 0.1684 and MCC 0.3505 tell the truth

# Choose the threshold from a business input, not from the sigmoid
threshold, cost_per_item = threshold_for_min_cost(
    data.y_true, data.y_score, cost_fp=1.0, cost_fn=20.0
)
```

### Find the slice that is failing

```python
from sklearn.metrics import roc_auc_score
from evalcore.metrics.slicing import evaluate_slices_by_metric

report = evaluate_slices_by_metric(
    data.y_true, data.y_score, data.slice_tags,
    metric=roc_auc_score, min_slice_size=30,
)
print(report.summary())
for row in report.flagged:
    print(row.as_row())
```

> Use `evaluate_slices` for metrics you can average per item (accuracy, pass rate) and
> `evaluate_slices_by_metric` for metrics defined over a set (AUC, F1, precision). Averaging
> an AUC is meaningless, and averaging accuracy on an imbalanced task hides everything.

### Run a calibrated LLM judge

```python
from evalcore.judge import BUILTIN_RUBRICS, RubricJudge, calibrate_judge

judge = RubricJudge(BUILTIN_RUBRICS["Grounded Answer Quality"], n_samples=3)
result = judge.judge(
    task="What is the overage rate on the Starter plan?",
    response="Starter overage is 0.0009 USD per event, and unused events roll over.",
    contexts=["Overage on Starter is billed at 0.0009 USD per event."],
)
print(result.verdict.overall_score, result.sample_agreement)
print(result.verdict.failure_modes)     # names the unsupported claim

# Before trusting it, prove it agrees with humans:
calibration = calibrate_judge(human_scores=[...], judge_scores=[...])
print(calibration.verdict_text())
assert calibration.deployable          # kappa >= 0.6 AND false-pass <= 0.10
```

Available rubrics: `"Grounded Answer Quality"`, `"Instruction Following"`, `"Response Safety"`.

### Evaluate a RAG pipeline stage by stage

```python
from evalcore.rag import ClaimVerifier, RagIndex, RagPipeline, evaluate_citations

index = RagIndex("my_corpus")
index.build(index.load_corpus("data/corpus"))

pipeline = RagPipeline(index, k=5, mode="hybrid")
response = pipeline.answer("What is the overage rate on the Starter plan?")

for stage in response.trace:                 # retrieve → rerank → assemble → generate
    print(stage.stage, round(stage.latency_ms), stage.payload)

print(evaluate_citations(response.answer, len(response.contexts)).as_row())

faith = ClaimVerifier().faithfulness(response.answer, response.contexts)
print(faith.faithfulness, faith.unsupported_claims)   # the list is the bug report
```

### Score an agent's path, not just its answer

```python
from evalcore.agents import EvaluableAgent, build_evaluation_tools, evaluate_trajectory

agent = EvaluableAgent(build_evaluation_tools(None), max_iterations=6)
run = agent.run("A Growth customer used 2,400,000 events. What is the total bill?")

report = evaluate_trajectory(
    run,
    expected_tools=["calculator"],
    forbidden_tools=["delete_database"],
    outcome_success=1.0,
)
print(report.as_row())
print(report.composite())      # 0.0 if any forbidden tool was called, whatever the outcome
```

### Measure whether a safety defence does anything

```python
from evalcore.safety import build_probe_suite, guarded_system_prompt, score_probe_results

probes = build_probe_suite()          # 12 probes: 9 attacks + 3 benign controls
responses = [...]                     # your system's reply to each probe.payload

report = score_probe_results(probes, responses)
print(report.summary())
print(report.passes_gate)             # zero criticals AND ASR<=5% AND over-refusal<=10%
```

Run the suite twice — once with `guarded_system_prompt(...)` and once without. The difference
in attack success rate is the defence's *measured* value. If it is zero, you have a comfort
blanket, not a control.

### Turn all of it into a build decision

```python
from evalcore.harness import Assertion, EvaluationSuite, Metric, contains_metric
from evalcore.datasets import DatasetRegistry
from evalcore.report import MetricGate, run_regression_gate
from evalcore.runner import ExperimentStore

dataset = DatasetRegistry().load("rag_qa")

async def target(case):                      # your system under test
    return await my_system.answer(case.input)

suite = EvaluationSuite(
    "nightly", dataset, target,
    metrics=[contains_metric(), Metric("non_empty", lambda c, o: float(bool(o.strip())),
                                       binary=True, threshold=1.0)],
    assertions=[Assertion("contains_expected", minimum=0.70, require_ci_clear=True)],
)
report = suite.run()
print(report.summary(), report.passed)

store = ExperimentStore()
run_id = store.save_run(report.run, label="pr-482")

gate = run_regression_gate(
    {"contains_expected": store.case_scores(baseline_id, "contains_expected")},
    {"contains_expected": store.case_scores(run_id, "contains_expected")},
    [MetricGate("contains_expected", floor=0.70, max_regression=0.02, binary=True)],
)
print(gate.markdown())        # paste-ready PR comment
raise SystemExit(gate.exit_code)
```

---

## Configuration

All settings live in `.env` and are read through one object (`evalcore.config.get_settings()`).
Nothing anywhere reads `os.environ` directly, so a run is reproducible from a single snapshot.

| Variable | Default | Notes |
| --- | --- | --- |
| `GOOGLE_API_KEY` | — | Free key from [AI Studio](https://aistudio.google.com/app/apikey) |
| `EVAL_GENERATION_MODEL` | `gemini-2.0-flash` | The system under test |
| `EVAL_JUDGE_MODEL` | `gemini-2.0-flash-lite` | Keep it different from the generator — see self-preference bias |
| `EVAL_EMBEDDING_MODEL` | `models/text-embedding-004` | Vector store and embedding-drift labs |
| `EVAL_TEMPERATURE` | `0.0` | Deterministic decoding for evaluation |
| `EVAL_JUDGE_TEMPERATURE` | `0.0` | Raised automatically when self-consistency is on |
| `EVAL_SEED` | `1337` | Every synthetic dataset and bootstrap is seeded |
| `EVAL_MAX_CONCURRENCY` | `4` | Parallel in-flight requests |
| `EVAL_REQUESTS_PER_MINUTE` | `14` | Token bucket; free tier is quota limited |
| `EVAL_CACHE_PATH` | `artifacts/llm_cache.sqlite` | Shared across UI, pytest and CI |
| `EVAL_STORE_PATH` | `artifacts/experiments.sqlite` | Run history |
| `EVAL_CHROMA_DIR` | `artifacts/chroma` | Persisted vector index |

**The run fingerprint.** `settings.fingerprint()` hashes only the values that can change a
score — models, temperatures, seed — and deliberately excludes paths and concurrency, which
change *where* and *how fast* a run happens but never *what* it scores. Two runs with the same
fingerprint are comparable; two runs with different fingerprints are not, and the regression
gate refuses to compare them.

---

## Golden datasets

Datasets are JSONL, one `EvalCase` per line, versioned in git and content-hashed.

| Dataset | Cases | Composition |
| --- | --- | --- |
| `rag_qa` | 26 | 20 answerable + **6 unanswerable** (23%), including near-misses |
| `agent_tasks` | 12 | Declares the expected tool sequence per case |
| `instruction_following` | 12 | Format, length, negation and multilingual constraints |

### Case schema

```jsonc
{
  "case_id": "rag-004",                    // stable join key across every result table
  "input": "Which header carries the retry delay on ingestion 429s?",
  "expected_output": "X-Orbital-Retry-After, in milliseconds.",
  "expected_label": null,                  // for classification tasks
  "contexts": [],                          // pre-supplied context, if any
  "expected_tools": [],                    // for agent tasks
  "slice_tags": ["api", "rate-limits"],    // drives per-slice reporting
  "difficulty": "hard",                    // easy | medium | hard
  "group_id": null,                        // keeps related rows on one side of a split
  "is_answerable": true,                   // false -> the system must abstain
  "metadata": {}
}
```

### Adding your own

```python
from evalcore.datasets import DatasetRegistry, EvalCase, GoldenDataset

cases = [EvalCase(case_id="mine-001", input="…", expected_output="…",
                  slice_tags=["billing"], difficulty="medium")]
registry = DatasetRegistry()
descriptor = registry.save(GoldenDataset("my_suite", cases))
print(descriptor.content_hash)
```

Three rules that matter more than the schema:

1. **Reserve 15–25% for unanswerable cases**, including near-misses ("what is the overage rate
   for the *Enterprise* plan?" when only Starter/Growth/Scale exist). Without them, abstention
   is entirely unmeasured, and abstention is where production hallucinations happen.
2. **Every production incident becomes a case.** The suite becomes a record of everything that
   has ever broken, and the same failure cannot ship twice.
3. **Somebody other than the model's author reviews dataset changes.** Otherwise the
   measurement drifts with the thing being measured — the technical machinery (hashing,
   diffing, comparability refusal) only makes that reviewable, it does not prevent it.

---

## Testing

```bash
pytest                              # all 78, ~20s, no network
pytest tests/test_stats.py -v       # the statistics, against published values
pytest -k slice                     # one topic
```

| File | Tests | Covers |
| --- | ---: | --- |
| `test_stats.py` | 16 | Interval coverage, paired-test power, kappa, BH, MDE ↔ power inverse |
| `test_metrics.py` | 20 | Hand-computed metric values, calibration invariants, slice discovery |
| `test_pipeline_components.py` | 42 | Datasets, splits, validators, rubrics, RAG, agents, drift, gate |

The statistics are asserted against **published values**, not against their own output — the
Wilson interval for 10/20 is checked against `[0.2993, 0.7007]`, bootstrap coverage is verified
empirically over 60 replications, and temperature scaling is asserted to leave ROC-AUC
bit-identical. If a number in the book disagrees with what the code prints, the code is right.

---

## CI integration

```yaml
- name: Run the evaluation suite
  run: python scripts/run_suite.py --suite nightly     # your entry point
  env:
    GOOGLE_API_KEY: ${{ secrets.GOOGLE_API_KEY }}

- name: Evaluation gate
  run: |
    python scripts/ci_gate.py \
      --suite nightly --metric faithfulness \
      --budget 0.02 --floor 0.85 \
      --baseline-label main --comment-path gate.md
```

| Exit code | Meaning | What CI should do |
| :---: | --- | --- |
| `0` | No blocking regression | Ship |
| `1` | Significant regression beyond budget | Block; `gate.md` names the cases that flipped |
| `2` | Incomparable, or too few shared cases | Investigate the *evaluation*, not the change |

Code 2 is deliberately distinct from code 1. A blocked build is a signal about your change; an
incomparable one is a signal about your evaluation setup. Conflating them trains everyone to
ignore both.

### Suggested trigger topology

| Trigger | Suite | Budget | Gate |
| --- | --- | --- | --- |
| Pull request | Smoke, 50–100 cases | < 5 min | Block |
| Merge to main | Full quality suite | < 30 min | Block |
| Nightly | Full + safety + adversarial | hours | Alert |
| Pre-release | Everything + human review sample | days | Block |

---

## Cost and rate limits

The Gemini free tier is quota limited, so the runner enforces a token bucket
(`EVAL_REQUESTS_PER_MINUTE`, default 14) and every judge call is cached in
`artifacts/llm_cache.sqlite`. Re-running an unchanged suite is nearly free.

Estimate before you run:

```python
from evalcore.production import evaluation_run_cost

print(evaluation_run_cost(500, judge_model="gemini-2.0-flash-lite",
                          n_judge_samples=1, cache_hit_rate=0.6))
```

**The cache key** is `(model, temperature, rendered prompt, rubric version)`. Editing a rubric
therefore invalidates exactly the affected entries. It does *not* detect a provider changing
the model behind a stable name — for that, run a frozen canary suite on a schedule and watch
for a step change (Chapter 8). Clear the cache from the app's overview page when you suspect
one.

The four levers that dominate evaluation cost, in order of return: **caching**, **tiering by
trigger**, **deterministic pre-filtering** (move every rule-expressible check out of the judge
prompt), and **dropping routine self-consistency sampling**.

---

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `MissingAPIKeyError` | No `GOOGLE_API_KEY` in `.env`, or the app was started before you saved it. Restart Streamlit — settings are cached for the process lifetime. |
| Many `429` / "resource exhausted" rows | Free-tier quota. Lower `EVAL_REQUESTS_PER_MINUTE` and `EVAL_MAX_CONCURRENCY`. Errored rows are *recorded*, not dropped, so your score is still honest. |
| Suite is slow on the second run | Cache miss. Check `artifacts/llm_cache.sqlite` exists and is writable; the app's overview page shows the row count. |
| RAG page rebuilds the index every time | `@st.cache_resource` is keyed on chunk size and overlap — changing either is a genuine rebuild. |
| Gate says `incomparable` | Working as intended: the dataset hash or settings fingerprint moved. The message names which. |
| Gate says `insufficient_data` | Fewer than `--min-cases` shared cases. The gate abstains rather than guessing on a handful of rows. |
| Slice table shows nothing flagged | On an imbalanced task, check you are not slicing accuracy. Switch to `evaluate_slices_by_metric` with AUC. |
| Import errors on Python 3.9 | Requires 3.10+ (PEP 604 unions, `match`-free but modern typing). |

---

## Design decisions worth knowing

These are the choices most likely to surprise you, and why they are what they are.

**Errors are scored as failures, not dropped.** A row the system could not answer is a row it
got wrong from the user's point of view. Dropping errors produces the classic
survivorship-biased dashboard reporting 0.94 for a service that fails on 20% of requests.
`RunResult.scores()` takes `include_errors=True` by default.

**Retries are classified, not blanket.** A 429 or a timeout is retried with backoff; a schema
validation error is not, because it will fail identically five times and burn quota doing it.

**The judge's reasoning is generated before its score.** `JudgeVerdict` declares `criteria`
before `overall_score`, and structured-output decoding follows field order. This forces the
reasoning to precede the number rather than rationalise it afterwards, which measurably
reduces score inflation. It is a one-line change with a real effect.

**Pairwise comparison always runs both orders.** Judges prefer whichever candidate is shown
first. Running one order gives a number that is partly a measurement of the judge's positional
preference. When the two orders disagree the result is recorded as a tie and flagged.

**A correct abstention scores 1.0 on faithfulness.** "I cannot answer from the provided
context" has nothing to fabricate. Scoring it 0.0 would punish exactly the behaviour you are
trying to teach.

**Unsafe agent actions zero the composite rather than reducing it.** Averaging a forbidden
write into a weighted score implies enough efficiency can compensate for it. It cannot.

**Dataset hashes are order-independent.** Re-exporting from a database with a different
`ORDER BY` must not look like a new dataset version and invalidate a year of history.

**PSI bins are cut once on the reference window and frozen.** Re-cutting bins on each new
window makes PSI structurally incapable of detecting the shift it exists to detect — the bins
move with the data, so the proportions never change. This is the most common drift-monitoring
bug in production.

**Safety probes use benign canary payloads.** A probe instructs the model to emit a specific
harmless token; compliance is detected by exact string match. No judge, no threshold, no
ambiguity — and the suite is safe to commit publicly and run in shared CI.

---

## Chapter map

| Chapter | Lab page | Primary modules |
| --- | --- | --- |
| 1 Evaluation Foundations | `1_Evaluation_Foundations` | `stats`, `splits`, `datasets` |
| 2 Classical ML Evaluation | `2_Classical_ML_Evaluation` | `metrics/*`, `labdata` |
| 3 LLM Evaluation | `3_LLM_Judge` | `judge/*` |
| 4 RAG Evaluation | `4_RAG_Evaluation` | `rag/*` |
| 5 AI Agent Evaluation | `5_Agent_Evaluation` | `agents/*` |
| 6 Safety & Alignment | `6_Safety_Evaluation` | `safety/*` |
| 7 Evaluation Frameworks | `7_Evaluation_Harness` | `harness`, `report` |
| 8 Production Evaluation | `8_Production_Monitoring` | `production/*` |
| 9 Evaluation Infrastructure | `9_Evaluation_Infrastructure` | `runner/*`, `datasets`, `llm` |
| 10 Production Projects | `10_Evaluation_Dashboard` | everything |

---

## Stack

LangChain and LangGraph for orchestration · Google Gemini (free tier) for generation, judging
and embeddings · Chroma for the vector store · Streamlit for the UI · SQLite for the response
cache and experiment store · NumPy, SciPy and scikit-learn for the statistics.

Python 3.10+.

## Licence and attribution

Companion code for *Cracking AI & ML Evaluation System Design Interviews*, published by
[AI Engineering Insider](http://aiengineeringinsider.com).
Product names are the property of their respective owners.
# ai-ml-evaluation
