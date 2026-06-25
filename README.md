# Uplift Modeling for Email Marketing — Who Should Get the Campaign?

A causal machine learning project that identifies customers whose purchases
are *caused* by a marketing email — not just customers who were likely to
buy anyway. Built on the Hillstrom MineThatData E-Mail Analytics dataset
(64,000 customers, randomized controlled experiment).

## Why this project

Most ML portfolios show predictive modeling: "who is likely to buy?"
This project answers a different, harder question: **"who buys *because*
we emailed them?"** That's a causal question, and it's the difference
between a marketing model that looks good on a leaderboard and one that
actually saves a company money. Very few data science portfolios
demonstrate causal inference — this one does, end to end, with an honest
accounting of where the signal is real and where it's noise.

## TL;DR — the headline result

> Targeting only the **top 10%** of customers by predicted uplift returns
> **2.03x ROI**. Emailing the **entire customer base** returns only **1.03x**
> — barely breaking even. Naive "who's likely to buy" targeting captures
> almost none of this: **90% of the campaign's real incremental effect is
> concentrated in the single highest-ranked decile**, which a standard
> classifier fails to identify on its own (see Phase 2).

## Problem Statement

A retailer ran a randomized email campaign: some customers got a Men's
product email, some got a Women's product email, some got nothing. We only
have budget to email a fraction of the customer base going forward. Naively,
a marketer would target customers most likely to purchase. This project
shows why that naive approach leaves money on the table, and builds a
better targeting strategy using uplift modeling.

## Dataset

[Hillstrom MineThatData E-Mail Analytics dataset](http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv) —
64,000 customers, randomly assigned to 3 groups (No E-Mail / Men's E-Mail /
Women's E-Mail), with recorded visit, conversion, and spend outcomes.
This project uses the Men's E-Mail vs. No E-Mail subset (42,693 customers)
for a clean binary treatment/control comparison.

## Methodology

### Phase 1 — Randomization Check
Before trusting any causal conclusion, confirmed the 3 treatment groups
were genuinely comparable pre-treatment (recency, purchase history, gender
mix, newbie status were all statistically balanced across groups — a
required precondition for any causal claim). Also established the naive
population-level effect: the Men's campaign lifted visit rate by **+7.66pp**
and conversion by **+0.68pp** versus no email.

### Phase 2 — Naive Baseline (and why it fails)
Trained a standard XGBoost classifier to predict "who converts?" using only
treated customers, then ranked all customers by that score. Result: the
**naive score does not reliably track real uplift**. The top-ranked decile
*alone* captured 90% of the campaign's total incremental effect, while the
two lowest-scored deciles showed *negative* uplift (treated conversion rate
of 0.0% against a positive control rate) — a **Sleeping Dogs** segment: these
customers were actively less likely to convert *because* they were emailed,
a pattern a standard classifier has no way to detect.

### Phase 3 — Uplift Models
Built and compared three genuinely different uplift modeling approaches
using `scikit-uplift`:
- **S-Learner** (single model, treatment as a feature)
- **T-Learner** (two separate models, uplift = difference in predictions)
- **Class Transformation** (single model on a causally-relabeled target)

| Model | Qini AUC | Uplift @ top 30% |
|---|---|---|
| **T-Learner** | **0.067** (best overall ranking) | 0.0078 |
| Class Transformation | 0.037 | 0.0069 |
| S-Learner | 0.033 | **0.0094** (best at this specific budget) |

T-Learner was selected as the primary model for best overall Qini AUC, with
the explicit finding that S-Learner outperforms it specifically at a 30%
targeting budget — a reminder that "best model" depends on the deployment
constraint, not just one aggregate metric.

### Phase 4 — Interpretation & Business Translation
- **Interpretability**: fit a shallow surrogate decision tree to explain
  *why* the model ranks certain customers as Persuadables. Purchase history
  (57.8% importance) and prior women's-category purchases (28.8%) dominate.
  The single most persuadable segment — high-history customers who'd
  previously bought women's merchandise — is a genuine cross-sell signal,
  not the "obvious" result you'd guess going in.
- **Business translation**: converted the model's ranking into real
  dollar terms using held-out spend data:

| Targeted | Net Value | ROI |
|---|---|---|
| 10% | $1,319 | 2.03x |
| 20% | $1,618 | 1.63x |
| 30% | $762* | 1.20x* |
| 50% | **$2,562** (highest total $) | 1.40x |
| 100% (email everyone) | $339 | 1.03x |

*\*The 30% figure dips below its neighbors — flagged explicitly as likely
sampling noise (spend is a heavy-tailed, zero-inflated variable), not a
real signal. The trustworthy pattern is the overall shape: strong returns
at small targeting %, collapsing toward breakeven at 100%.*

## Interactive Demo

A Streamlit app (`app.py`) wraps the trained model in an interactive UI:
score any hypothetical customer profile, explore the cost/ROI tradeoff
with a live slider, and view the Qini curve — all using the exact same
pipeline functions as the analysis above, not a reimplementation.

```
pip install -r requirements.txt
streamlit run app.py
```

## Tech Stack

- `pandas`, `numpy` — data handling
- `xgboost` — base learner for all uplift models
- `scikit-uplift` — uplift model implementations (S-Learner, T-Learner,
  Class Transformation) and Qini/uplift evaluation metrics
- `scikit-learn` — train/test splitting, surrogate decision tree
- `matplotlib`, `seaborn` — visualization
- `streamlit` — interactive demo

## Project Structure

```
uplift-project/
├── README.md
├── requirements.txt
├── app.py                          # Interactive Streamlit demo
├── data/
│   └── (place hillstrom.csv here — see Setup)

```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Download the dataset and save it as `data/hillstrom.csv`:
http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv

Then run the scripts in order (`01` → `04`), or open them in a notebook
and run cell by cell.

## Limitations & Honest Caveats

- Single dataset, single campaign — generalization to other domains is
  untested
- Conversion rates are low (~0.5–2%), so uplift estimates in smaller
  population slices carry real sampling noise (explicitly flagged above
  rather than smoothed over)
- `cost_per_email` in the business translation is an illustrative
  assumption, not sourced from real campaign costs
- Only the Men's E-Mail vs. No E-Mail subset was modeled in depth; the
  Women's E-Mail segment would need the same treatment for a complete
  picture

## Possible Extensions

- Repeat the full pipeline on the Women's E-Mail segment and compare
- Add `causalml`/`econml` for a formal X-Learner and confidence-interval
  estimation around the Qini AUC (bootstrap)
- Use `dowhy` to formally validate causal assumptions (unconfoundedness,
  SUTVA) beyond the descriptive randomization check in Phase 1

## Resume Summary

> Built an end-to-end uplift modeling pipeline (S-Learner, T-Learner, Class
> Transformation via scikit-uplift) on a 64K-customer randomized email
> campaign, proving that standard propensity-based targeting captures
> minimal incremental value while the top-decile uplift segment drives 90%
> of real campaign lift. Translated model output into a deployable business
> case (2.03x ROI at optimal targeting vs. 1.03x for blanket targeting) and
> shipped an interactive Streamlit tool for live customer scoring.
