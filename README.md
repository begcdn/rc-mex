# RC-MEX

Relation-Card Marginalized Execution for KGQA.

This repository contains the current MVP code for RC-MEX and the earlier CIGR-D baseline utilities.

Start with `README_RC_MEX_MVP1.md`.

## Current Framing

RC-MEX is focused on relation-card grounding, not on claiming top-k graph search as novel.

- **MVP1:** Relation cards beat relation-name-only baselines and remain useful under hidden or misleading relation labels.
- **MVP2:** Question slots can retrieve/rank the correct relation card from a local candidate frontier.
- **MVP3:** Top-k relation-card execution improves recall but adds noise. This is a diagnostic showing that MVP2's top-k uncertainty contains useful recoverable graph evidence. It is not the novelty claim.
- **MVP3.5:** Card-vs-name retrieval/execution compares relation cards against raw relation names, anonymized IDs, and misleading labels under the same controlled gold-prefix slot setup.

The main RC-MEX claim is:

> Relation cards provide a more robust semantic interface for KG relations than raw relation names or arbitrary IDs, especially when schema labels are hidden, noisy, or misleading.

See `docs/rc_mex_prior_work_lessons.md` for the prior-work lesson motivating this framing.
