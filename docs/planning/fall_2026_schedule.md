Sprint
Dates
Main question
Your work (~70%)
Nargiz (~30%)
Wednesday meeting outcome
Sprint 0 — Clean rebuild
Oct 1–2
Can we get back to a framework we completely understand?
Reset repo; preserve reports/architecture; rebuild minimal module structure; define interfaces and experiment conventions; start core implementation.
Read architecture/report; review planned model interface; identify one surrogate family she can own.
No meeting needed before Oct 1; this is setup.
Sprint 1 — Framework complete
Oct 5–9
Can any two models plug into the same optimization loop?
Finish SearchSpace, DataStore, BaseSurrogate, merge, worker pool, orchestrator, logging, reproducibility, minimal experiment harness. Run end-to-end smoke tests.
Implement/test assigned surrogate wrapper and independently verify one benchmark run.
Oct 7: architecture review. Demo one complete HPO round and confirm experimental design before large runs.
Fall recess
Oct 10–13
Buffer only
Fix anything discovered in Sprint 1; avoid scheduling critical experiments here.
Light cleanup/documentation if available.
—
Sprint 2 — Initial surrogate screen
Oct 14–23
Which model combinations are worth investigating?
Integrate all available surrogates; establish identical budgets/seeds/initial points; launch pairwise experiments; monitor logs and failures.
Run assigned surrogate/pair experiments and validate outputs.
Oct 14: approve experiment matrix. Oct 21: first results review; decide whether more repetitions are needed.
Sprint 3 — Complete screening + select pair(s)
Oct 26–30
Which combinations consistently perform well?
Aggregate experiments; build plots for quality vs evaluations and quality vs wall-clock; analyze model contribution and overhead; lead pair selection.
Finish missing runs; independently inspect anomalies; help produce figures.
Oct 28: formal decision gate: choose one primary pair and possibly one comparison pair.
Sprint 4 — Understand the winning pair
Nov 2–6
Why does this pair work?
Contribution analysis: source of best improvements, early-vs-late behavior, duplicate rate, fit/proposal overhead, convergence behavior. Begin ablations.
Run contribution/ablation experiments assigned to her pair/model.
Nov 4: present evidence for mechanism, not merely final score.
Sprint 5 — Ablation studies
Nov 9–13
Is the benefit really caused by using both models?
Compare winning pair against each surrogate alone; test equal candidate budgets; possibly test different batch allocations.
Run single-surrogate baselines and repeat selected seeds.
Nov 11: assess whether multi-surrogate system actually beats or complements individual models.
Sprint 6 — Scaling / systems characterization
Nov 16–20
Does the idea remain useful as the problem gets harder?
Increase evaluation budget/problem size/worker count where feasible; profile fit, proposal, evaluation, and total wall-clock costs.
Run one scaling dimension and help validate timing data.
Nov 18: decide whether the research claim is primarily optimization-quality, systems-performance, or both.
Sprint 7 — Thanksgiving / analysis buffer
Nov 23–30
What is missing from the evidence?
No major new architecture. Clean data, rerun failures, standardize figures, write experiment methodology while details are fresh.
Reproduce a small subset independently; document run setup.
Thanksgiving recess is Nov 25–29, so use this as a deliberately lighter sprint. (Smith College)
Sprint 8 — Robustness experiments
Dec 1–4
Do conclusions hold across seeds/tasks?
Run only experiments necessary to close identified evidence gaps; confidence intervals/variation across seeds; finalize tables.
Execute targeted repeats and check reproducibility.
Dec 2: professor review of near-final experimental story.
Sprint 9 — Research synthesis
Dec 7–11
What exactly did we learn?
Freeze core implementation; produce final plots/tables; draft Results, Methods, Architecture, and Discussion; identify limitations honestly.
Verify plots/results against raw outputs; contribute experiment descriptions.
Dec 9: present paper-style story from motivation → method → evidence → conclusion.
Sprint 10 — Semester checkpoint
Dec 14–15
What do we hand off to Spring?
Tag/release stable repo version; update README; write continuation plan; package datasets/results/figures; draft paper outline or partial manuscript.
Final reproducibility check and documentation review.
Final meeting:choose Spring publication plan and remaining experiments.



