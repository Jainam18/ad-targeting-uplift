Problem Statement
Business context

An online advertising platform runs paid ad campaigns for its advertisers. Showing ads has a cost, and not every user needs one to convert. Some users would have bought anyway, and some never will. The team ran a large randomized experiment, in which users were randomly assigned to see the ad (treatment) or not (control), and now needs to decide how to use the result.

Decision to make

Should the platform (a) show the ad to all users, (b) stop showing it, or (c) show it only to a targeted segment of users likely to respond?

Metrics
Type	Metric	Definition
Primary	Conversion rate	Share of users who convert, treatment vs. control
Secondary	Visit rate	Share of users who visit the advertiser’s site
Guardrail	Cost per incremental conversion	Ad cost divided by conversions caused by the ad, not total conversions

The dataset has no cost data, so the project states an explicit assumption for cost per ad shown and value per conversion, and tests how sensitive the recommendation is to those assumptions.

Data

The Criteo Uplift Modeling dataset has about 14 million users, 12 anonymized features, a binary treatment flag, and visit and conversion outcomes. Treatment was randomized, so differences between groups can be read as causal effects. Treatment and control groups are intentionally unequal in size (roughly 85/15), and conversions are rare, so statistical power is a real concern. Verify both facts in the data and report them.

Questions
Validity: Was the experiment run correctly? (Sample ratio check against the intended split, covariate balance across groups.)
Average effect: What is the overall lift in conversion and visit rate, with 95% confidence intervals?
Power: What minimum detectable effect could this test reliably detect, and how large would a follow-up test need to be?
Precision: How much does variance reduction (CUPED) tighten the intervals?
Heterogeneity: Do some segments respond far more than others, and can an uplift model identify them?
Targeting value: Does treating only the top-ranked segment deliver more incremental conversions per dollar than treating everyone? (Compared using a Qini curve.)
Approach

Metrics and segment cuts in SQL; validity checks, inference, and CUPED in Python; uplift modeling (T-learner, then a causal forest if time allows); sensitivity analysis on the cost and value assumptions.

Deliverables
Reproducible notebook and clean repository
Charts: lift with confidence intervals, Qini curve, segment breakdown
One-page decision memo with a recommendation, expected impact with intervals, risks, and a proposed follow-up experiment
Success criteria

A decision-maker can read the memo alone and know what to do, and any reviewer can rerun the notebook and get the same numbers.

Before you start:

Treat the 85/15 split and the cost and value assumptions as things to confirm, not assume. Write what you actually find.
Decide on your primary metric and test before looking at results. Writing this document first is what makes it a legitimate analysis rather than a hunt for significant results, and interviewers will ask about it.