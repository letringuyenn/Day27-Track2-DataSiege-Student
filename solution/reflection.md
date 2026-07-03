# Reflection

The hardest faults were the subtle AI-infrastructure and distribution-shift cases. Obvious contract, lineage, freshness, null-rate, and volume faults map cleanly to one toolkit signal, but subtle drift can sit close to the published baseline. I used the baseline thresholds as the main guardrail and tightened only the signals that stayed clean in practice: mean amount and embedding/corpus drift. This improved recall without turning the detector into an alert-everything strategy.

The main tradeoff is cost versus coverage. One metered call per event gives strong coverage and stays under budget in practice and private-sized runs, but the public stream exceeds its smaller budget by 20 credits. I tested a budget guard, but the missed true positives cost more than the overage penalty. With another pass, I would add a better risk model for deciding which late expensive AI checks to skip instead of using a simple remaining-budget cutoff.
