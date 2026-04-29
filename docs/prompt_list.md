# ARIA — Demo Prompt List

Ordered for a natural narrative flow: *find the right threshold → understand who your customers are → see where alerts concentrate → decide where to focus.*

---

## Threshold Tuning

**1. Business FP/FN sweep**
> Show FP/FN threshold tuning for Business customers — weekly transaction count

**2. Individual FP/FN sweep**
> Show FP/FN threshold tuning for Individual customers — monthly transaction amount

**3. SAR catch rate — Individual**
> Run SAR backtest for Individual customers — monthly transaction amount

**4. SAR catch rate — Business**
> Run SAR backtest for Business customers — weekly transaction amount

**5. Workload direction**
> If I lower the threshold for Business customers, does that increase or decrease my alert workload?

---

## Segmentation

**6. Cluster all customers**
> Cluster all customers into behavioral segments and show the treemap

**7. Drill into Business**
> Cluster Business customers into 4 segments

**8. Filter to highest-risk cluster**
> Show only the highest-risk Business cluster

**9. Alert distribution**
> Show alerts and false positive distribution across segments

**10. Connect back to thresholds**
> Which segment should I prioritize for threshold tuning?

---

## Notes

- Prompts 3–4 (SAR backtest) and 8 (filter clusters) are new in V5 — verify these pass before using in demo.
- Prompts 1–2, 6–7, 9 are side panel prompts confirmed passing in V4.
- Prompt 5 tests the workload direction question (Gap 20, confirmed fixed in V4).
