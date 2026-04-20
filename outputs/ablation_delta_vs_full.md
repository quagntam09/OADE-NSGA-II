# Ablation Delta vs Full Model

Full model: `improved_nsga2`

## Overall (average across problems)

| ablation_algorithm | n_problems | hv_improvement_abs_mean | hv_improvement_pct_mean | igd_improvement_abs_mean | igd_improvement_pct_mean | delta_runtime_abs_mean | delta_runtime_pct_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| improved_nsga2_ablation_no_adaptive_de | 5 | 0.001346 | 0.237250 | 0.000544 | 8.047640 | -0.017065 | -1.216478 |
| improved_nsga2_ablation_no_obl_init | 5 | -0.000861 | -0.058494 | -0.001809 | -38.911354 | 0.050738 | 3.384080 |
| improved_nsga2_ablation_no_restart | 5 | 0.001214 | 0.217667 | 0.000739 | 11.998183 | -0.032764 | -2.163213 |
