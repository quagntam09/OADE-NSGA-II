# Ablation Delta vs Full Model

Full model: `improved_nsga2_incremental_baseline`

## Overall (average across problems)

| ablation_algorithm | n_problems | hv_improvement_abs_mean | hv_improvement_pct_mean | igd_improvement_abs_mean | igd_improvement_pct_mean | delta_runtime_abs_mean | delta_runtime_pct_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| improved_nsga2_incremental_baseline | 5 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| improved_nsga2_incremental_plus_de_adaptive | 5 | 0.141070 | 14.791906 | 1.290145 | 74.710254 | 1.442913 | 110.986280 |
| improved_nsga2_incremental_plus_de_fixed | 5 | 0.095715 | 14.803210 | 1.199073 | 73.631208 | 1.251568 | 95.742423 |
| improved_nsga2_incremental_plus_obl_init | 5 | 0.023272 | 9.413962 | 0.108674 | -3.639039 | 0.639715 | 48.832848 |
| improved_nsga2_incremental_plus_periodic_obl | 5 | 0.009150 | 1.905898 | 0.807474 | 24.254095 | 1.790664 | 138.613249 |
| improved_nsga2_incremental_plus_restart | 5 | -0.004151 | -1.359388 | -0.005664 | -5.358573 | 1.636198 | 126.699676 |
