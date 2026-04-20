# Incremental Chain Report

## Chain

improved_nsga2_incremental_baseline -> improved_nsga2_incremental_plus_obl_init -> improved_nsga2_incremental_plus_de_fixed -> improved_nsga2_incremental_plus_de_adaptive -> improved_nsga2_incremental_plus_periodic_obl -> improved_nsga2_incremental_plus_restart

## Overall (mean across problems)

| step_index | algorithm | n_problems | step_hv_improvement_abs_mean | step_hv_improvement_pct_mean | step_igd_improvement_abs_mean | step_igd_improvement_pct_mean | cum_hv_improvement_abs_mean | cum_hv_improvement_pct_mean | cum_igd_improvement_abs_mean | cum_igd_improvement_pct_mean | cum_delta_runtime_abs_mean | cum_delta_runtime_pct_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | improved_nsga2_incremental_baseline | 5 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 1 | improved_nsga2_incremental_plus_obl_init | 5 | 0.023272 | 9.413962 | 0.108674 | -3.639039 | 0.023272 | 9.413962 | 0.108674 | -3.639039 | 0.639715 | 48.832848 |
| 2 | improved_nsga2_incremental_plus_de_fixed | 5 | 0.072444 | 4.361867 | 1.090399 | 72.994541 | 0.095715 | 14.803210 | 1.199073 | 73.631208 | 1.251568 | 95.742423 |
| 3 | improved_nsga2_incremental_plus_de_adaptive | 5 | 0.045354 | 18.422535 | 0.091072 | 9.628486 | 0.141070 | 14.791906 | 1.290145 | 74.710254 | 1.442913 | 110.986280 |
| 4 | improved_nsga2_incremental_plus_periodic_obl | 5 | -0.131919 | -26.555984 | -0.482671 | -944.640256 | 0.009150 | 1.905898 | 0.807474 | 24.254095 | 1.790664 | 138.613249 |
| 5 | improved_nsga2_incremental_plus_restart | 5 | -0.013301 | -22.483596 | -0.813138 | -67.731560 | -0.004151 | -1.359388 | -0.005664 | -5.358573 | 1.636198 | 126.699676 |
