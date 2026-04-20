# Ablation Delta vs Full Model

Full model: `pymoo_nsga2`

## Overall (average across problems)

| ablation_algorithm | n_problems | hv_improvement_abs_mean | hv_improvement_pct_mean | igd_improvement_abs_mean | igd_improvement_pct_mean | delta_runtime_abs_mean | delta_runtime_pct_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| improved_nsga2_incremental_baseline | 5 | -0.212999 | -26.696464 | -1.340654 | -17127.701401 | -0.006378 | 32.877540 |
| improved_nsga2_incremental_plus_de_adaptive | 5 | -0.071930 | -7.716709 | -0.050509 | -650.830595 | 1.436535 | 126.078319 |
| improved_nsga2_incremental_plus_de_fixed | 5 | -0.117284 | -12.972986 | -0.141581 | -1795.104559 | 1.245190 | 91.138593 |
| improved_nsga2_incremental_plus_obl_init | 5 | -0.189728 | -21.951627 | -1.231979 | -15639.494006 | 0.633337 | 51.495227 |
| improved_nsga2_incremental_plus_periodic_obl | 5 | -0.203849 | -25.430597 | -0.533180 | -6936.054667 | 1.784286 | 166.725040 |
| improved_nsga2_incremental_plus_restart | 5 | -0.217150 | -27.429967 | -1.346318 | -17231.549601 | 1.629820 | 159.403615 |
