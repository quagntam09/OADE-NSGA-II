# Parameter Sensitivity Analysis

Thu muc nay tach rieng khoi benchmark chinh. No khong dung den:

- `config/config.yaml`
- `plot_all_problems.py`

## Noi dung kiem dinh

Script `run_parameter_sensitivity.py` chay rieng OADE-NSGA-II de kiem tra:

- `pop_size`: `50`, `100`, `200`
- `stagnation_patience`: `5`, `10`, `15`, `20`, `25`, `30`, `40`
- `f_cr_init`: fixed baseline, low init, high init, random uniform

Mac dinh moi cau hinh chay `30` lan voi seed `42..71`, tren cac bai toan ZDT trong `sensitivity_analysis/config.yaml`.

## Lenh chay

Chay sensitivity study:

```powershell
.\.venv\Scripts\python.exe sensitivity_analysis\run_parameter_sensitivity.py
```

Ve bieu do sau khi co ket qua:

```powershell
.\.venv\Scripts\python.exe sensitivity_analysis\plot_parameter_sensitivity.py
```

## Output

Tat ca ket qua nam trong `sensitivity_analysis/results/`:

- `sensitivity_raw_runs.csv`: ket qua tung run, gom IGD, HV, runtime, restart_count, F/CR dau-cuoi.
- `sensitivity_summary.csv`: mean/std/best theo problem va muc tham so.
- `sensitivity_statistics.csv`: Friedman test va Wilcoxon signed-rank post-hoc vs baseline, co Holm correction.
- `sensitivity_fcr_trace.csv`: trace mean_F, mean_CR, prob_de theo tung generation.
- `plots/`: cac bieu do IGD, HV, runtime/restart, va trace F/CR.

## Luu y dien giai

- Friedman test dung de phat hien anh huong tong quat cua mot tham so co nhieu muc.
- Wilcoxon signed-rank dung de so tung muc voi baseline vi cac cau hinh dung cung tap seed.
- `pop_size=100`, `stagnation_patience=20`, va `fixed_baseline` la baseline mac dinh.
- Neu muon chay nhanh de thu pipeline, sua `runs` hoac `n_gen` trong `sensitivity_analysis/config.yaml`.
