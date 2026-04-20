# OADE-NSGA-II Benchmark Suite

So sánh `improved_nsga2` với các thuật toán đa mục tiêu từ `pymoo` trên bộ benchmark ZDT và xuất các kết quả phục vụ báo cáo: biểu đồ cột `HV/IGD` (best, mean, ablation) và bảng thống kê tương ứng.

## Cấu trúc

- `algorithm_src/`: code thuật toán chính
- `comparison/`: code benchmark và vẽ biểu đồ
- `outputs/`: toàn bộ CSV và ảnh kết quả

## Cài đặt

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Chạy benchmark

### Lệnh chạy đầy đủ (khuyến nghị)

```powershell
.\.venv\Scripts\python.exe comparison\benchmark.py --problems zdt1,zdt2,zdt3,zdt4,zdt6 --algorithms improved_nsga2,pymoo_nsga2,pymoo_rnsga2,pymoo_dnsga2,pymoo_mopso --seeds 0,1,2,3,4,5,6,7,8,9 --pop-size 100 --n-gen 200 --csv outputs/benchmark_results.csv --summary-csv outputs/benchmark_summary.csv
```

Liệt kê tên thuật toán hợp lệ:

```powershell
.\.venv\Scripts\python.exe comparison\benchmark.py --list-algorithms
```

Lưu ý:

- Một số thuật toán có thể không tồn tại trong phiên bản `pymoo` hiện tại.
- Script sẽ tự `skip` thuật toán không khả dụng và tiếp tục chạy phần còn lại.

### Chạy ablation (bật/tắt từng cơ chế cải tiến)

Để so sánh công bằng, ablation nên giữ cùng `problems`, `seeds`, `pop-size`, `n-gen` như benchmark chuẩn.

Liệt kê các biến thể ablation của thuật toán cải tiến:

```powershell
.\.venv\Scripts\python.exe comparison\benchmark.py --list-ablations
```

Liệt kê các biến thể incremental (từ baseline rồi bật từng cơ chế):

```powershell
.\.venv\Scripts\python.exe comparison\benchmark.py --list-incrementals
```

Ý nghĩa từng biến thể:

- `improved_nsga2`: bản đầy đủ (full mechanism)
- `improved_nsga2_ablation_sbx_only`: tắt DE và adaptive DE (chỉ còn SBX)
- `improved_nsga2_ablation_no_adaptive_de`: giữ DE nhưng tắt thích nghi F/CR
- `improved_nsga2_ablation_no_obl_init`: tắt khởi tạo OBL
- `improved_nsga2_ablation_no_periodic_obl`: tắt phun OBL định kỳ
- `improved_nsga2_ablation_no_restart`: tắt cơ chế restart khi trì trệ

Ví dụ chạy incremental để trả lời câu hỏi "NSGA2 gốc + từng cơ chế cải thiện bao nhiêu":

```powershell
.\.venv\Scripts\python.exe comparison\benchmark.py --problems zdt1,zdt2,zdt3,zdt4,zdt6 --algorithms improved_nsga2_incremental_baseline,improved_nsga2_incremental_plus_obl_init,improved_nsga2_incremental_plus_de_fixed,improved_nsga2_incremental_plus_de_adaptive,improved_nsga2_incremental_plus_periodic_obl,improved_nsga2_incremental_plus_restart,pymoo_nsga2 --seeds 0,1,2,3,4,5,6,7,8,9 --pop-size 100 --n-gen 200 --csv outputs/benchmark_results_incremental.csv --summary-csv outputs/benchmark_summary_incremental.csv
```

Tạo bảng delta tự động so với full model (để copy vào báo cáo):

```powershell
.\.venv\Scripts\python.exe comparison\ablation_delta_report.py outputs/benchmark_summary_ablation.csv --output-dir outputs --full-model improved_nsga2 --output-prefix ablation
```

File sinh ra:

- `outputs/ablation_delta_vs_full.csv`: delta chi tiết theo từng benchmark
- `outputs/ablation_delta_vs_full_overall.csv`: delta trung bình theo từng biến thể cơ chế
- `outputs/ablation_delta_vs_full.md`: bảng Markdown dán thẳng vào báo cáo

Tạo bảng cải thiện so với baseline incremental (nên dùng để phân tích đóng góp từng cơ chế):

```powershell
.\.venv\Scripts\python.exe comparison\ablation_delta_report.py outputs/benchmark_summary_incremental.csv --output-dir outputs --full-model improved_nsga2_incremental_baseline --ablation-prefix improved_nsga2_incremental_ --output-prefix incremental_internal_baseline
```

Nếu muốn so trực tiếp với `pymoo_nsga2` gốc:

```powershell
.\.venv\Scripts\python.exe comparison\ablation_delta_report.py outputs/benchmark_summary_incremental.csv --output-dir outputs --full-model pymoo_nsga2 --ablation-prefix improved_nsga2_incremental_ --output-prefix incremental_vs_pymoo_nsga2
```

Tạo bảng incremental chain (đúng thứ tự cộng dồn cơ chế):

```powershell
.\.venv\Scripts\python.exe comparison\incremental_chain_report.py outputs/benchmark_summary_incremental.csv --output-dir outputs --output-prefix incremental_chain
```

Mặc định chain sẽ là:

- `improved_nsga2_incremental_baseline`
- `improved_nsga2_incremental_plus_obl_init`
- `improved_nsga2_incremental_plus_de_fixed`
- `improved_nsga2_incremental_plus_de_adaptive`
- `improved_nsga2_incremental_plus_periodic_obl`
- `improved_nsga2_incremental_plus_restart`

File sinh ra:

- `outputs/incremental_chain_detail.csv`: chi tiết theo từng benchmark và từng bước chain
- `outputs/incremental_chain_overall.csv`: trung bình đóng góp theo từng bước chain
- `outputs/incremental_chain.md`: bảng Markdown dán thẳng vào luận văn/bài báo

## Quy trình

### 1) Chạy benchmark chuẩn (so sánh thuật toán)

```powershell
.\.venv\Scripts\python.exe comparison\benchmark.py --problems zdt1,zdt2,zdt3,zdt4,zdt6 --algorithms improved_nsga2,pymoo_nsga2,pymoo_rnsga2,pymoo_dnsga2,pymoo_mopso --seeds 0,1,2,3,4,5,6,7,8,9 --pop-size 100 --n-gen 200 --csv outputs/benchmark_results.csv --summary-csv outputs/benchmark_summary.csv
```

### 2) Chạy benchmark ablation

```powershell
.\.venv\Scripts\python.exe comparison\benchmark.py --problems zdt1,zdt2,zdt3,zdt4,zdt6 --algorithms improved_nsga2,improved_nsga2_ablation_no_adaptive_de,improved_nsga2_ablation_no_obl_init,improved_nsga2_ablation_no_periodic_obl,improved_nsga2_ablation_no_restart,pymoo_nsga2 --seeds 0,1,2,3,4,5,6,7,8,9 --pop-size 100 --n-gen 200 --csv outputs/benchmark_results_ablation.csv --summary-csv outputs/benchmark_summary_ablation.csv
```

### 3) Vẽ đúng 6 biểu đồ cột + xuất 3 bảng (CSV/Markdown)

Lưu ý mặc định hiển thị nhãn cột:

- Tất cả biểu đồ cột từ `comparison/visualize_benchmark.py` và `comparison/plot_hv_igd_from_table.py` đều hiển thị nhãn đã scale theo `×10^-1`.
- Trên cột chỉ hiện số rút gọn; chú thích scale được ghi một lần trên hình.

```powershell
.\.venv\Scripts\python.exe comparison\visualize_benchmark.py outputs/benchmark_results.csv --summary-csv outputs/benchmark_summary.csv --ablation-summary-csv outputs/benchmark_summary_ablation.csv --igd-best-output outputs/benchmark_igd_best_bar.png --hv-best-output outputs/benchmark_hv_best_bar.png --igd-mean-output outputs/benchmark_igd_mean_bar.png --hv-mean-output outputs/benchmark_hv_mean_bar.png --igd-ablation-output outputs/ablation_igd_bar.png --hv-ablation-output outputs/ablation_hv_bar.png --best-table-output outputs/table_best_hv_igd.csv --mean-table-output outputs/table_mean_hv_igd.csv --ablation-table-output outputs/table_ablation_hv_igd.csv --best-table-md-output outputs/table_best_hv_igd.md --mean-table-md-output outputs/table_mean_hv_igd.md --ablation-table-md-output outputs/table_ablation_hv_igd.md --ui-scale 1.2
```

Kết quả đúng theo yêu cầu ban đầu:

- Biểu đồ: `benchmark_igd_best_bar.png`, `benchmark_hv_best_bar.png`, `benchmark_igd_mean_bar.png`, `benchmark_hv_mean_bar.png`, `ablation_igd_bar.png`, `ablation_hv_bar.png`
- Bảng CSV: `table_best_hv_igd.csv`, `table_mean_hv_igd.csv`, `table_ablation_hv_igd.csv`
- Bảng Markdown: `table_best_hv_igd.md`, `table_mean_hv_igd.md`, `table_ablation_hv_igd.md`

### 4) (Tùy chọn) Vẽ 2 biểu đồ HV/IGD trực tiếp từ bảng ablation CSV

```powershell
.\.venv\Scripts\python.exe comparison\plot_hv_igd_from_table.py outputs/table_ablation_hv_igd.csv --hv-output outputs/ablation_hv_from_table.png --igd-output outputs/ablation_igd_from_table.png --ui-scale 1.2
```

Vẽ 2 biểu đồ HV/IGD từ bảng incremental so với `pymoo_nsga2`:

```powershell
.\.venv\Scripts\python.exe comparison\plot_hv_igd_from_table.py outputs/table_incremental_vs_nsga2_hv_igd.csv --hv-output outputs/incremental_vs_nsga2_hv_from_table.png --igd-output outputs/incremental_vs_nsga2_igd_from_table.png --ui-scale 1.2
```

### 5) (Tùy chọn) So sánh từng cơ chế incremental với NSGA2 gốc

```powershell
.\.venv\Scripts\python.exe comparison\visualize_benchmark.py outputs/benchmark_results_incremental.csv --summary-csv outputs/benchmark_summary_incremental.csv --ablation-summary-csv outputs/benchmark_summary_incremental.csv --full-algorithm pymoo_nsga2 --ablation-prefix improved_nsga2_incremental_ --igd-ablation-output outputs/incremental_vs_nsga2_igd_bar.png --hv-ablation-output outputs/incremental_vs_nsga2_hv_bar.png --ablation-table-output outputs/table_incremental_vs_nsga2_hv_igd.csv --ablation-table-md-output outputs/table_incremental_vs_nsga2_hv_igd.md --ui-scale 1.2
```

## Cách chỉnh tham số

### Chỉnh số biến theo benchmark

Sửa dictionary `DEFAULT_PROBLEM_N_VARS` trong [comparison/benchmark.py](comparison/benchmark.py), ví dụ:

```python
DEFAULT_PROBLEM_N_VARS = {
    "zdt1": 30,
    "zdt2": 30,
    "zdt3": 30,
    "zdt4": 10,
    "zdt6": 10,
}
```

### Chỉnh số vòng chạy

Thay `DEFAULT_SEEDS`, `DEFAULT_POP_SIZE`, `DEFAULT_N_GEN` trong [comparison/benchmark.py](comparison/benchmark.py), hoặc truyền bằng CLI. `DEFAULT_SEEDS` chính là danh sách seed được lặp để tính trung bình và vẽ so sánh.

### Chọn thuật toán tham gia benchmark

Sửa `DEFAULT_ALGORITHMS` trong [comparison/benchmark.py](comparison/benchmark.py), hoặc truyền qua `--algorithms`.

### Chỉnh tên file đầu ra

Thay `DEFAULT_CSV_PATH` và `DEFAULT_SUMMARY_CSV_PATH` trong [comparison/benchmark.py](comparison/benchmark.py).

## Ghi chú

`comparison/visualize_benchmark.py` hiện tập trung vào biểu đồ cột HV/IGD và các bảng thống kê tương ứng.