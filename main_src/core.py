"""
Các kiểu dữ liệu nền tảng của thuật toán.
"""

from __future__ import annotations

from typing import Any, List, Optional
import numpy as np


class ProblemWrapper:
    def __init__(self, prob: Any) -> None:
        self.n_var = prob.n_var
        self.n_obj = prob.n_obj
        self.xl    = prob.xl
        self.xu    = prob.xu
        self._prob = prob

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Nhận x shape (n, n_var), trả về F shape (n, n_obj)."""
        return self._prob.evaluate(x)


class Individual:
    """Một cá thể trong quần thể — lưu toạ độ, fitness và metadata chọn lọc.

    Thuộc tính chính
    ----------------
    X             : vector quyết định, shape (n_var,)
    F             : vector mục tiêu,   shape (n_obj,)
    rank          : chỉ số Pareto front, 1 là front tốt nhất
    crowding_dist : khoảng cách chật chội, cao hơn nghĩa là ít bị chen lấn hơn
    creation_mode : toán tử đã tạo ra cá thể này — xem CreationMode
    used_F / used_CR : tham số DE đã dùng, cần để cập nhật Lehmer-mean
    """

    __slots__ = [
        "X", "F", "rank", "crowding_dist",
        "domination_count", "dominated_set",
        "creation_mode", "used_F", "used_CR",
    ]

    def __init__(self) -> None:
        self.X:                Optional[np.ndarray] = None
        self.F:                Optional[np.ndarray] = None
        self.rank:             Optional[int]        = None
        self.crowding_dist:    float                = 0.0
        self.domination_count: int                  = 0
        self.dominated_set:    List[int]            = []
        self.creation_mode:    str                  = CreationMode.INIT
        self.used_F:           float                = 0.5
        self.used_CR:          float                = 0.5


class CreationMode:
    INIT = "init"
    DE   = "de"
    SBX  = "sbx"
    OBL  = "obl"