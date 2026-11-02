"""Stage 07: Matrix operations.

A `Mat` (2-D matrix of `Value` scalars) on top of the stage_05 scalar engine;
gradients flow through `Value`. For C = A @ B: dL/dA = G @ B.T, dL/dB = A.T @ G.
"""

from __future__ import annotations

import os as _os
import sys as _sys
from typing import Iterable, List, Tuple, Union

_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

from dlfs import stage_import  # noqa: E402

# Reuse the scalar engine (stage_05) and vector container (stage_06) unchanged.
Stage5_Value = stage_import("stage_05", "Value")
Stage6_Vec = stage_import("stage_06", "Vec")

# Re-export under canonical public names for later stages / tests.
Value = Stage5_Value
Vec = Stage6_Vec

Scalar = Union[int, float, Stage5_Value]


class Mat:
    """A 2-D matrix of `Value` scalars (row-major `List[List[Value]]`). Ops compose
    scalar `Value` arithmetic; no `backward()` (reduce to a scalar first). Each op
    returns a NEW `Mat`."""

    def __init__(self, data: Iterable[Iterable[Scalar]]) -> None:
        """Build self.data: List[List[Value]] from a 2-D iterable; set rows/cols
        (wrap non-`Value` entries, keep shared nodes; ValueError on ragged rows)."""
        self.data: List[List[Stage5_Value]] = []
        self.rows: int = 0
        self.cols: int = 0

        self.rows = len(data)
        self.cols = len(data[0])
        for i in range(self.rows):
            temp_list = []

            assert len(data[i]) == self.cols
            
            for j in range(self.cols):
                num = data[i][j]
                if not isinstance(num, Value):
                    temp_list.append(Value(num))
                else:
                    temp_list.append(num)
            self.data.append(temp_list)

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.rows, self.cols)

    def __getitem__(self, i: int) -> List[Stage5_Value]:
        """Return row `i` as a list of `Value`."""
        return self.data[i]

    def __iter__(self):
        """Iterate over rows."""
        return iter(self.data)

    def __repr__(self) -> str:
        return f"Mat(shape={self.shape})"

    def matmul(self, other: "Mat") -> "Mat":
        """Matrix product C = self @ other; (m,k)@(k,n) -> (m,n), where
        C[i][j] = sum_p self[i][p] * other[p][j]. ValueError on inner-dim mismatch."""
        
        assert self.cols == other.rows
        
        res = [[0 for c in range(other.cols)] for r in range(self.rows)]

        for i in range(self.rows):
            for j in range(other.cols):
                temp = 0
                for p in range(self.cols):
                    temp += self.data[i][p] * other.data[p][j]
                res[i][j] = temp
        
        return Mat(res)

    def __matmul__(self, other: "Mat") -> "Mat":
        """The `@` operator; delegates to `matmul`."""
        return self.matmul(other)

    def transpose(self) -> "Mat":
        """Transpose: C[j][i] = self[i][j], shape (cols, rows); reuse the same
        `Value` objects so gradient flows back to the originals."""

        res = [[0 for c in range(self.rows)] for r in range(self.cols)]

        for i in range(self.rows):
            for j in range(self.cols):
                res[j][i] = self[i][j]
        return Mat(res)

    @property
    def T(self) -> "Mat":
        """Property alias for `transpose()`."""
        return self.transpose()

    def reshape(self, rows: int, cols: int) -> "Mat":
        """Reshape to (rows, cols) row-major, reusing the `Value`s; ValueError if
        the element count changes."""

        res = [[0 for c in range(cols)] for r in range(rows)]
        flatten = []
        
        for i in range(self.rows):
            for j in range(self.cols):
                flatten.append(self.data[i][j])

        if rows * cols != len(flatten):
            raise ValueError(f"Cannot reshape matrix of size {len(flatten)} into ({rows}, {cols})")

        for i in range(rows):
            for j in range(cols):
                res[i][j] = flatten.pop(0)
        return Mat(res)

    def sum(self) -> Stage5_Value:
        """Sum of all elements as one `Value`: s = sum_{i,j} self[i][j]."""
        res = 0
        for i in range(self.rows):
            for j in range(self.cols):
                res += self.data[i][j]
        return res

    def mean(self) -> Stage5_Value:
        """Mean of all elements as one `Value`: sum() / N, N = rows*cols."""
        N = self.rows * self.cols
        return self.sum() / N
