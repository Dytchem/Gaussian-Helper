import re
import math
import random
from copy import deepcopy
from typing import List, Tuple, Optional


# =========================
# 基础向量运算（纯 Python）
# =========================


def _dot(a: List[float], b: List[float]) -> float:
    """点乘 a·b"""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: List[float], b: List[float]) -> List[float]:
    """叉乘 a×b（右手法则）"""
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _norm(v: List[float]) -> float:
    """向量长度 ||v||"""
    return math.sqrt(_dot(v, v))


def _normalize(v: List[float], eps: float = 1e-12) -> List[float]:
    """单位化 v/||v||，长度过小则报错"""
    n = _norm(v)
    if n < eps:
        raise ValueError("规范化失败：向量长度过小（接近零）。")
    return [v[0] / n, v[1] / n, v[2] / n]


def _sub(a: List[float], b: List[float]) -> List[float]:
    """向量相减 a - b"""
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def _add(a: List[float], b: List[float]) -> List[float]:
    """向量相加 a + b"""
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def _random_noncollinear(seed: Optional[int] = None) -> List[float]:
    """
    生成一个在 [-1,1]^3 的随机非零向量（用于构造与主轴不共线的辅助方向）。
    """
    rng = random.Random(seed)
    for _ in range(100):
        v = [rng.uniform(-1.0, 1.0) for _ in range(3)]
        if _norm(v) > 1e-6:
            return v
    return [0.3, 0.5, -0.7]  # 兜底


def _random_noncollinear(seed: Optional[int] = None) -> List[float]:
    """
    生成一个在 [-1,1]^3 区间的随机非零向量（用于构造与主轴不共线的辅助方向）。
    """
    rng = random.Random(seed)
    for _ in range(100):
        v = [rng.uniform(-1.0, 1.0) for _ in range(3)]
        if _norm(v) > 1e-6:
            return v
    return [0.3, 0.5, -0.7]  # 兜底


# =========================
# Molecule 类
# =========================


class Molecule:
    """
    管理分子坐标的类。支持：
    - 从文本格式构造（固定原子数、符号与输出精度）。
    - 输出为同样风格的字符串（保留原始前导空白、按列对齐）。
    - 坐标系重建（以 i->j 为新的 x/y/z 轴；可约束某原子落在指定平面并为指定轴的正方向，确保 i,j,k 组成的平面平行于 axis 和 constraint_axis 张成的平面）。
    - 平移、旋转等几何变换。
    - 代数运算（+、-、*、/）支持逐元素运算，仅在有原子符号的位置符号匹配时允许 Molecule 间的运算（空符号忽略）。

    所有变换方法都返回新的 Molecule 对象（原对象不变，便于链式操作与回溯）。
    """

    _line_regex = re.compile(
        r"^(?P<lead>\s*)(?P<sym>[A-Za-z]*)\s*"
        r"(?P<x>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s+"
        r"(?P<y>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s+"
        r"(?P<z>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$"
    )

    def __init__(self, text: str):
        """
        使用形如：
        N          -0.60217709   -0.03579554    1.28527700
        N           0.63769791   -0.03579554    1.28527700
        ...
        的字符串构造。

        解析时记录：
          - symbols: 原子符号（顺序固定）
          - coords:  坐标列表（Nx3，float）
          - precision: 小数位数（固定为输入中出现的最大位数，用于后续输出）
          - leading_ws: 每行原子符号前的原始前缀（空白），用于尽量复刻格式
        """
        lines = [ln for ln in text.splitlines() if ln.strip() != ""]
        if not lines:
            raise ValueError("输入文本为空。")

        symbols: List[str] = []
        coords: List[List[float]] = []
        leading_ws: List[str] = []

        # 统计小数位数（取所有数值中出现的最大位数，作为固定输出精度）
        dec_places_max = 0

        for ln in lines:
            m = self._line_regex.match(ln)
            if not m:
                raise ValueError(f"无法解析行：{ln!r}")
            lead = m.group("lead")
            sym = m.group("sym")
            x_str, y_str, z_str = m.group("x"), m.group("y"), m.group("z")

            def _dec_places(s: str) -> int:
                if "." in s:
                    return len(s.split(".")[1])
                return 0

            dec_places_max = max(
                dec_places_max,
                _dec_places(x_str),
                _dec_places(y_str),
                _dec_places(z_str),
            )

            x = float(x_str)
            y = float(y_str)
            z = float(z_str)

            leading_ws.append(lead)
            symbols.append(sym)
            coords.append([x, y, z])

        self.symbols: List[str] = symbols
        self.coords: List[List[float]] = coords
        self.n_atoms: int = len(coords)
        self.precision: int = (
            dec_places_max if dec_places_max > 0 else 6
        )  # 兜底至少 6 位小数
        self.leading_ws: List[str] = leading_ws

    # === 辅助：从现有对象创建新对象（复制元数据） ===
    def _new_with_coords(self, new_coords: List[List[float]]) -> "Molecule":
        if len(new_coords) != self.n_atoms:
            raise ValueError("新坐标的原子数与当前对象不一致。")
        m = object.__new__(Molecule)
        m.symbols = self.symbols
        m.coords = new_coords
        m.n_atoms = self.n_atoms
        m.precision = self.precision
        m.leading_ws = self.leading_ws
        return m

    # === 输出 ===
    def to_string(self, keep_leading_ws: bool = True) -> str:
        """
        将当前坐标输出为字符串。默认保留每行的原始前导空白。
        为了整齐，会按列对齐：先格式化到固定小数位，再按每列的最长宽度右对齐。
        """
        # 先把每个数格式化为固定小数位字符串
        fmt_vals = [
            [
                f"{row[0]:.{self.precision}f}",
                f"{row[1]:.{self.precision}f}",
                f"{row[2]:.{self.precision}f}",
            ]
            for row in self.coords
        ]

        # 组装每行
        lines = []
        for idx in range(self.n_atoms):
            lead = self.leading_ws[idx] if keep_leading_ws else ""
            sym = self.symbols[idx]
            x_s = f"{self.coords[idx][0]:.{self.precision}f}"
            y_s = f"{self.coords[idx][1]:.{self.precision}f}"
            z_s = f"{self.coords[idx][2]:.{self.precision}f}"
            if sym.strip() == "":
                line = f"{lead}{x_s}   {y_s}   {z_s}"
            else:
                line = f"{lead}{sym}  {x_s}   {y_s}   {z_s}"
            lines.append(line)
        return "\n".join(lines)

    # === 坐标运算（支持像 NumPy 数组的运算） ===
    def reframe(
        self,
        i: int,
        j: int,
        axis: str = "x",
        constraint_atom: Optional[int] = None,
        constraint_axis: Optional[str] = None,
        random_seed: Optional[int] = None,
    ) -> "Molecule":
        """
        重新定向坐标系。

        参数：
        - i, j: 原子索引 (1-based)，i->j 定义主轴方向。
        - axis: 主轴 ('x', 'y', 'z')。
        - constraint_atom: 约束原子索引 (1-based)。
        - constraint_axis: 法向量轴 ('x', 'y', 'z')，不能与 axis 相同。

        逻辑：
        - 主轴 (axis) 设为 i->j 方向。
        - 法向量轴 (constraint_axis) 设为 i,j,constraint_atom 平面的法向量。
        - 第三个轴通过右手系计算。
        """
        axis = axis.lower()
        if axis not in ("x", "y", "z"):
            raise ValueError("axis 必须是 'x'、'y' 或 'z'。")

        if not (1 <= i <= self.n_atoms and 1 <= j <= self.n_atoms):
            raise ValueError("i 或 j 越界。")
        if i == j:
            raise ValueError("i 与 j 不可相同。")

        if constraint_axis is None:
            raise ValueError("必须提供 constraint_axis。")
        if constraint_atom is None:
            raise ValueError("必须提供 constraint_atom。")

        c = constraint_axis.lower()
        if c not in ("x", "y", "z"):
            raise ValueError("constraint_axis 必须是 'x'、'y' 或 'z'。")
        if c == axis:
            raise ValueError("constraint_axis 不能与主轴 axis 相同。")
        if not (1 <= constraint_atom <= self.n_atoms):
            raise ValueError("constraint_atom 越界。")
        if constraint_atom == i or constraint_atom == j:
            raise ValueError("constraint_atom 不能与 i 或 j 相同。")

        # 计算向量
        ri = self.coords[i - 1]
        rj = self.coords[j - 1]
        rk = self.coords[constraint_atom - 1]
        u_main = _normalize(_sub(rj, ri))  # 主轴方向
        normal = _normalize(_cross(_sub(rj, ri), _sub(rk, ri)))  # 法向量

        # 设置轴
        if axis == "x":
            xhat = u_main
            if c == "y":
                yhat = normal
                zhat = _normalize(_cross(xhat, yhat))
            elif c == "z":
                zhat = normal
                yhat = _normalize(_cross(zhat, xhat))  # z × x = y (右手系)
        elif axis == "y":
            yhat = u_main
            if c == "x":
                xhat = normal
                zhat = _normalize(_cross(xhat, yhat))
            elif c == "z":
                zhat = normal
                xhat = _normalize(_cross(yhat, zhat))  # y × z = x (右手系)
        elif axis == "z":
            zhat = u_main
            if c == "x":
                xhat = normal
                yhat = _normalize(_cross(zhat, xhat))  # z × x = y (右手系)
            elif c == "y":
                yhat = normal
                xhat = _normalize(_cross(yhat, zhat))  # y × z = x (右手系)

        # 确保右手系
        triple = _dot(_cross(xhat, yhat), zhat)
        if triple < 0:
            zhat = [-z for z in zhat]

        # 计算新坐标
        new_coords: List[List[float]] = []
        for r in self.coords:
            new_coords.append([_dot(r, xhat), _dot(r, yhat), _dot(r, zhat)])

        return self._new_with_coords(new_coords)

    # === 其它便捷操作（可选） ===
    def translate(self, dx: float, dy: float, dz: float) -> "Molecule":
        """整体平移，返回新对象。"""
        delta = [dx, dy, dz]
        return self._new_with_coords([_add(r, delta) for r in self.coords])

    def rotate_about_axes(
        self, ax: float = 0.0, ay: float = 0.0, az: float = 0.0
    ) -> "Molecule":
        """
        以当前坐标系的 x,y,z 轴依次旋转（右手法则，单位：弧度），返回新对象。
        """
        sx, cx = math.sin(ax), math.cos(ax)
        sy, cy = math.sin(ay), math.cos(ay)
        sz, cz = math.sin(az), math.cos(az)

        # 旋转矩阵 Rz * Ry * Rx
        Rx = [[1, 0, 0], [0, cx, -sx], [0, sx, cx]]
        Ry = [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]
        Rz = [[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]

        def matmul3(A, B):
            return [
                [
                    A[0][0] * B[0][j] + A[0][1] * B[1][j] + A[0][2] * B[2][j]
                    for j in range(3)
                ],
                [
                    A[1][0] * B[0][j] + A[1][1] * B[1][j] + A[1][2] * B[2][j]
                    for j in range(3)
                ],
                [
                    A[2][0] * B[0][j] + A[2][1] * B[1][j] + A[2][2] * B[2][j]
                    for j in range(3)
                ],
            ]

        R = matmul3(Rz, matmul3(Ry, Rx))

        def apply_R(r):
            return [
                r[0] * R[0][0] + r[1] * R[0][1] + r[2] * R[0][2],
                r[0] * R[1][0] + r[1] * R[1][1] + r[2] * R[1][2],
                r[0] * R[2][0] + r[1] * R[2][1] + r[2] * R[2][2],
            ]

        return self._new_with_coords([apply_R(r) for r in self.coords])

    # === 运算符重载：支持像 NumPy 数组的逐元素运算 ===
    def __add__(self, other):
        if isinstance(other, Molecule):
            if self.n_atoms != other.n_atoms:
                raise ValueError(f"原子数不一致：{self.n_atoms} vs {other.n_atoms}")
            # 检查原子符号：忽略空符号，只比较非空符号
            for s1, s2 in zip(self.symbols, other.symbols):
                if s1.strip() and s2.strip() and s1 != s2:
                    raise ValueError("原子符号不匹配。")
            new_coords = [
                [a + b for a, b in zip(c1, c2)]
                for c1, c2 in zip(self.coords, other.coords)
            ]
            new_leading_ws = [
                (
                    self.leading_ws[idx]
                    if self.symbols[idx].strip()
                    else other.leading_ws[idx]
                )
                for idx in range(self.n_atoms)
            ]
            m = Molecule.__new__(Molecule)
            m.symbols = self.symbols
            m.coords = new_coords
            m.n_atoms = self.n_atoms
            m.precision = self.precision
            m.leading_ws = new_leading_ws
            return m
        elif isinstance(other, (int, float)):
            new_coords = [[c + other for c in coord] for coord in self.coords]
            m = Molecule.__new__(Molecule)
            m.symbols = self.symbols
            m.coords = new_coords
            m.n_atoms = self.n_atoms
            m.precision = self.precision
            m.leading_ws = self.leading_ws
            return m
        else:
            return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Molecule):
            if self.n_atoms != other.n_atoms:
                raise ValueError(f"原子数不一致：{self.n_atoms} vs {other.n_atoms}")
            # 检查原子符号：忽略空符号，只比较非空符号
            for s1, s2 in zip(self.symbols, other.symbols):
                if s1.strip() and s2.strip() and s1 != s2:
                    raise ValueError("原子符号不匹配。")
            new_coords = [
                [a - b for a, b in zip(c1, c2)]
                for c1, c2 in zip(self.coords, other.coords)
            ]
            new_leading_ws = [
                (
                    self.leading_ws[idx]
                    if self.symbols[idx].strip()
                    else other.leading_ws[idx]
                )
                for idx in range(self.n_atoms)
            ]
            m = Molecule.__new__(Molecule)
            m.symbols = self.symbols
            m.coords = new_coords
            m.n_atoms = self.n_atoms
            m.precision = self.precision
            m.leading_ws = new_leading_ws
            return m
        elif isinstance(other, (int, float)):
            new_coords = [[c - other for c in coord] for coord in self.coords]
            m = Molecule.__new__(Molecule)
            m.symbols = self.symbols
            m.coords = new_coords
            m.n_atoms = self.n_atoms
            m.precision = self.precision
            m.leading_ws = self.leading_ws
            return m
        else:
            return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            new_coords = [[other - c for c in coord] for coord in self.coords]
            return self._new_with_coords(new_coords)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Molecule):
            if self.n_atoms != other.n_atoms:
                raise ValueError(f"原子数不一致：{self.n_atoms} vs {other.n_atoms}")
            # 检查原子符号：忽略空符号，只比较非空符号
            for s1, s2 in zip(self.symbols, other.symbols):
                if s1.strip() and s2.strip() and s1 != s2:
                    raise ValueError("原子符号不匹配。")
            new_coords = [
                [a * b for a, b in zip(c1, c2)]
                for c1, c2 in zip(self.coords, other.coords)
            ]
            new_leading_ws = [
                (
                    self.leading_ws[idx]
                    if self.symbols[idx].strip()
                    else other.leading_ws[idx]
                )
                for idx in range(self.n_atoms)
            ]
            m = Molecule.__new__(Molecule)
            m.symbols = self.symbols
            m.coords = new_coords
            m.n_atoms = self.n_atoms
            m.precision = self.precision
            m.leading_ws = new_leading_ws
            return m
        elif isinstance(other, (int, float)):
            new_coords = [[c * other for c in coord] for coord in self.coords]
            m = Molecule.__new__(Molecule)
            m.symbols = self.symbols
            m.coords = new_coords
            m.n_atoms = self.n_atoms
            m.precision = self.precision
            m.leading_ws = self.leading_ws
            return m
        else:
            return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, Molecule):
            if self.n_atoms != other.n_atoms:
                raise ValueError(f"原子数不一致：{self.n_atoms} vs {other.n_atoms}")
            # 检查原子符号：忽略空符号，只比较非空符号
            for s1, s2 in zip(self.symbols, other.symbols):
                if s1.strip() and s2.strip() and s1 != s2:
                    raise ValueError("原子符号不匹配。")
            new_coords = [
                [a / b for a, b in zip(c1, c2)]
                for c1, c2 in zip(self.coords, other.coords)
            ]
            new_leading_ws = [
                (
                    self.leading_ws[idx]
                    if self.symbols[idx].strip()
                    else other.leading_ws[idx]
                )
                for idx in range(self.n_atoms)
            ]
            m = Molecule.__new__(Molecule)
            m.symbols = self.symbols
            m.coords = new_coords
            m.n_atoms = self.n_atoms
            m.precision = self.precision
            m.leading_ws = new_leading_ws
            return m
        elif isinstance(other, (int, float)):
            new_coords = [[c / other for c in coord] for coord in self.coords]
            m = Molecule.__new__(Molecule)
            m.symbols = self.symbols
            m.coords = new_coords
            m.n_atoms = self.n_atoms
            m.precision = self.precision
            m.leading_ws = self.leading_ws
            return m
        else:
            return NotImplemented

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            new_coords = [[other / c for c in coord] for coord in self.coords]
            return self._new_with_coords(new_coords)
        return NotImplemented
