
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
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


def _cross(a: List[float], b: List[float]) -> List[float]:
    """叉乘 a×b（右手法则）"""
    return [a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0]]


def _norm(v: List[float]) -> float:
    """向量长度 ||v||"""
    return math.sqrt(_dot(v, v))


def _normalize(v: List[float], eps: float = 1e-12) -> List[float]:
    """单位化 v/||v||，长度过小则报错"""
    n = _norm(v)
    if n < eps:
        raise ValueError("规范化失败：向量长度过小（接近零）。")
    return [v[0]/n, v[1]/n, v[2]/n]


def _sub(a: List[float], b: List[float]) -> List[float]:
    """向量相减 a - b"""
    return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]


def _add(a: List[float], b: List[float]) -> List[float]:
    """向量相加 a + b"""
    return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]


def _scale(v: List[float], s: float) -> List[float]:
    """标量乘法 s * v"""
    return [v[0]*s, v[1]*s, v[2]*s]


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
    - 坐标系重建（以 i->j 为新的 x/y/z 轴；可约束某原子落在指定平面并为指定轴的正方向）。
    - add 操作（逐元素相加，返回新对象）。

    所有变换方法都返回新的 Molecule 对象（原对象不变，便于链式操作与回溯）。
    """

    _line_regex = re.compile(
        r'^(?P<lead>\s*)(?P<sym>[A-Za-z]*)\s*'
        r'(?P<x>[+-]?\d+(?:\.\d+)?)\s+'
        r'(?P<y>[+-]?\d+(?:\.\d+)?)\s+'
        r'(?P<z>[+-]?\d+(?:\.\d+)?)\s*$'
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

            dec_places_max = max(dec_places_max, _dec_places(x_str), _dec_places(y_str), _dec_places(z_str))

            x = float(x_str)
            y = float(y_str)
            z = float(z_str)

            leading_ws.append(lead)
            symbols.append(sym)
            coords.append([x, y, z])

        self.symbols: List[str] = symbols
        self.coords: List[List[float]] = coords
        self.n_atoms: int = len(coords)
        self.precision: int = dec_places_max if dec_places_max > 0 else 6  # 兜底至少 6 位小数
        self.leading_ws: List[str] = leading_ws

    # === 辅助：从现有对象创建新对象（复制元数据） ===
    def _new_with_coords(self, new_coords: List[List[float]]) -> "Molecule":
        if len(new_coords) != self.n_atoms:
            raise ValueError("新坐标的原子数与当前对象不一致。")
        m = deepcopy(self)
        m.coords = deepcopy(new_coords)
        return m

    # === 输出 ===
    def to_string(self, keep_leading_ws: bool = True) -> str:
        """
        将当前坐标输出为字符串。默认保留每行的原始前导空白。
        为了整齐，会按列对齐：先格式化到固定小数位，再按每列的最长宽度右对齐。
        """
        # 先把每个数格式化为固定小数位字符串
        fmt_vals = [[
            f"{row[0]:.{self.precision}f}",
            f"{row[1]:.{self.precision}f}",
            f"{row[2]:.{self.precision}f}"
        ] for row in self.coords]

        # 找到每列的最大字符串长度用于对齐
        col_widths = [max(len(v[i]) for v in fmt_vals) for i in range(3)]

        # 组装每行
        lines = []
        for idx in range(self.n_atoms):
            lead = self.leading_ws[idx] if keep_leading_ws else ""
            sym = self.symbols[idx]
            x_s, y_s, z_s = fmt_vals[idx]
            # 右对齐每列
            x_s = x_s.rjust(col_widths[0])
            y_s = y_s.rjust(col_widths[1])
            z_s = z_s.rjust(col_widths[2])
            if sym.strip() == '':
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
        random_seed: Optional[int] = None
    ) -> "Molecule":
        """
        将 i -> j 定义为新的 axis（'x' / 'y' / 'z'）轴正方向。
        若提供 constraint_atom=k 与 constraint_axis=c（c ∈ {'x','y','z'} 且 c != axis），则：
          - 原子 k 位于由 axis 与 c 张成的平面（即在剩余那一轴方向上坐标 = 0）；
          - 同时原子 k 在新坐标的 c 分量为正（若几何上该分量本来为 0，则保留为 0）。

        未提供约束时，随机选择其余轴方向（可用 random_seed 控制复现）。

        说明：
        - 本方法只改变坐标系方向，不做平移（原坐标原点保持不变）。
        - 新坐标计算采用点乘投影：r' = [r·x̂, r·ŷ, r·ẑ]。
        - i、j、constraint_atom 为 1-based 下标。
        """
        axis = axis.lower()
        if axis not in ("x", "y", "z"):
            raise ValueError("axis 必须是 'x'、'y' 或 'z'。")

        if not (1 <= i <= self.n_atoms and 1 <= j <= self.n_atoms):
            raise ValueError("i 或 j 越界。")
        if i == j:
            raise ValueError("i 与 j 不可相同。")

        # 主轴方向：i -> j
        ri = self.coords[i-1]
        rj = self.coords[j-1]
        uA = _normalize(_sub(rj, ri))  # 主轴单位向量

        # 处理约束逻辑
        if constraint_axis is not None:
            c = constraint_axis.lower()
            if c not in ("x", "y", "z"):
                raise ValueError("constraint_axis 必须是 'x'、'y' 或 'z'。")
            if c == axis:
                raise ValueError("constraint_axis 不能与主轴 axis 相同。")
            if constraint_atom is None:
                raise ValueError("提供了 constraint_axis，但缺少 constraint_atom。")
            if not (1 <= constraint_atom <= self.n_atoms):
                raise ValueError("constraint_atom 越界。")
        else:
            c = None  # 无约束

        # 将 (axis, c) 两轴张成的平面作为约束平面；剩余轴为 B（该轴坐标需为 0）
        # 例如 axis='y', c='z' -> 平面为 yz；剩余轴 B='x'
        axes = ("x", "y", "z")
        # 按选择确定 A(主轴)、C(约束指定轴)、B(剩余轴)
        A = axis
        C = c
        B = None
        if C is not None:
            B = ({*axes} - {A, C}).pop()

        # 依据 A、C 构造 x̂,ŷ,ẑ（三者正交且右手系，满足约束）
        xhat = yhat = zhat = None

        def build_without_constraint():
            """无约束：从随机种子构造其余两轴，确保右手系。"""
            seed = _random_noncollinear(random_seed)
            # second 轴：在与主轴正交的子空间内从种子正交化
            second = _sub(seed, _scale(uA, _dot(seed, uA)))
            if _norm(second) < 1e-12:
                # 兜底选择与主轴不共线的固定方向
                candidates = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                for cand in candidates:
                    second = _sub(cand, _scale(uA, _dot(cand, uA)))
                    if _norm(second) > 1e-12:
                        break
            second = _normalize(second)
            # 根据主轴是哪一个，给 second 指定到合适的轴位，再由叉乘补全第三轴
            if A == "x":
                nonlocal xhat, yhat, zhat
                xhat = uA
                yhat = second
                zhat = _normalize(_cross(xhat, yhat))
            elif A == "y":
                yhat = uA
                xhat = second
                zhat = _normalize(_cross(xhat, yhat))
            else:  # A == "z"
                zhat = uA
                yhat = second
                xhat = _normalize(_cross(yhat, zhat))

        def build_with_constraint():
            """有约束：将 k 放在由 A 与 C 张成的平面，并令 k 在 C 轴上为正。"""
            nonlocal xhat, yhat, zhat
            rk = self.coords[constraint_atom - 1]

            # 计算 rk 在主轴 A 的正交分量（用于定义 C 轴方向）
            # rk_perp = rk - (rk·uA) uA
            rk_perp = _sub(rk, _scale(uA, _dot(rk, uA)))

            if _norm(rk_perp) >= 1e-12:
                uC = _normalize(rk_perp)
                # 保证 k 在 C 轴为正方向：若投影为负则翻转 C 轴
                if _dot(rk, uC) < 0:
                    uC = _scale(uC, -1.0)
            else:
                # 退化：rk 与主轴 A 共线（或几乎共线）
                # 此时 k 自然位于任何包含 A 的平面；在 C 轴上的分量为 0，无需强制正向。
                # 这里选择一个与 A 不共线的固定向量，并正交化得到 uC。
                candidates = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                uC = None
                for cand in candidates:
                    tmp = _sub(cand, _scale(uA, _dot(cand, uA)))
                    if _norm(tmp) > 1e-12:
                        uC = _normalize(tmp)
                        break
                if uC is None:
                    raise RuntimeError("无法为约束轴选择与主轴正交的方向（数值退化）。")

            # 根据 A、C 的组合，确定三轴，并确保右手系与正交性：
            if A == "x" and C == "y":
                xhat = uA
                yhat = uC
                zhat = _normalize(_cross(xhat, yhat))  # z = x × y
            elif A == "x" and C == "z":
                xhat = uA
                zhat = uC
                yhat = _normalize(_cross(zhat, xhat))  # 保证 z = x × y
            elif A == "y" and C == "x":
                yhat = uA
                xhat = uC
                zhat = _normalize(_cross(xhat, yhat))  # z = x × y
            elif A == "y" and C == "z":
                yhat = uA
                zhat = uC
                xhat = _normalize(_cross(yhat, zhat))  # z = x × y
            elif A == "z" and C == "x":
                zhat = uA
                xhat = uC
                yhat = _normalize(_cross(zhat, xhat))  # z = x × y
            elif A == "z" and C == "y":
                zhat = uA
                yhat = uC
                xhat = _normalize(_cross(yhat, zhat))  # z = x × y
            else:
                raise RuntimeError("未预期的 A/C 组合。")

            # 到此：k 的新坐标满足在 B 轴为 0（因为 rk 属于 span(uA, uC)），
            # 且在 C 轴方向上为正（非退化时保证；退化时为 0）。

        # 执行构造
        if C is None:
            build_without_constraint()
        else:
            build_with_constraint()

        # 数值健壮性检查：三轴需两两正交且单位化（容忍极小误差）
        def _check_orthonormal(xhat, yhat, zhat, tol=1e-8):
            if abs(_dot(xhat, yhat)) > tol or abs(_dot(xhat, zhat)) > tol or abs(_dot(yhat, zhat)) > tol:
                raise RuntimeError("构造的坐标轴未能正交（数值不稳定）。")
            if abs(_norm(xhat) - 1.0) > tol or abs(_norm(yhat) - 1.0) > tol or abs(_norm(zhat) - 1.0) > tol:
                raise RuntimeError("构造的坐标轴未单位化。")
            # 右手性检查：要求 (x × y) 与 z 同向
            triple = _dot(_cross(xhat, yhat), zhat)
            if triple < 0:
                # 若出现负向，翻转 z 轴（最不影响用户指定的 A/C 取向）
                zhat[:] = _scale(zhat, -1.0)

        _check_orthonormal(xhat, yhat, zhat)

        # 计算新坐标：r' = [r·x̂, r·ŷ, r·ẑ]
        new_coords: List[List[float]] = []
        for r in self.coords:
            new_coords.append([_dot(r, xhat), _dot(r, yhat), _dot(r, zhat)])

        return self._new_with_coords(new_coords)

    # === 其它便捷操作（可选） ===
    def translate(self, dx: float, dy: float, dz: float) -> "Molecule":
        """整体平移，返回新对象。"""
        delta = [dx, dy, dz]
        return self._new_with_coords([_add(r, delta) for r in self.coords])

    def rotate_about_axes(self, ax: float = 0.0, ay: float = 0.0, az: float = 0.0) -> "Molecule":
        """
        以当前坐标系的 x,y,z 轴依次旋转（右手法则，单位：弧度），返回新对象。
        """
        sx, cx = math.sin(ax), math.cos(ax)
        sy, cy = math.sin(ay), math.cos(ay)
        sz, cz = math.sin(az), math.cos(az)

        # 旋转矩阵 Rz * Ry * Rx
        Rx = [
            [1, 0, 0],
            [0, cx, -sx],
            [0, sx, cx]
        ]
        Ry = [
            [cy, 0, sy],
            [0, 1, 0],
            [-sy, 0, cy]
        ]
        Rz = [
            [cz, -sz, 0],
            [sz, cz, 0],
            [0, 0, 1]
        ]

        def matmul3(A, B):
            return [
                [
                    A[0][0]*B[0][j] + A[0][1]*B[1][j] + A[0][2]*B[2][j]
                    for j in range(3)
                ],
                [
                    A[1][0]*B[0][j] + A[1][1]*B[1][j] + A[1][2]*B[2][j]
                    for j in range(3)
                ],
                [
                    A[2][0]*B[0][j] + A[2][1]*B[1][j] + A[2][2]*B[2][j]
                    for j in range(3)
                ],
            ]

        R = matmul3(Rz, matmul3(Ry, Rx))

        def apply_R(r):
            return [
                r[0]*R[0][0] + r[1]*R[0][1] + r[2]*R[0][2],
                r[0]*R[1][0] + r[1]*R[1][1] + r[2]*R[1][2],
                r[0]*R[2][0] + r[1]*R[2][1] + r[2]*R[2][2],
            ]

        return self._new_with_coords([apply_R(r) for r in self.coords])

    # === 运算符重载：支持像 NumPy 数组的逐元素运算 ===
    def __add__(self, other):
        if isinstance(other, Molecule):
            if self.n_atoms != other.n_atoms:
                raise ValueError(f"原子数不一致：{self.n_atoms} vs {other.n_atoms}")
            new_coords = [[a + b for a, b in zip(c1, c2)] for c1, c2 in zip(self.coords, other.coords)]
        elif isinstance(other, (int, float)):
            new_coords = [[c + other for c in coord] for coord in self.coords]
        else:
            return NotImplemented
        return self._new_with_coords(new_coords)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Molecule):
            if self.n_atoms != other.n_atoms:
                raise ValueError(f"原子数不一致：{self.n_atoms} vs {other.n_atoms}")
            new_coords = [[a - b for a, b in zip(c1, c2)] for c1, c2 in zip(self.coords, other.coords)]
        elif isinstance(other, (int, float)):
            new_coords = [[c - other for c in coord] for coord in self.coords]
        else:
            return NotImplemented
        return self._new_with_coords(new_coords)

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            new_coords = [[other - c for c in coord] for coord in self.coords]
            return self._new_with_coords(new_coords)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Molecule):
            if self.n_atoms != other.n_atoms:
                raise ValueError(f"原子数不一致：{self.n_atoms} vs {other.n_atoms}")
            new_coords = [[a * b for a, b in zip(c1, c2)] for c1, c2 in zip(self.coords, other.coords)]
        elif isinstance(other, (int, float)):
            new_coords = [[c * other for c in coord] for coord in self.coords]
        else:
            return NotImplemented
        return self._new_with_coords(new_coords)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, Molecule):
            if self.n_atoms != other.n_atoms:
                raise ValueError(f"原子数不一致：{self.n_atoms} vs {other.n_atoms}")
            new_coords = [[a / b for a, b in zip(c1, c2)] for c1, c2 in zip(self.coords, other.coords)]
        elif isinstance(other, (int, float)):
            new_coords = [[c / other for c in coord] for coord in self.coords]
        else:
            return NotImplemented
        return self._new_with_coords(new_coords)

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            new_coords = [[other / c for c in coord] for coord in self.coords]
            return self._new_with_coords(new_coords)
        return NotImplemented
