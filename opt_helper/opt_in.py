import re
import sys

sys.path.append("..")
from Molecule import Molecule
from . import *


class opt_in:
    def __init__(self, file_path: str):
        with open(file_path, "r") as f:
            lines = f.readlines()
        self.content = "".join(lines)

        # 解析 MaxStep
        self.max_step = None
        for line in lines:
            if line.strip().startswith("#p"):
                match = re.search(r"MaxStep=(\d+)", line)
                if match:
                    self.max_step = int(match.group(1))
                break

        # 提取原子坐标
        atom_start = None
        for i, line in enumerate(lines):
            if re.match(r"^[A-Za-z]+\s+[-+]?\d", line.strip()):
                atom_start = i
                break
        if atom_start is None:
            raise ValueError("No atom coordinates found")
        atom_end = atom_start
        while atom_end < len(lines) and re.match(
            r"^[A-Za-z]+\s+[-+]?\d", lines[atom_end].strip()
        ):
            atom_end += 1
        atom_text = "".join(lines[atom_start:atom_end])
        self.molecule = Molecule(atom_text)

        # 存储原子坐标位置
        self.atom_start = atom_start
        self.atom_end = atom_end

        # 其他内容
        self.other_content = "".join(lines[:atom_start] + lines[atom_end:])

    def update_max_step(self, new_max_step: int):
        """
        更新 MaxStep 参数的值。
        """
        lines = self.content.splitlines(keepends=True)
        for i, line in enumerate(lines):
            if line.strip().startswith("#p"):
                if "MaxStep=" in line:
                    lines[i] = re.sub(r"MaxStep=\d+", f"MaxStep={new_max_step}", line)
                else:
                    lines[i] = line.rstrip() + f",MaxStep={new_max_step}\n"
                break
        self.content = "".join(lines)
        self.max_step = new_max_step

    def get_full_content(self) -> str:
        """
        返回完整的文件内容字符串，包括可能的分子坐标更新。
        """
        lines = self.content.splitlines(keepends=True)
        new_atom_lines = self.molecule.to_string().splitlines()
        new_lines_list = []
        current_atom_idx = 0
        for line in lines[self.atom_start : self.atom_end]:
            if re.match(r"^[A-Za-z]+\s+[-+]?\d", line.strip()):
                # 这是一个原子行，替换坐标
                new_line = new_atom_lines[current_atom_idx] + "\n"
                current_atom_idx += 1
            else:
                # 空行或其他，保持原样
                new_line = line
            new_lines_list.append(new_line)
        lines[self.atom_start : self.atom_end] = new_lines_list
        return "".join(lines)

    def save_to_file(self, file_path: str):
        """
        将当前内容保存到指定文件路径。
        """
        full_content = self.get_full_content()
        with open(file_path, "w") as f:
            f.write(full_content)
