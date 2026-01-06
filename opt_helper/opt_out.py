import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from Molecule import Molecule


class opt_out:
    def __init__(self, file_path: str):
        with open(file_path, 'r') as f:
            self.content = f.read()
        
        # 检查终止状态
        if "Normal termination" in self.content:
            self.termination = "normal"
        elif "Error termination" in self.content:
            self.termination = "error"
        else:
            self.termination = "unknown"
        
        # 提取最后一个 Input orientation 的坐标
        lines = self.content.splitlines()
        input_orientations = []
        for i, line in enumerate(lines):
            if "Input orientation:" in line:
                input_orientations.append(i)
        if input_orientations:
            last_io = input_orientations[-1]
            # 找到坐标开始行
            coord_start = None
            for j in range(last_io, len(lines)):
                if "Coordinates (Angstroms)" in lines[j]:
                    coord_start = j + 2  # 跳过分隔线
                    break
            if coord_start:
                coords = []
                k = coord_start
                while len(coords) < 10 and k < len(lines):
                    line = lines[k].strip()
                    parts = line.split()
                    if len(parts) >= 6:
                        try:
                            x = float(parts[3])
                            y = float(parts[4])
                            z = float(parts[5])
                            coords.append([x, y, z])
                        except ValueError:
                            pass
                    k += 1
                if coords:
                    coord_text = "\n".join(f"{c[0]} {c[1]} {c[2]}" for c in coords)
                    self.final_molecule = Molecule(coord_text)
                else:
                    self.final_molecule = None
            else:
                self.final_molecule = None
        else:
            self.final_molecule = None
        
        # 如果正常终止，读取频率信息
        if self.termination == "normal":
            lines = self.content.splitlines()
            
            # 找到频率部分
            freq_start = None
            for i, line in enumerate(lines):
                if "Frequencies --" in line:
                    freq_start = i
                    break
            if freq_start is None:
                raise ValueError("No frequencies found in normal termination file")
            
            # 找到原子振动行
            atom_line = None
            for i in range(freq_start, len(lines)):
                if "Atom  AN" in lines[i]:
                    atom_line = i
                    break
            if atom_line is None:
                raise ValueError("No atom vibration lines found")
            
            # 提取第一个频率值
            freq_line = lines[freq_start]
            freq_parts = freq_line.split()
            # 假设格式为 "Frequencies --    value1               value2               value3"
            self.first_freq = float(freq_parts[2])
            
            # 提取第一个频率的振动方向（每原子 X Y Z）
            vib_lines = lines[atom_line + 1 : atom_line + 11]  # 假设 10 原子
            vib_vectors = []
            for line in vib_lines:
                parts = line.split()
                if len(parts) < 5:
                    continue  # 跳过不完整的行
                x = float(parts[2])
                y = float(parts[3])
                z = float(parts[4])
                vib_vectors.append(f"{x:.2f}  {y:.2f}   {z:.2f}")
            self.vib_string = "\n".join(vib_vectors)
        else:
            self.first_freq = None
            self.vib_string = None
    
    def is_normal(self):
        """
        返回是否为正常终止。
        """
        return self.termination == "normal"
    
    def get_first_frequency_info(self):
        """
        返回第一个频率的值和振动方向字符串。
        如果不是正常终止，返回 (None, None)。
        """
        if self.termination != "normal":
            return None, None
        return self.first_freq, self.vib_string