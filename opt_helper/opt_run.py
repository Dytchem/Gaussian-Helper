import os
from opt_helper.opt_in import opt_in
from opt_helper.opt_out import opt_out
from Molecule import Molecule


class opt_run:
    def __init__(
        self,
        file_name: str,
        in_file_root: str,
        out_file_root: str,
        epoch: int,
        update_coefficient=1.0,
        pre_process=None,
    ):
        in_file = opt_in(os.path.join(in_file_root, file_name + ".in"))
        for i in range(1, epoch + 1):
            if pre_process:
                pre_process(in_file)
            in_file_path = os.path.join(in_file_root, f"{file_name}_step{i}.in")
            out_file_path = os.path.join(out_file_root, f"{file_name}_step{i}.out")
            in_file.save_to_file(in_file_path)
            print(
                f"Running optimization step {i}: g09 < {in_file_path} > {out_file_path}"
            )
            os.system(f"g09 < {in_file_path} > {out_file_path}")
            print(f"Finished optimization step {i}")
            out_file = opt_out(out_file_path)
            if out_file.is_normal():
                first_freq, vib_mol = out_file.get_first_frequency_info()
                print(f"First frequency: {first_freq}")
                if first_freq > 0:
                    print("Optimization has converged.")
                    break
                else:
                    print("Updating geometry based on first vibrational mode.")
                    in_file.molecule = in_file.molecule + update_coefficient * Molecule(
                        vib_mol
                    )
            else:
                print("Optimization did not terminate normally.")
                in_file.molecule = out_file.final_molecule
        # to do:  more work on in_file

    def default_pre_process(self, in_file):
        """
        默认前处理函数：重新定向坐标系，使原子 1->2 为 x 轴，原子 3 在 xz 平面。
        """
        in_file.molecule = in_file.molecule.reframe(1, 2, "x", 3, "z")
