import os
import datetime
import random
import argparse
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
            in_file_path = os.path.join(
                in_file_root, f"{file_name}_step{i:0{len(str(epoch))}d}.in"
            )
            out_file_path = os.path.join(
                out_file_root, f"{file_name}_step{i:0{len(str(epoch))}d}.out"
            )
            in_file.save_to_file(in_file_path)
            print(
                f"[{datetime.datetime.now()}] Running optimization step {i}: g09 < {in_file_path} > {out_file_path}"
            )
            os.system(f"g09 < {in_file_path} > {out_file_path}")
            print(f"[{datetime.datetime.now()}] Finished optimization step {i}")
            out_file = opt_out(out_file_path)
            if out_file.is_normal():
                first_freq, vib_mol = out_file.get_first_frequency_info()
                print(f"[{datetime.datetime.now()}] First frequency: {first_freq}")
                if first_freq > 0:
                    print(f"[{datetime.datetime.now()}] Optimization has converged.")
                    break
                else:
                    print(
                        f"[{datetime.datetime.now()}] Updating geometry based on first vibrational mode."
                    )
                    sign = random.choice([1, -1])
                    in_file.molecule = (
                        in_file.molecule + sign * update_coefficient * Molecule(vib_mol)
                    )
            else:
                print(
                    f"[{datetime.datetime.now()}] Optimization did not terminate normally."
                )
                in_file.molecule = 0 * in_file.molecule + out_file.final_molecule
        # to do:  more work on in_file

    @staticmethod
    def default_pre_process(in_file):
        """
        默认前处理函数：重新定向坐标系，使原子 1->2 为 x 轴，原子 3 在 xz 平面。
        """
        in_file.molecule = in_file.molecule.reframe(1, 2, "x", 3, "z")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Gaussian optimization.")
    parser.add_argument("file_name", type=str, help="Base name of the input file")
    parser.add_argument("in_file_root", type=str, help="Root directory for input files")
    parser.add_argument(
        "out_file_root", type=str, help="Root directory for output files"
    )
    parser.add_argument("epoch", type=int, help="Number of optimization epochs")
    parser.add_argument(
        "--update_coefficient",
        type=float,
        default=1.0,
        help="Update coefficient (default: 1.0)",
    )
    parser.add_argument(
        "--use_default_preprocess",
        action="store_true",
        help="Use default preprocessing function",
    )

    args = parser.parse_args()

    pre_process = opt_run.default_pre_process if args.use_default_preprocess else None

    opt_run(
        file_name=args.file_name,
        in_file_root=args.in_file_root,
        out_file_root=args.out_file_root,
        epoch=args.epoch,
        update_coefficient=args.update_coefficient,
        pre_process=pre_process,
    )
