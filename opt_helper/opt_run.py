import os
import datetime
import random
import sys

sys.path.append("..")
import argparse
from opt_in import opt_in
from opt_out import opt_out
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
        self.converged_once = False
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
                    if self.converged_once:
                        print(
                            f"[{datetime.datetime.now()}] Optimization has converged twice in a row."
                        )
                        break
                    else:
                        self.converged_once = True
                        print(
                            f"[{datetime.datetime.now()}] Optimization has converged once, running one more step."
                        )
                else:
                    self.converged_once = False
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
    def default_pre_process(
        in_file, i=1, j=2, axis="x", constraint_atom=3, constraint_axis="y"
    ):
        """
        默认前处理函数：重新定向坐标系，使原子 i->j 为指定的 axis，原子 constraint_atom 在 axis-constraint_axis 平面。
        """
        in_file.molecule = in_file.molecule.reframe(
            i, j, axis, constraint_atom, constraint_axis
        )


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
    parser.add_argument(
        "--reframe_params",
        type=str,
        default="1 2 x 3 y",
        help='Reframe parameters: "i j axis constraint_atom constraint_axis" (default: "1 2 x 3 y")',
    )

    args = parser.parse_args()

    if args.use_default_preprocess:
        # Parse reframe_params
        try:
            params = args.reframe_params.split()
            if len(params) != 5:
                raise ValueError("reframe_params must have 5 values")
            i, j, axis, constraint_atom, constraint_axis = params
            i = int(i)
            j = int(j)
            axis = axis.lower()
            constraint_atom = int(constraint_atom)
            constraint_axis = constraint_axis.lower()
            if axis not in ["x", "y", "z"] or constraint_axis not in ["x", "y", "z"]:
                raise ValueError("axis and constraint_axis must be x, y, or z")
        except ValueError as e:
            print(f"Error parsing reframe_params: {e}")
            sys.exit(1)

        pre_process = lambda in_file: opt_run.default_pre_process(
            in_file,
            i=i,
            j=j,
            axis=axis,
            constraint_atom=constraint_atom,
            constraint_axis=constraint_axis,
        )
    else:
        pre_process = None

    opt_run(
        file_name=args.file_name,
        in_file_root=args.in_file_root,
        out_file_root=args.out_file_root,
        epoch=args.epoch,
        update_coefficient=args.update_coefficient,
        pre_process=pre_process,
    )
