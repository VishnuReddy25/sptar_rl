import subprocess
import argparse


def multirun(args):
    # Convert args namespace to dictionary
    arg_dict = vars(args)

    for exp in arg_dict["exp_names"]:
        print(f"GPU {arg_dict['gpu_id']} Training: {arg_dict['dataset_name']} on {exp}")

        # Uncomment below if training is needed
        if arg_dict["version"] == "v1":
            train_command = [
                "python", "zhiyuan/retriever/dpr/train/train_sbert.py",
                "--dataset_name", arg_dict["dataset_name"],
                "--train_num", str(arg_dict["train_num"]),
                "--weak_num", arg_dict["weak_num"],
                "--exp_name", exp
            ]
        elif arg_dict["version"] == "v2":
            train_command = [
                "python", "zhiyuan/retriever/dpr/train/train_sbert_BM25_hardnegs.py",
                "--dataset_name", arg_dict["dataset_name"],
                "--train_num", str(arg_dict["train_num"]),
                "--weak_num", arg_dict["weak_num"],
                "--exp_name", exp
            ]
        print("Running training command:", " ".join(train_command))
        subprocess.call(train_command)

        # Always run evaluation
        eval_command = [
            "python", "zhiyuan/retriever/dpr/eval/evaluate_sbert.py",
            "--dataset_name", arg_dict["dataset_name"],
            "--train_num", str(arg_dict["train_num"]),
            "--exp_name", exp,
            "--dpr_v", arg_dict["version"]
        ]
        print("Running evaluation command:", " ".join(eval_command), "\n")
        subprocess.call(eval_command)


def main():
    parser = argparse.ArgumentParser(description='Run training and evaluation scripts.')
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--version", type=str, choices=["v1", "v2"], required=True)
    parser.add_argument("--gpu_id", type=int, required=True)
    parser.add_argument("--train_num", type=int, required=True)
    parser.add_argument("--weak_num", type=str, required=True)
    parser.add_argument("--exp_names", nargs='+', required=True, help="List of experiment names")
    
    args = parser.parse_args()
    multirun(args)


if __name__ == "__main__":
    main()
