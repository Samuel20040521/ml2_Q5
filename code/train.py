from argutils import print_args
from encoder.train import train
from pathlib import Path
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("game_type", type=str, help="Name for game type.")
    parser.add_argument("run_id", type=str, help="Name for this model instance. If a model state from the same run ID was previously "
                        "saved, the training will restart from there. Pass -f to overwrite saved states and "
                        "restart from scratch.")
    parser.add_argument("-d", "--data_dir", type=Path, default="/datadrive/final_encoder_data/train", help="Path to preprocessed data")
    parser.add_argument("-vd", "--validate_data_dir", type=Path, default="/datadrive/final_encoder_data/validate", help="Path to preprocessed data")
    parser.add_argument(
        "-m",
        "--models_dir",
        type=Path,
        default="./models",
        help="Path to the output directory that will contain the saved model weights, as well as "
        "backups of those weights and plots generated during training.")
    parser.add_argument("-s", "--save_every", type=int, default=500, help="Number of steps between updates of the model on the disk. Set to 0 to never save the "
                        "model.")
    parser.add_argument("-b", "--backup_every", type=int, default=1000, help="Number of steps between backups of the model. Set to 0 to never make backups of the "
                        "model.")
    parser.add_argument("-ve", "--validate_every", type=int, default=500, help="Number of steps between validation step.")
    parser.add_argument("-f", "--force_restart", action="store_true", help="Do not load any saved model.")
    parser.add_argument("-c", "--conf_file", type=str, default="./conf.cfg", help="Configure file path")
    args = parser.parse_args()
    print_args(args, parser)

    return args


if __name__ == "__main__":
    args = parse_args()

    args.models_dir.mkdir(exist_ok=True, parents=True)

    train(**vars(args))
