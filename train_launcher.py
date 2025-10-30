import argparse
from typing import List, Optional

from Q5 import build_argument_parser, run_train


def apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    preset = args.preset
    if preset == "smoke":
        args.mode = "train"
        args.device = args.device or "cpu"
        args.epochs = 1
        args.batch_size = min(args.batch_size, 4)
        args.eval_batch_size = min(args.eval_batch_size, 4)
        args.max_steps = args.max_steps or 5
        args.max_eval_batches = args.max_eval_batches or 2
        args.ge2e_weight = 0.0
        args.seq_type = getattr(args, "seq_type", "lstm")
        args.logit_head = getattr(args, "logit_head", "linear")
        args.use_episode = False
    elif preset == "normal":
        args.mode = "train"
        args.epochs = max(args.epochs, 10)
        args.batch_size = max(args.batch_size, 16)
        args.eval_batch_size = max(args.eval_batch_size, 32)
        args.max_steps = 0
        args.max_eval_batches = 0
        args.seq_type = "transformer"
        args.logit_head = "cosine"
        args.ge2e_weight = max(getattr(args, "ge2e_weight", 0.0), 0.2)
        args.use_episode = True
        args.episode_players = max(getattr(args, "episode_players", 8), 8)
        args.episode_games = max(getattr(args, "episode_games", 4), 4)
        args.frame_dropout = max(getattr(args, "frame_dropout", 0.0), 0.1)
    elif preset == "ultra":
        args.mode = "train"
        args.epochs = max(args.epochs, 30)
        args.batch_size = max(args.batch_size, 32)
        args.eval_batch_size = max(args.eval_batch_size, 64)
        args.lr = min(args.lr, 3e-4)
        args.max_steps = 0
        args.max_eval_batches = 0
        args.seq_type = "transformer"
        args.logit_head = "cosine"
        args.ge2e_weight = max(getattr(args, "ge2e_weight", 0.0), 0.3)
        args.use_episode = True
        args.episode_players = max(getattr(args, "episode_players", 12), 12)
        args.episode_games = max(getattr(args, "episode_games", 4), 4)
        args.frame_dropout = max(getattr(args, "frame_dropout", 0.0), 0.15)
    else:
        raise ValueError(f"Unknown preset: {preset}")
    return args


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_argument_parser()
    parser.set_defaults(mode="train")
    parser.add_argument(
        "--preset",
        choices=["smoke", "normal", "ultra"],
        default="normal",
        help="Training preset controlling dataset size and runtime.",
    )
    args = parser.parse_args(argv)
    args = apply_preset(args)
    if args.mode != "train":
        raise ValueError("train_launcher only supports training mode.")
    run_train(args)


if __name__ == "__main__":
    main()
