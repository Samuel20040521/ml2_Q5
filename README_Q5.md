Q5 Submission Instructions

Overview
- Q5.py contains both training and inference logic.
- By default, running Q5.py performs inference and writes submission.csv in Kaggle format.
- Training can be enabled with a flag. A run manifest (submission.meta.json) is saved to support reproducibility.

Quick Start
- Generate submission (uses default smoke checkpoint if present):
  - `python style_detection/code/Q5.py`
- Custom checkpoint:
  - `python style_detection/code/Q5.py --model_path /path/to/your.ckpt`

Reproducibility
- Fixed seeds are used and deterministic cuDNN flags are set for both training and inference.
- A run manifest JSON is saved by default next to submission.csv:
  - `submission.meta.json` contains args, environment info, git commit (if available), weights SHA256, dataset file list, and checkpoint metadata.
- Optional tag for the manifest:
  - `--run_tag myexp1`
- Change manifest path:
  - `--run_meta_path runs/myexp1.json`

Automatic Weights Download (optional)
- If `--model_path` is not provided and the default `smoke_ckpt/smoke_epoch1.pt` is missing, set an environment variable to auto-download weights:
  - `export Q5_WEIGHTS_URL="https://your-host/your-weights.pt"`
  - Then run `python style_detection/code/Q5.py`

Training
- Example (transformer + cosine head + GE2E + episode sampling):
  - `python style_detection/code/Q5.py --mode train --conf conf.cfg --device cuda \
     --seq_type transformer --logit_head cosine --ge2e_weight 0.2 --use_episode \
     --episode_players 8 --episode_games 4 --frame_dropout 0.1`
- Checkpoints are saved to `--save_dir` (default: checkpoints/) and include args and meta for later reproducibility.

Key CLI Flags
- Inference:
  - `--model_path` (default: smoke_ckpt/smoke_epoch1.pt)
  - `--query_glob` (default: test_set/query_set/*.sgf)
  - `--candidate_glob` (default: test_set/cand_set/*.sgf)
  - `--submission_path` (default: submission.csv)
  - `--save_run_meta`, `--run_meta_path`, `--run_tag`
- Training:
  - `--train_glob`, `--val_glob` (or `--val_ratio`)
  - `--seq_type` [lstm|transformer], `--logit_head` [linear|cosine]
  - `--ge2e_weight`, `--use_episode`, `--episode_players`, `--episode_games`, `--frame_dropout`

Notes
- The backend module `build.go.style_py` and config `conf.cfg` are used for feature extraction.
- submission.csv format: header `id,label` followed by integer ids per row.

