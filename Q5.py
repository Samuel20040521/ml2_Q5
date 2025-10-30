import argparse
import csv
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import os
import hashlib
import json
import platform
import subprocess
from datetime import datetime
import shutil

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

from model import GE2ELoss, Model


def parse_kv_config(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            row = raw.split("#", 1)[0].strip()
            if not row or "=" not in row:
                continue
            key, value = row.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def read_player_name(path: Path) -> str:
    with path.open("r", encoding="utf-8") as fh:
        first_line = fh.readline()
    return first_line.strip()


def count_moves(game_text: str) -> int:
    return len(re.findall(r";[BW]\[", game_text))


def extract_games(path: Path) -> List[int]:
    with path.open("r", encoding="utf-8") as fh:
        lines = fh.read()
    body = lines.split("\n", 1)[1] if "\n" in lines else ""
    games: List[int] = []
    depth = 0
    start = None
    for idx, ch in enumerate(body):
        if ch == "(":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and start is not None:
                game_str = body[start : idx + 1]
                games.append(count_moves(game_str))
                start = None
    return games


@dataclass
class GameRecord:
    player_id: str
    backend_index: int
    label_index: int
    game_id: int
    total_moves: int
    path: Path


class StyleBackend:
    def __init__(self, conf_path: Path, module_path: str, sgf_paths: Sequence[Path]) -> None:
        self.conf_path = conf_path
        self.cfg = parse_kv_config(conf_path)
        self.module = import_module(module_path)
        self.module.load_config_file(str(conf_path))
        self.loader = self.module.DataLoader(str(conf_path))

        self.player_files: Dict[str, List[Path]] = defaultdict(list)
        for sgf_path in sgf_paths:
            self.loader.load_data_from_file(str(sgf_path))
            pid = read_player_name(sgf_path)
            self.player_files[pid].append(sgf_path)

        self.player_ids = sorted(self.player_files.keys())
        self.player_to_backend = {pid: idx for idx, pid in enumerate(self.player_ids)}

        self.n_frames = int(self.module.get_n_frames())
        self.in_ch = int(self.module.get_nn_num_input_channels())
        self.board_h = int(self.module.get_nn_input_channel_height())
        self.board_w = int(self.module.get_nn_input_channel_width())
        move_step = int(self.cfg.get("move_step_to_choose", "1"))
        self.backend_window = self.n_frames * move_step

    def feature_vector(self, player_backend_idx: int, game_id: int, start: int) -> torch.Tensor:
        feat = self.loader.get_feature_and_label(player_backend_idx, game_id, start, False)
        tensor = torch.tensor(feat, dtype=torch.float32)
        return tensor.view(self.n_frames, self.in_ch, self.board_h, self.board_w)


class GameDataset(Dataset):
    def __init__(
        self,
        backend: StyleBackend,
        records: Sequence[GameRecord],
        random_start: bool = False,
        center_start: bool = False,
    ) -> None:
        self.backend = backend
        self.records = list(records)
        self.random_start = random_start
        self.center_start = center_start

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        max_start = max(record.total_moves - self.backend.backend_window, 0)
        if max_start <= 0:
            start = 0
        elif self.random_start:
            start = random.randint(0, max_start)
        elif self.center_start:
            start = max_start // 2
        else:
            start = 0

        frames = self.backend.feature_vector(record.backend_index, record.game_id, start)
        mask = (frames.abs().sum(dim=(1, 2, 3)) > 1e-6).float()
        return frames, mask, record.label_index, record.player_id, record.game_id


class BalancedEpisodeBatchSampler(Sampler[List[int]]):
    """Yields batches with P players × U games per player for GE2E-friendly training."""

    def __init__(self, labels: Sequence[int], players_per_batch: int, games_per_player: int, steps_per_epoch: int = 0) -> None:
        self.labels = list(labels)
        self.players_per_batch = int(players_per_batch)
        self.games_per_player = int(games_per_player)

        mapping: Dict[int, List[int]] = defaultdict(list)
        for idx, y in enumerate(self.labels):
            mapping[int(y)].append(idx)
        self.by_label = {k: v for k, v in mapping.items() if v}

        total = len(self.labels)
        denom = max(1, self.players_per_batch * self.games_per_player)
        self._len = int(total // denom) if steps_per_epoch <= 0 else int(steps_per_epoch)
        self._len = max(1, self._len)

    def __len__(self) -> int:
        return self._len

    def __iter__(self):
        label_ids = list(self.by_label.keys())
        for _ in range(self._len):
            if self.players_per_batch >= len(label_ids):
                chosen = random.sample(label_ids, k=len(label_ids))
            else:
                chosen = random.sample(label_ids, k=self.players_per_batch)
            batch: List[int] = []
            for y in chosen:
                pool = self.by_label[y]
                if len(pool) >= self.games_per_player:
                    sel = random.sample(pool, k=self.games_per_player)
                else:
                    sel = [random.choice(pool) for _ in range(self.games_per_player)]
                batch.extend(sel)
            yield batch


def build_records(
    backend: StyleBackend,
    sgf_paths: Sequence[Path],
    label_map: Optional[Dict[str, int]] = None,
) -> List[GameRecord]:
    records: List[GameRecord] = []
    for path in sgf_paths:
        player_id = read_player_name(path)
        if player_id not in backend.player_to_backend:
            continue
        if label_map is not None and player_id not in label_map:
            continue
        backend_idx = backend.player_to_backend[player_id]
        label_idx = label_map[player_id] if label_map is not None else backend_idx
        games = extract_games(path)
        for gid, moves in enumerate(games):
            records.append(GameRecord(player_id, backend_idx, label_idx, gid, moves, path))
    return records


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    train_paths = sorted(Path(path) for path in glob_paths(args.train_glob))
    if not train_paths:
        raise FileNotFoundError(f"No training SGF files found for pattern: {args.train_glob}")

    if args.val_glob:
        val_paths = sorted(Path(path) for path in glob_paths(args.val_glob))
    else:
        split_idx = int(len(train_paths) * (1.0 - args.val_ratio))
        split_idx = max(1, min(split_idx, len(train_paths) - 1))
        val_paths = train_paths[split_idx:]
        train_paths = train_paths[:split_idx]

    train_backend = StyleBackend(Path(args.conf), args.module, train_paths)
    label_map = {pid: idx for idx, pid in enumerate(train_backend.player_ids)}
    train_records = build_records(train_backend, train_paths, label_map)

    val_backend: Optional[StyleBackend] = None
    val_records: List[GameRecord] = []
    if val_paths:
        val_backend = StyleBackend(Path(args.conf), args.module, val_paths)
        val_records = build_records(val_backend, val_paths, label_map)

    if not train_records:
        raise RuntimeError("Training record list is empty. Check SGF parsing or configuration.")

    model = Model(
        in_ch=train_backend.in_ch,
        d_move=args.d_move,
        d_seq=args.d_seq,
        d_vec=args.d_vec,
        cnn_depth=args.cnn_depth,
        seq_type=args.seq_type,
        num_classes=len(label_map),
        dropout=args.dropout,
        per_move_dropout=args.per_move_dropout,
        logit_head=args.logit_head,
        cosine_scale=args.cosine_scale,
        cosine_margin=args.cosine_margin,
    )

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    ge2e = GE2ELoss().to(device) if args.ge2e_weight > 0.0 else None

    global_step = 0
    start_epoch = 1
    start_step_in_epoch = 0
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = max(ckpt.get("epoch", 1), 1)
        has_epoch_step = "epoch_step" in ckpt
        start_step_in_epoch = max(ckpt.get("epoch_step", 0), 0) if has_epoch_step else 0
        if not has_epoch_step:
            start_epoch = min(start_epoch + 1, args.epochs + 1)
        global_step = max(ckpt.get("global_step", 0), 0)

    train_ds = GameDataset(train_backend, train_records, random_start=True)
    if getattr(args, "use_episode", False):
        sampler = BalancedEpisodeBatchSampler(
            [rec.label_index for rec in train_ds.records],
            players_per_batch=args.episode_players,
            games_per_player=args.episode_games,
            steps_per_epoch=args.steps_per_epoch,
        )
        train_loader = DataLoader(
            train_ds,
            batch_sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )

    val_loader: Optional[DataLoader] = None
    if val_backend and val_records:
        val_loader = DataLoader(
            GameDataset(val_backend, val_records, random_start=False, center_start=True),
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_pattern = f"{args.run_name}_*.pt"
    saved_checkpoints: List[Path] = sorted(save_dir.glob(checkpoint_pattern))
    if args.keep_last and len(saved_checkpoints) > args.keep_last:
        for old_path in saved_checkpoints[:-args.keep_last]:
            try:
                old_path.unlink()
            except FileNotFoundError:
                pass
        saved_checkpoints = saved_checkpoints[-args.keep_last :]

    def register_checkpoint(path: Path) -> None:
        saved_checkpoints.append(path)
        if args.keep_last and len(saved_checkpoints) > args.keep_last:
            old_path = saved_checkpoints.pop(0)
            if old_path != path:
                try:
                    old_path.unlink()
                except FileNotFoundError:
                    pass

    def save_checkpoint(tag: str, epoch_num: int, epoch_step: int) -> None:
        payload = {
            "epoch": epoch_num,
            "epoch_step": epoch_step,
            "global_step": global_step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "player_ids": train_backend.player_ids,
            "meta": {
                "in_ch": train_backend.in_ch,
                "num_classes": len(label_map),
                "d_move": args.d_move,
                "d_seq": args.d_seq,
                "d_vec": args.d_vec,
                "cnn_depth": args.cnn_depth,
                "seq_type": args.seq_type,
                "dropout": args.dropout,
                "per_move_dropout": args.per_move_dropout,
                "logit_head": args.logit_head,
                "cosine_scale": args.cosine_scale,
                "cosine_margin": args.cosine_margin,
            },
            "args": vars(args),
        }
        save_path = save_dir / f"{args.run_name}_{tag}.pt"
        torch.save(payload, save_path)
        register_checkpoint(save_path)

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_step_offset = start_step_in_epoch if epoch == start_epoch else 0
        total_batches = len(train_loader)
        if epoch_step_offset >= total_batches:
            start_step_in_epoch = 0
            continue

        model.train()
        running_loss = 0.0
        running_acc = 0.0
        steps = 0
        pbar = tqdm(total=total_batches, desc=f"train {epoch}")
        if epoch_step_offset:
            pbar.update(epoch_step_offset)

        data_iter = iter(train_loader)
        for _ in range(epoch_step_offset):
            try:
                next(data_iter)
            except StopIteration:
                data_iter = iter([])
                break

        last_epoch_step = epoch_step_offset
        for batch_idx, batch in enumerate(data_iter, start=epoch_step_offset + 1):
            frames, mask, labels, _, _ = batch
            last_epoch_step = batch_idx

            frames = frames.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            # Temporal augmentation: randomly drop a fraction of frames during training
            if getattr(args, "frame_dropout", 0.0) > 0.0 and model.training:
                drop = (torch.rand_like(mask) < args.frame_dropout)
                valid = (mask > 0.5) & (~drop)
                mask = valid.float()
                frames = frames * mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                vectors, logits = model(frames, mask, labels=labels)
                loss = 0.0
                if logits is not None:
                    loss_rank = F.cross_entropy(logits, labels)
                    loss = loss + loss_rank
                    pred = logits.argmax(dim=-1)
                    running_acc += (pred == labels).float().mean().item()
                if ge2e is not None:
                    loss = loss + args.ge2e_weight * ge2e(vectors, labels)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            steps += 1
            global_step += 1
            avg_loss = running_loss / steps
            avg_acc = running_acc / steps if steps > 0 else 0.0
            pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.4f}")
            pbar.update(1)

            if args.save_interval and global_step % args.save_interval == 0:
                save_checkpoint(f"step{global_step}", epoch, last_epoch_step)

            if args.max_steps and steps >= args.max_steps:
                break

        pbar.close()
        if val_loader:
            evaluate(model, val_loader, device, epoch, args.max_eval_batches)

        save_checkpoint(f"epoch{epoch}", epoch, last_epoch_step)
        start_step_in_epoch = 0


def evaluate(model: Model, loader: DataLoader, device: torch.device, epoch: int, max_batches: int = 0) -> None:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    steps = 0
    with torch.no_grad():
        for frames, mask, labels, _, _ in tqdm(loader, desc=f"valid {epoch}", leave=False):
            frames = frames.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            vectors, logits = model(frames, mask)
            loss = 0.0
            if logits is not None:
                loss = loss + F.cross_entropy(logits, labels)
                pred = logits.argmax(dim=-1)
                total_acc += (pred == labels).float().mean().item()
            total_loss += loss.item()
            steps += 1
            if max_batches and steps >= max_batches:
                break
    if steps == 0:
        return
    print(f"[epoch {epoch}] val_loss={total_loss/steps:.4f} val_acc={total_acc/steps:.4f}")


def glob_paths(pattern: str) -> List[str]:
    return [str(p) for p in Path().glob(pattern)]


def run_infer(args: argparse.Namespace) -> None:
    # Reproducibility knobs
    set_seed(args.seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    model_path = resolve_model_path(args)
    checkpoint = torch.load(model_path, map_location="cpu")
    meta = checkpoint.get("meta", {})

    model = Model(
        in_ch=meta.get("in_ch", 18),
        d_move=meta.get("d_move", args.d_move),
        d_seq=meta.get("d_seq", args.d_seq),
        d_vec=meta.get("d_vec", args.d_vec),
        cnn_depth=meta.get("cnn_depth", args.cnn_depth),
        seq_type=meta.get("seq_type", args.seq_type),
        num_classes=meta.get("num_classes"),
        dropout=meta.get("dropout", args.dropout),
        per_move_dropout=meta.get("per_move_dropout", args.per_move_dropout),
        logit_head=meta.get("logit_head", getattr(args, "logit_head", "cosine")),
        cosine_scale=meta.get("cosine_scale", getattr(args, "cosine_scale", 16.0)),
        cosine_margin=meta.get("cosine_margin", getattr(args, "cosine_margin", 0.0)),
    )
    try:
        ret = model.load_state_dict(checkpoint["model"], strict=False)
        miss = getattr(ret, 'missing_keys', [])
        unexp = getattr(ret, 'unexpected_keys', [])
        if miss or unexp:
            print(f"state_dict loaded with mismatches; missing={len(miss)} unexpected={len(unexp)}")
    except Exception as e:
        raise RuntimeError(f"Failed to load model state_dict from {model_path}: {e}")

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    model.eval()

    query_paths = sorted(Path(path) for path in glob_paths(args.query_glob))
    cand_paths = sorted(Path(path) for path in glob_paths(args.candidate_glob))

    query_backend = StyleBackend(Path(args.conf), args.module, query_paths)
    cand_backend = StyleBackend(Path(args.conf), args.module, cand_paths)

    query_ds = GameDataset(query_backend, build_records(query_backend, query_paths), random_start=False, center_start=True)
    cand_ds = GameDataset(cand_backend, build_records(cand_backend, cand_paths), random_start=False, center_start=True)

    query_vecs = extract_player_embeddings(model, query_ds, device, args.eval_batch_size)
    cand_vecs = extract_player_embeddings(model, cand_ds, device, args.eval_batch_size)

    cand_ids = sorted(cand_vecs.keys())
    cand_matrix = torch.stack([cand_vecs[cid] for cid in cand_ids], dim=0)
    cand_matrix = F.normalize(cand_matrix, dim=-1)

    submission_rows: List[Tuple[int, int]] = []
    for query_id, qvec in query_vecs.items():
        qvec = F.normalize(qvec, dim=-1)
        sims = torch.mv(cand_matrix, qvec)
        best_idx = int(torch.argmax(sims))
        cand_name = cand_ids[best_idx]
        submission_rows.append((player_name_to_int(query_id), player_name_to_int(cand_name)))

    out_path = Path(args.submission_path)
    write_submission(submission_rows, out_path)

    if args.save_run_meta:
        save_run_manifest(
            args=args,
            model_path=str(model_path),
            checkpoint=checkpoint,
            query_paths=[str(p) for p in query_paths],
            cand_paths=[str(p) for p in cand_paths],
            cand_ids=cand_ids,
            submission_path=str(out_path),
        )


def file_sha256(path: str) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def resolve_model_path(args: argparse.Namespace) -> Path:
    # If explicit path exists, use it
    if args.model_path and Path(args.model_path).exists():
        return Path(args.model_path)
    # Try default smoke checkpoint
    default_smoke = Path("smoke_ckpt/smoke_epoch1.pt")
    if default_smoke.exists():
        return default_smoke
    # Try environment-provided URL (auto-download)
    url = os.environ.get("Q5_WEIGHTS_URL", "")
    if url.startswith("http://") or url.startswith("https://"):
        target = Path("weights_auto.pt")
        if not target.exists():
            try_download(url, target)
        if target.exists():
            return target
    raise FileNotFoundError(
        "No valid model weights found. Provide --model_path or set Q5_WEIGHTS_URL, or include smoke_ckpt/smoke_epoch1.pt"
    )


def try_download(url: str, target: Path) -> None:
    try:
        import urllib.request

        print(f"downloading weights from {url} -> {target}")
        urllib.request.urlretrieve(url, str(target))
        return
    except Exception as e:
        print(f"urllib download failed: {e}")
    # Fallback to gdown if available (for Google Drive)
    try:
        import gdown  # type: ignore

        gdown.download(url, str(target), quiet=False)
        return
    except Exception as e:
        print(f"gdown download failed: {e}")
    # Fallback to curl or wget if available
    try:
        if shutil.which("curl"):
            subprocess.run(["curl", "-L", url, "-o", str(target)], check=True)
            return
        if shutil.which("wget"):
            subprocess.run(["wget", "-O", str(target), url], check=True)
            return
    except Exception as e:
        print(f"curl/wget download failed: {e}")


def save_run_manifest(
    args: argparse.Namespace,
    model_path: str,
    checkpoint: Dict,
    query_paths: List[str],
    cand_paths: List[str],
    cand_ids: List[str],
    submission_path: str,
) -> None:
    meta = checkpoint.get("meta", {})
    manifest = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "run_tag": getattr(args, "run_tag", ""),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "model_path": model_path,
        "weights_sha256": file_sha256(model_path),
        "checkpoint_meta": meta,
        "player_ids_in_ckpt": len(checkpoint.get("player_ids", [])) if isinstance(checkpoint.get("player_ids", []), list) else None,
        "conf_path": args.conf,
        "conf_sha256": file_sha256(args.conf) if os.path.exists(args.conf) else None,
        "backend_module": args.module,
        "query_glob": args.query_glob,
        "candidate_glob": args.candidate_glob,
        "query_files": query_paths,
        "candidate_files": cand_paths,
        "candidate_id_list": cand_ids,
        "submission_path": submission_path,
        "env": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "platform": platform.platform(),
        },
        "git": {
            "commit": _safe_git(["git", "rev-parse", "HEAD"]),
            "status": _safe_git(["git", "status", "--porcelain"]),
        },
    }
    try:
        with open(args.run_meta_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print(f"run manifest saved to {args.run_meta_path}")
    except Exception as e:
        print(f"failed to save run manifest: {e}")


def _safe_git(cmd: List[str]) -> Optional[str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip()
        return out
    except Exception:
        return None


def extract_player_embeddings(
    model: Model,
    dataset: GameDataset,
    device: torch.device,
    batch_size: int,
) -> Dict[str, torch.Tensor]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    storage: Dict[str, List[torch.Tensor]] = defaultdict(list)
    with torch.no_grad():
        for frames, mask, _, player_ids, _ in tqdm(loader, desc="encode", leave=False):
            frames = frames.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            vectors, _ = model(frames, mask)
            vectors = vectors.cpu()
            for pid, vec in zip(player_ids, vectors):
                storage[pid].append(vec)
    return {pid: torch.stack(vecs, dim=0).mean(dim=0) for pid, vecs in storage.items()}


def player_name_to_int(name: str) -> int:
    digits = "".join(ch for ch in name if ch.isdigit())
    return int(digits) if digits else -1


def write_submission(rows: Iterable[Tuple[int, int]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "label"])
        for row in rows:
            writer.writerow(row)
    print(f"submission saved to {path}")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Q5 style detection training and inference script.")
    # Default to inference to satisfy submission requirement
    parser.add_argument("--mode", choices=["train", "infer"], default="infer")
    parser.add_argument("--conf", type=str, default="conf.cfg")
    parser.add_argument("--module", type=str, default="build.go.style_py")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    # Training configuration
    parser.add_argument("--train_glob", type=str, default="train_set/*.sgf")
    parser.add_argument("--val_glob", type=str, default=None)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--run_name", type=str, default="style_encoder")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument(
        "--save_interval",
        type=int,
        default=0,
        help="Number of optimizer steps between intra-epoch checkpoints (0 disables).",
    )
    parser.add_argument(
        "--keep_last",
        type=int,
        default=0,
        help="Keep only the most recent N checkpoints (0 keeps all).",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--ge2e_weight", type=float, default=0.0)
    parser.add_argument("--use_episode", action="store_true")
    parser.add_argument("--episode_players", type=int, default=8)
    parser.add_argument("--episode_games", type=int, default=4)
    parser.add_argument("--steps_per_epoch", type=int, default=0, help="Override number of training batches per epoch.")
    parser.add_argument("--frame_dropout", type=float, default=0.0, help="Probability of dropping a frame in training.")
    parser.add_argument("--max_steps", type=int, default=0, help="Limit number of train batches per epoch (0 = full epoch).")
    parser.add_argument("--max_eval_batches", type=int, default=0, help="Limit evaluation batches (0 = full validation set).")

    # Model configuration
    parser.add_argument("--d_move", type=int, default=128)
    parser.add_argument("--d_seq", type=int, default=256)
    parser.add_argument("--d_vec", type=int, default=128)
    parser.add_argument("--cnn_depth", type=int, default=4)
    parser.add_argument("--seq_type", choices=["lstm", "transformer"], default="lstm")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--per_move_dropout", type=float, default=0.0)
    parser.add_argument("--logit_head", choices=["linear", "cosine"], default="cosine")
    parser.add_argument("--cosine_scale", type=float, default=16.0)
    parser.add_argument("--cosine_margin", type=float, default=0.0)

    # Inference configuration
    parser.add_argument("--model_path", type=str, default="smoke_ckpt/smoke_epoch1.pt")
    parser.add_argument("--query_glob", type=str, default="test_set/query_set/*.sgf")
    parser.add_argument("--candidate_glob", type=str, default="test_set/cand_set/*.sgf")
    parser.add_argument("--submission_path", type=str, default="submission.csv")
    parser.add_argument("--save_run_meta", action="store_true", default=True, help="Save reproducibility manifest JSON next to submission.")
    parser.add_argument("--run_meta_path", type=str, default="submission.meta.json", help="Where to save the reproducibility manifest.")
    parser.add_argument("--run_tag", type=str, default="", help="Optional tag to include in run metadata.")

    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    if args.mode == "train":
        run_train(args)
    else:
        run_infer(args)


if __name__ == "__main__":
    main()
