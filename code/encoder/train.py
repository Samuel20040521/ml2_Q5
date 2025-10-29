from encoder.model import Encoder
from profiler import Profiler
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset
from torch.utils.data import get_worker_info
import numpy as np
import sys
import torch
import glob
import time
import random


class MinizeroDataset(IterableDataset):
    def __init__(self, files, conf_file, game_type):

        # Fully implemented dataloader that connects to the C++ side via pybind
        # and retrieves complete features from the backend.

        _temps = __import__(f'build.{game_type}', globals(), locals(), ['style_py'], 0)
        style_py = _temps.style_py
        style_py.load_config_file(conf_file)

        self.data_loader = style_py.DataLoader(conf_file)
        self.player_choosed = 0
        self.games_per_player = style_py.get_games_per_player()
        self.players_per_batch = style_py.get_players_per_batch()
        self.n_frames = style_py.get_n_frames()
        self.training_feature_mode = 2
        self.input_channel_feature = style_py.get_nn_num_input_channels()
        self.board_size_h = style_py.get_nn_input_channel_height()
        self.board_size_w = style_py.get_nn_input_channel_width()
        for file_name in files:
            self.data_loader.load_data_from_file(file_name)
        self.random_player = [i for i in range(self.data_loader.get_num_of_player())]
        random.shuffle(self.random_player)

    def __iter__(self):

        # Each time calling "inputs = next(data_loader_iterator)"
        # will directly get one complete batch of data.
        # That is, style_py.get_players_per_batch() players,
        # each player has style_py.get_games_per_player() games,
        # and each game takes style_py.get_n_frames() moves.
        # If self.training_feature_mode == 1,
        # it takes consecutive n_frames moves.
        # If self.training_feature_mode == 2,
        # it takes random non-repeating n_frames moves.
        # Students can directly use mode 2, but mode 1 is also allowed.

        while True:
            if self.player_choosed == self.players_per_batch:
                self.player_choosed = 0
                random.shuffle(self.random_player)
            if self.training_feature_mode == 1:
                features = self.data_loader.get_feature_and_label(self.random_player[self.player_choosed], 1, 0, 1)
            elif self.training_feature_mode == 2:
                features = self.data_loader.get_random_feature_and_label(self.random_player[self.player_choosed], 1, 0, 1)
            yield torch.FloatTensor(features).view(1 * self.games_per_player, self.n_frames, self.input_channel_feature, self.board_size_h, self.board_size_w)
            self.player_choosed = self.player_choosed + 1


def sync(device: torch.device):
    # For correct profiling (cuda operations are async)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train(run_id: str, game_type: str, data_dir: str, validate_data_dir: str, models_dir: Path,
          save_every: int, backup_every: int, validate_every: int, force_restart: bool, conf_file: str
          ):
    # This function shows a simplified example of the overall training flow.
    # It demonstrates how to create the dataset, data loader, model, and
    # perform a forward pass to obtain outputs from the neural network.
    # Students are expected to design their own model architecture, loss
    # computation, and optimization steps.

    # Import the compiled pybind library for the given game type
    _temps = __import__(f'build.{game_type}', globals(), locals(), ['style_py'], 0)
    style_py = _temps.style_py
    style_py.load_config_file(conf_file)

    # Load all SGF files from the training set directory
    sgf_location = "./train_set"
    all_file_list = glob.glob(sgf_location)

    # Create dataset and dataloader
    dataset = MinizeroDataset(all_file_list, conf_file, game_type)
    data_loader = DataLoader(dataset, batch_size=style_py.get_players_per_batch(), num_workers=8)
    data_loader_iterator = iter(data_loader)

    # Setup device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model and optimizer
    model = Encoder(device, conf_file, game_type)

    multi_gpu = False
    if torch.cuda.device_count() > 1:
        multi_gpu = True
        model = torch.nn.DataParallel(model)
    model.to(device)

    model.train()

    # Get board dimensions from config
    board_size_h = style_py.get_nn_input_channel_height()
    board_size_w = style_py.get_nn_input_channel_width()

    # Main training loop
    # This part only demonstrates how to retrieve the input batch
    # and perform the forward pass.
    # The full loss computation, backpropagation, optimizer steps,
    # checkpoint saving, and validation are left to be implemented.
    while step < style_py.get_training_step():
        step = step + 1

        # Retrieve one batch of input features
        inputs = next(data_loader_iterator)

        # Reshape input to the expected tensor shape:
        # [players_per_batch * games_per_player,
        #  n_frames, num_input_channels, board_H, board_W]
        inputs = inputs.reshape(
            style_py.get_players_per_batch() * style_py.get_games_per_player(),
            style_py.get_n_frames(),
            style_py.get_nn_num_input_channels(),
            board_size_h,
            board_size_w
        )

        # Synchronize device before forward pass (optional utility)
        sync(device)

        # Forward pass through the model
        output = model(inputs)

        # Example ends here:
        # Students should implement their own loss, optimizer, and update logic below.

        


