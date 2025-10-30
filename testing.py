<<<<<<< ours
import argparse 
import glob 
from numpy import dot 
import numpy as np 
import torch 
from encoder.model import Encoder 
import sys
import torch.nn.functional as F
sys.path.append("../style_detection/")

def parse_args() : 
    # parse command-line arguments for game type and config file
    parser = argparse.ArgumentParser()
    parser.add_argument("game_type", type=str, help="Name for game type.")
    parser.add_argument("-c", "--conf_file", type=str, default="./conf.cfg", help="Configuration file path")
    args = parser.parse_args()
    return args


class style_detection:
    """
    Example class showing a possible structure for inference/testing.
    This is NOT the final implementation — students should modify
    and complete it according to their own task design.
    """

    def __init__(self, conf_file, game_type):
        # Example: initialize configuration and DataLoader from C++ backend
        self.game_type = game_type
        self.conf_file = conf_file
        self.data_loader = style_py.DataLoader(conf_file)
        self.n_frames = style_py.get_n_frames()
        self.board_size_h = style_py.get_nn_input_channel_height()
        self.board_size_w = style_py.get_nn_input_channel_width()
        self.input_channel_feature = style_py.get_nn_num_input_channels()

    def read_sgf(self, sgf_dir):
        # load all SGF files
        # This is only a demonstration of how data can be read.
        print('start read sgf')
        print("[{}] ".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

        for i in range(600):
            self.data_loader.load_data_from_file(sgf_dir + "player" + str(i + 1) + ".sgf")

        print('end read sgf')
        print("[{}] ".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

    def load_model(self, model_path_):
        # Example: load a trained model (students should modify as needed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Encoder(self.device, self.conf_file, self.game_type)
        multi_gpu = False
        if torch.cuda.device_count() > 1:
            multi_gpu = True
            self.model = torch.nn.DataParallel(self.model)

        self.model.to(self.device)
        self.model.eval()  # set model to eval mode
        # model_path_ is expected to be provided externally (e.g., trained model path)

    def inference(self, data_loader):
        # Example inference loop: process all players and games
        # This code is only a template for how inference might be structured.
        for _ in range(600):
            for game_id in range(100):
                if self.testing_feature_mode == 1:
                    features = data_loader.get_feature_and_label( _ , game_id, self.start, 0)
                elif self.testing_feature_mode == 2:
                    features = data_loader.get_random_feature_and_label( _ , game_id, self.start, 0)

                # Convert features to tensor and move to GPU
                features = torch.FloatTensor(features).view(
                    1, self.n_frames, self.input_channel_feature, self.board_size_h, self.board_size_w
                )
                features = features.to('cuda')

                # Forward pass (no loss or output processing shown)
                output = self.model(features)

    def testing(self):
        # Example testing procedure:
        # Load query set → run inference → clear → load candidate set → run inference again
        # Students should extend this to compute style similarity, predictions, etc.
        with torch.no_grad():
            # Load and test query set
            self.read_sgf(sgf_dir="./test_set/query_set")
            self.load_model(model_path_="???")  # model path should be specified by the user
            self.inference(self.data_loader)

            # Clear SGF data after inference
            self.data_loader.Clear_Sgf()  # really important

            # Load and test candidate set
            self.read_sgf(sgf_dir="./test_set/cand_set")
            self.inference(self.data_loader)

            # Clear again after completion
            self.data_loader.Clear_Sgf()


if __name__ == "__main__":
    # Example entry point for running the script
    # Parses command-line args, initializes style_py backend, and runs testing()
    args = parse_args()
    _temps = __import__(f'build.{args.game_type}', globals(), locals(), ['style_py'], 0)
    style_py = _temps.style_py

    style_py.load_config_file(args.conf_file)
    test = style_detection(args.conf_file, args.game_type)
    test.testing()
=======
import argparse
from Q5 import build_argument_parser, run_infer


def main() -> None:
    parser = build_argument_parser()
    parser.set_defaults(mode="infer")
    args = parser.parse_args()
    run_infer(args)


if __name__ == "__main__":
    main()
>>>>>>> theirs
