import configargparse
import argparse

def config_parser():
    parser = configargparse.ArgumentParser()
    
    parser.add_argument("--config", is_config_file=True, help="config file path")
    parser.add_argument("--config_path", type=str, default="configurations/mrnn.json")
    parser.add_argument("--model_path", type=str, default="checkpoints/mrnn")
    parser.add_argument("--model_file", type=str, default="mrnn.pth")
    parser.add_argument("--exp_path", type=str, default="results/mrnn/center_out_random")
    parser.add_argument("--experiment", type=str, default="train_2link_multi")

    return parser