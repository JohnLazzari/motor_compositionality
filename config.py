import configargparse
import argparse

def config_parser():
    parser = configargparse.ArgumentParser()
    
    parser.add_argument("--config", is_config_file=True, help="config file path")
    parser.add_argument("--model_name", type=str, default="rnn256_softplus")
    parser.add_argument("--experiment", type=str, default="train_2link_multi")

    return parser