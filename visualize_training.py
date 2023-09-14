'''
This file takes collected training samples and creates boards and graphics of training
It will take the collected samples and use them to create chess board SVGs
It will return/save the collection of chess boards
It will create a GIF of training with all visualizations
'''
import argparse
import yaml
# import chess



parser = argparse.ArgumentParser(description='Chess-Vision')
parser.add_argument('--config', default='.\\configs\\config_CNN.yaml', help='Path to the configuration file. Default: .\\configs\\config_CNN.yaml')


def create_board():
    pass

def create_gif():
    pass

'''
Usage: python inference.py [OPTIONS]

Options:
  --config CONFIG_PATH  Path to the configuration file.
                        Default: .\\configs\\config_CNN.yaml
'''
def main():
    #args from yaml file
    global args
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    # set args object
    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)
        
    