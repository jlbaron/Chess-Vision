'''
This file takes collected training samples and creates boards and graphics of training
It will take the collected samples and use them to create chess board SVGs
It will return/save the collection of chess boards
It will create a GIF of training with all visualizations
'''
import argparse
import yaml
import chess
import chess.svg
import pandas as pd

piece_vocab = {
    "p": chess.Piece(chess.PAWN, chess.BLACK), "P": chess.Piece(chess.PAWN, chess.WHITE),
    "n": chess.Piece(chess.KNIGHT, chess.BLACK), "N": chess.Piece(chess.KNIGHT, chess.WHITE),
    "b": chess.Piece(chess.BISHOP, chess.BLACK), "B": chess.Piece(chess.BISHOP, chess.WHITE),
    "r": chess.Piece(chess.ROOK, chess.BLACK), "R": chess.Piece(chess.ROOK, chess.WHITE),
    "q": chess.Piece(chess.QUEEN, chess.BLACK), "Q": chess.Piece(chess.QUEEN, chess.WHITE),
    "k": chess.Piece(chess.KING, chess.BLACK), "K": chess.Piece(chess.KING, chess.WHITE)
}

parser = argparse.ArgumentParser(description='Chess-Vision')
parser.add_argument('--config', default='.\\configs\\config_CNN.yaml', help='Path to the configuration file. Default: .\\configs\\config_CNN.yaml')

# take a raw string of space separated values
# turn into piece map for python chess board
def process_raw_input(raw):
    raw = raw.split(" ")
    processed = {}
    # TODO: squares are inverted from data I have so need to reconfigure loop
    for idx, square in enumerate(chess.SQUARES):
        if raw[idx] != "0":
            processed[square] = piece_vocab[raw[idx]]
    return processed

def create_board(piece_map):
    board = chess.BaseBoard()
    board.clear_board()
    board.set_piece_map(pieces=piece_map)
    svg_board = chess.svg.board(board, size=350)
    # TODO: dynamic naming and possibly save image instead of svg
    with open('visualizations\\board_svgs\\board_test.svg', 'w') as f:
        f.write(svg_board)
    print(board)

# TODO: take sequence of images and turn into GIF
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

    df = pd.read_csv("visualizations\\eval_sample_per_epoch.csv")
    for idx in df.index:
        # print(process_raw_input(df['Pred'][idx]))
        print(process_raw_input(df['True'][idx]))
        piece_map = process_raw_input(df['True'][idx])
        create_board(piece_map=piece_map)
        assert(0)

if __name__ == '__main__':
    main()