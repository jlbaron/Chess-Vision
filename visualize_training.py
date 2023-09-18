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
from cairosvg import svg2png

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

# take space delimited label and convert to FEN
# Ex: 0 b 0 B 0 Q r 0 0 0 0 0 0 0 0 p 0 0 0 0 0 0 r 0 0 0 P 0 0 0 0 0 0 0 0 0 R k 0 0 0 K 0 0 0 0 0 0 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 
# becomes 1b1B1Qr1/7p/6r1/2P5/4Rk2/1K6/4B3/8
def convert_label_to_fen(label):
    fen = ""
    return fen

def create_board_squares():
    row = []
    rows = []
    ctr = 0
    for i in chess.SQUARES:
        ctr += 1
        row.append(i)
        if ctr % 8 == 0:
            rows.append(row)
            row = []
            ctr = 0
    rows.reverse()
    board_squares = []
    for i in rows:
        for j in i:
            board_squares.append(j)
    return board_squares

# take a raw string of space separated values
# turn into piece map for python chess board
def process_raw_input(raw, squares):
    raw = raw.split(" ")
    processed = {}
    # TODO: squares are inverted from data I have so need to reconfigure loop
    for idx, square in enumerate(squares):
        if raw[idx] != "0":
            processed[square] = piece_vocab[raw[idx]]
    return processed

def create_board_svg(piece_map, filename):
    board = chess.BaseBoard()
    board.clear_board()
    board.set_piece_map(pieces=piece_map)
    svg_board = chess.svg.board(board, size=350)
    svg2png(bytestring=svg_board, write_to='visualizations\\training_progress_boards\\'+ filename + '.png')


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

    squares = create_board_squares()

    df = pd.read_csv("visualizations\\eval_sample_per_epoch.csv")
    # store labels counts as dict
    # sample names include label (true/pred) and which count it is at
    labels = df['True'].unique
    labels_dict = {}
    for i in labels:
        fen = convert_label_to_fen(i)
        labels_dict[fen] = 0
    assert(0) #need to test label to FEN
    for idx in df.index:
        # TRUE
        piece_map = process_raw_input(df['True'][idx], squares)
        create_board_svg(piece_map=piece_map, filename=convert_label_to_fen(df['True'][idx]))
        # PRED
        filename = convert_label_to_fen(df['True'][idx])
        piece_map = process_raw_input(df['Pred'][idx], squares)
        create_board_svg(piece_map=piece_map, filename=f'filename({labels_dict[filename]})')
        labels_dict[filename] += 1

if __name__ == '__main__':
    main()