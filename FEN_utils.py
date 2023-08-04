'''
data uses Forsyth-Edwards Notation with dashes instead of slashes
this notation comes with 6 parts
1) pieces by rank (separated with -)
2) active color
3) castling availability
4) en passant target square
5) halfmove clock
6) fullmove number

to make the data easier to classify, I only include the pieces by rank
for example here is the opening (uppercase is white): 
rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR
and after e4:
rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR

I am not sure if any model can look at a random board and say:
1) what the last move was
2) how many moves have been played
both of which are needed for 2-6 in the full FEN
'''
fen_to_int = {
    "P" : 1,
    "N" : 2,
    "B" : 3,
    "R" : 4,
    "Q" : 5,
    "K" : 6,
    "p" : -1,
    "n" : -2,
    "b" : -3,
    "r" : -4,
    "q" : -5,
    "k" : -6
}

# convert fen to number for classification
def encode_fen(input_fen):
    # all space numbers are 0s
    # pieces get a numerical value
    # white: 1P, 2N, 3B, 4R, 5Q, 6K
    # black: -1p, -2n, -3b, -4r, -5q, -6k
    output_fen = ""
    return output_fen


# convert transformer output to fen to pretty printing
def decode_fen(output_fen):
    # encoded: -4-2-3-5-6-3-2-4/-1-1-1-1-1-1-1-1/0s/0s/0s/0s/11111111/42356324
    # decoded: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR
    input_fen = ""
    return input_fen