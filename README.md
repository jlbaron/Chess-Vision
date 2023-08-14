# Chess-Vision
A deep computer vision model to take images of chess boards and return their FEN representation.

data source from https://www.kaggle.com/datasets/koryakinp/chess-positions

Inputs: 400x400 pixel images of digital chess boards
        28 board styles and 32 piece styles for variety
        all positions 5-15 pieces
        train: 80000  test: 20000

process:
convert labels to encodings
downsample each image (carefully)
divide into squares for input to transformer
append cls token and retrieve from output
cross-entropy loss 