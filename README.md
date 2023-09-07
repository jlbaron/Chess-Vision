# Chess-Vision
A deep computer vision model to take images of chess boards and return their FEN representation.

data source from https://www.kaggle.com/datasets/koryakinp/chess-positions

Inputs: 400x400 pixel images of digital chess boards
        28 board styles and 32 piece styles for variety
        all positions 5-15 pieces
        train: 80000  test: 20000

process:
        convert labels to encodings
        divide image into squares
        embed and positional encoding
        pass transformer output through linear for vocab scores
        softmax vocab scores and convert to sequence of tokens
        cross-entropy loss 

progress:
        data loading process complete
        transformer embeddings functional
        train loop functional
        working to debug rest of transformer forward pass
                additionally going to add more metrics to better understand training
                then will optimize and train

TODO:
        analyze trained model

Visualizations:
        sample same label from each epoch and see how it improves over time
                choose label from test set, if label in labels then add info to variables (later save to csv)
                also snag the cnn output
        can make a gif of process
                original image -> grayscale -> split -> guesses over time