Arnav : In file B-cos_computer_vision i have tried to modified version of VGG16light with a B-cos layer added at the end of the forward method. This layer normalizes the output using the L2 norm, aligning with the B-cos network approach.
Updated the training and evaluation process accordingly. Used a cosine similarity-based loss function and include cosine accuracy as an evaluation metric.
This updated code implements a cosine similarity-based loss function (cosine_loss) and incorporates cosine accuracy calculation (cosine_accuracy) in the evaluation process. The training loop now uses cosine_loss for optimization, and the validation function calculates both validation loss and cosine accuracy.
#Feel free to make any changes
