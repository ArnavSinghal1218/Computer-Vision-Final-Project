Arnav : In file B-cos_computer_vision i have tried to modified version of VGG16light with a B-cos layer added at the end of the forward method. This layer normalizes the output using the L2 norm, aligning with the B-cos network approach.
Updated the training and evaluation process accordingly. Used a cosine similarity-based loss function and include cosine accuracy as an evaluation metric.
This updated code implements a cosine similarity-based loss function (cosine_loss) and incorporates cosine accuracy calculation (cosine_accuracy) in the evaluation process. The training loop now uses cosine_loss for optimization, and the validation function calculates both validation loss and cosine accuracy.
#Feel free to make any changes.

(25/09/2024) Mukul :- I tried to update the architecture in which -
1) I Reduce the number of layers or neurons in the current model (VGG16lightBcos). This can lead to faster training and inference times i believe.
2) And changed the activation function from Relu to leakyRelu (Not sure but it might work as i was reading on internet about this)
If this does not work i have also added the code for transfer learning where we use a pre-trained model (MobileNetV2) which offers a good balance between speed and accuracy as stated on internet. We can fine tune this model on our dataset and check.


(26/09/2024) Arnav - More Suggestions for Architecture update:-
1. We can try to Augment our dataset to artificially increase its size and diversity. This might improve the model's ability to generalize and prevent overfitting.
we can use techniques like random cropping, flipping, and rotations.
2. We can also Experiment with different optimizers like Adam.
Adjust the learning rate.
optimizer = optim.Adam(model.parameters(), lr=0.001)
