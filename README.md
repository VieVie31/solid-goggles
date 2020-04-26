# Active Learning Label Cleaning

The aim of this project is to provide a simple interface to clean images annotations for a given label.

The images should be putted in `images/` with a `.jpg` extension.
The features corresponding to each image should be putted in `features/` with the same name and with the `.feat` extension (the format shoud be a simple numpy vector containing floats…).

The interface is a web interface with backend powered by `bonapity`.


## How does it work ?

A first heuristic will look for anoalous exampes in the features base,
the program will display the classification (anomaly detection) results.

From this interface, the user can correct the classification done by the model,
by changing labels of some images.

The a suppervised model will be fitted from this user annotation and provide
another classification.
The user will be able to iterate with this step and the previous one until 
results are found satisfactory.

Finaly a `.txt` file containing the ids of images to keep will be produced.



