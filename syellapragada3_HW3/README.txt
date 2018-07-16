In this project we implement six algorithms.

Four algorithms are dimensionality reduction algorithms:
1. PCA
2. ICA
3. Randomized Projections
4. A feature selection algorithm based on information gain
All the these four algorithms have been implement using scipy library, in DimensionalityReduction.py

Two are clustering algorithms:

1. k-means clustering - implemented using scipy's kmeans library
2. Expectation Maximization - implemented with scipy's gaussian mixture model

To run this program, the Main.py file needs to be run, it makes calls to all relevant files as follows - 

DimensionalityReduction -  #1. Performs dimensionality reduction (PCA, ICA, RF, RP) and produces metrics used to indetify the number of features to retain in each of the reduction methods
DR_Clustering - #2. Upon identifying the best number of features to retain in PCA, ICA, RF and RP in the previous step, we use this to create new reduced datasets
Clustering - #3. We perform KM and EM clustering on the original and the reduced datasets produced in the previous step
NeuralNetwork - #4. We apply neural network on the original and the dimensionally reduced datasets
Clustering_NN - #5. We transform the original and reduced datasets using the clustering algorithms and rerun the neural network algorithm on this transformed space.

This program generate relevant graphs that were used for analysis and are included in the report submitted along with this code base.

The report is also included - "syellapragada3-hw3.pdf"