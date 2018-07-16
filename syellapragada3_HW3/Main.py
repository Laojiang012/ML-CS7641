import Clustering
import DimensionalityReduction
import DR_Clustering
import Clustering_NN
import NeuralNetwork

DimensionalityReduction() #1. Performs dimensionality reduction (PCA, ICA, RF, RP) and produces metrics used to indetify the number of features to retain in each of the reduction methods
DR_Clustering() #2. Upon identifying the best number of features to retain in PCA, ICA, RF and RP in the previous step, we use this to create new reduced datasets
Clustering() #3. We perform KM and EM clustering on the original and the reduced datasets produced in the previous step
NeuralNetwork() #4. We apply neural network on the original and the dimensionally reduced datasets
Clustering_NN() #5. We transform the original and reduced datasets using the clustering algorithms and rerun the neural network algorithm on this transformed space.
