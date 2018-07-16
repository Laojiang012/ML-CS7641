1. Changes have been made to ABAGAIL library, so as to implement different optimization algorithms (RHC, SA, GA and MIMIC) over different iterations and different input parameters
2. Data pertaining to the neural network implementation is saved under the data folder. The split ratio of test and train dataset was 0.3
3. Under jython, 4 have been introduced - CP (Continuous peaks), Knapsack, Spam (neural networks on spam data set) and TSP (Travelling Salesman Problem)
4. Under CP, TSP and Knapsack folders, we also maintain three other folders for GA, MIMIC and SA executions of each of these problems. Executing the py files in each of these folders by using jython <filename>, corresponding csv files are generated for each of the input parameters
5. The generated csv files contain the data for each input parameters over different iteration counts
6. Similar technique was adopted for the neural network implementation using randomized optimization algorithms
7. Spam folder contains two sub folders namely GA and SA. Executing the spam_GA.py in GA and spam_SA.py in GA, from the terminal as jython files, produces the corresponding csv files (same as above)
8. The jython files for each of these problems are in the main folder for each problem and can be executed similarly.
9. Comparision plots have been generated using valueplot.py and timeplot.py files in each of the folders
10. Comparision of accuracy levels for the most suitable algorithm for each of these problems across different input parameters was also generated