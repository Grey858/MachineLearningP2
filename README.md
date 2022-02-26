# MachineLearningP2



File gpt contains Grey's implimentation on the second part of the project (SVMs). To use this folder:

  main.py: simple SMO implimentation with each kernel type. Function smo() contains the algorithm
  
  opm1.py: optimized SMO based on main.py. Takes optimal j finding and cuts time significantly
  
  datasets.py: run to recalculate blobs or spirals dataset if new blobs or spirals are desired
  
  UnusedDatasets: a list of datasets that are not being used. mnist.csv was too large of a file to upload, so it has a placeholder
  
  *.csv: dataset able to be used in either main.py or opm1.py

File SVM contains a simplified SMO algorithm, the multiclass algorithm, and the accuracy metrics from real world datasets
  DecisionTree.py: Contains implementation for the decision tree used in our paper as well as a method for one-hot-encoding data
  SMO.py: Contains the simplified and multiclass SMO implementations
  svmDriver.py: Contains the hyper-param search for each dataset as well as the decision boundary graphing for the spirals dataset
  svmtest.py: Containst the 3 cluster multiclassification visualization code
  
Main Directory 
  DecisionTree.py: Copy of the one in SVM
  mainDriver.py: Generated our random forest results
  ML_Democracy.py: Voting algoritm used in our paper. Can also be used for arbitrary models, not just Dtrees
  \*.py: other python files were used either to validate functionality or to explain to eachother hot to interface our code together. 
  \*\_results.csv: The dtree results files showing our feature search. The SVM results files are not included because the repo is getting cluttered, but are availible upon request  
