# MachineLearningP2



File gpt contains Grey's implimentation on the second part of the project (SVMs). To use this folder:
  main.py: simple SMO implimentation with each kernel type. Function smo() contains the algorithm
  opm1.py: optimized SMO based on main.py. Takes optimal j finding and cuts time significantly
  datasets.py: run to recalculate blobs or spirals dataset if new blobs or spirals are desired
  UnusedDatasets: a list of datasets that are not being used. mnist.csv was too large of a file to upload, so it has a placeholder
  *.csv: dataset able to be used in either main.py or opm1.py
