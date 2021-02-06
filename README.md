# SNA-PySpark-GraphFrames
This repository is for the SNA project on graph sampling using PySpark, GraphFrames and NetworkX.
We implemented a clustering-based technique to sample a graph, using parallel random walks, one on each cluster. The percentage of nodes we decided to obtain from each cluster is proportional to the number of nodes in it, and inversely proportional to its clustering coefficient.

The main code is in main.py, you can open the file in an editor, choose a value for the alpha and maxIter parameters and run it.

The technical report for this implementation with detailed description of methodology, experiments and results can be found in file "Community-Based-Sampling-Using-Parallel-Random-Walks.pdf"
