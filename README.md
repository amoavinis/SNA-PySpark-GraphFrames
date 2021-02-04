# SNA-PySpark-GraphFrames
This repository is for the SNA project, on graph sampling with PySpark and GraphFrames

Basic Idea of Implementation:
implement a technique based on the use of a clustering algorithm, before sampling starts
from the initial population, and also try to use a stratification-like sampling
Clear cluster sampling techniques operate as follows: after "splitting" nodes into clusters, only some of the clusters
are used for sampling. Contrary to that, we consider that sampling has to use all of the created clusters. Since clusters 
are not overlapping, sampling can be conducted in parallel on the clusters. The percentage of nodes of the sample (denoted as φ),
which we will sample from the overall population, is something that we cannot clarify yet. What we can make clear is 
the percentage of nodes that we will try to obtain from each cluster. That percentage will be proportional 
i. of the size of the cluster, 
and ii. of the cluster’s variance, regarding some property (or properties).
I.e. having a cluster whose nodes have variate node degrees, the sample size, that we will extract, will be bigger.


https://www.overleaf.com/6229828369qhjybwzvnthb
