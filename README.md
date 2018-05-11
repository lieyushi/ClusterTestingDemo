# ClusterTestingDemo

Author: Lieyu Shi

		Ph.D. student in Computer Science, University of Houston

Email: shilieyu91@gmail.com

1. spectralClusteringTesting
	
   This includes testing program to correctly classify ring-alike point cloud data set by spectral clustering algorithm.

   I provide eigenrotation minimization and k-means++ as post-processing after Laplacian-based dimension reduction.

   ------------ Possibility is that ring structures can not be detected by k-means++ due to random initilization.

   ------------ Since no Gold Rules exist for comparison, all external measurement, including (adjusted) Rand Index, Entropy, V-measure are all not working.
   
   ------------ Eigenvector rotation minization might not be able to detect correct structures.


