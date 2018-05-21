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


2. k-means
	
   Provide two initial sampling methods for k-means, i.e., directly taking samples or k-means++. The clustering results should be spherical and convex.

3. k-medoids
	
   Provide two medoid-finding methods, i.e., median or iterative methods. The clustering results should be convex (not strictly spherical).

3. Agglomerative hierarchical clustering (AHC)

   Provide three linkages for option of user-input, i.e., single linakge, complete linkage and average linkage. Wards method is omitted and interested readers can refer to wikipage 

   for a detailed exploration.

   Results show that single-linkage AHC could detect natural clusters of any shape (specially chaining shapes).

4. DEBSCAN

   The parameter setting process is a critical step for DBSCAN clustering algorithm and we provide an automatic setting method. Also, manual input for \epsilon is an option for our 

   program.



