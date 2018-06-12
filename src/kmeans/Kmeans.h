#ifndef _KMEANS_H_
#define _KMEANS_H_


#include <algorithm>
#include <sys/time.h>
#include "IOHandler.h"
#include "Initialization.h"
#include "ClusteringAnalysis.h"

class Kmeans
{

public:

/* default constructor */	
	Kmeans();

/* argument constructor */
	Kmeans(const int& argc, char* argv[]);

/* destructor */	
	~Kmeans();

/* run k-means clustering */
	void performClustering();

private:

/* group information */
	std::vector<int> group;

/* distance matrix */
	Eigen::MatrixXf distanceMatrix;

/* coordinates */
	Eigen::MatrixXf coordinates;

/* data set name */
	string name;

/* number of nodes */
	int numOfNodes;

/* number of clusters */
	int numOfClusters;

/* run k-means clustering */
	void performKmeans();

/* activityList vector to store event */
	std::vector<string> activityList;

/* timeList vector to store time information */
	std::vector<string> timeList;

};


#endif