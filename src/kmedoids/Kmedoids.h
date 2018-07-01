#ifndef _KMEDOIDS_H_
#define _KMEDOIDS_H_


#include <algorithm>
#include <sys/time.h>
#include "IOHandler.h"
#include "Initialization.h"
#include "ClusteringAnalysis.h"
#include "ValidityMeasurement.h"


class Kmedoids
{

public:

/* default constructor */	
	Kmedoids();

/* argument constructor */
	Kmedoids(const int& argc, char* argv[]);

/* destructor */	
	~Kmedoids();

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

/* activityList vector to store event */
	std::vector<string> activityList;

/* timeList vector to store time information */
	std::vector<string> timeList;

/* whether get the medoids from sample or by iteration */
	bool isSample;

/* run k-means clustering */
	void performKmeans();

/* compute the medoids coordinates */
	void computeMedoids(MatrixXf& centerTemp, const vector<vector<int> >& neighborVec);

};


#endif