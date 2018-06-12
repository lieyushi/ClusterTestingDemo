#ifndef _DBSCAN_H_
#define _DBSCAN_H_


#include <algorithm>
#include <sys/time.h>
#include <queue>
#include "IOHandler.h"
#include "ClusteringAnalysis.h"
#include "Initialization.h"

enum PointType
{
	CORE = 0,
	BORDER,
	NOISE
};

struct PointNode
{
	int type;
	bool visited;
	int group;
	PointNode():type(-1), visited(false), group(-1)
	{}

	~PointNode()
	{}
};


class DBSCAN
{

public:

/* default constructor */	
	DBSCAN();

/* argument constructor */
	DBSCAN(const int& argc, char* argv[]);

/* destructor */	
	~DBSCAN();

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

/* pointNode list */
	std::vector<PointNode> nodeVec;

/* activityList vector to store event */
	std::vector<string> activityList;

/* timeList vector to store time information */
	std::vector<string> timeList;

/* expand cluster function in DBSCAN */
	void expandCluster(const int& index, vector<int>& neighbor, const int& cluster_id, 
	                   const float& radius_eps, const int& minPts);

/* regionQuery function for DBSCAN clustering */
	const vector<int> regionQuery(const int& index, const float& radius_eps);

/* compute minPts-th dist for all candidates */
	const float getAverageDist(const int& minPts);

/* Density-based clustering */
	void DensityClustering(const float& radius_eps, const int& minPts);

/* get the dist range for user-input of radius_epsilon */
	void getDistRange(float& minDist, float& maxDist);

};


#endif