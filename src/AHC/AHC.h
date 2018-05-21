#ifndef _AHC_H_
#define _AHC_H_


#include <algorithm>
#include <sys/time.h>
#include <unordered_map>
#include "IOHandler.h"
#include "Initialization.h"


// define a treeNode structure to store AHC clustering tree
struct Ensemble
{
	int index = -1;

	/* to alleviate the computational cost to traverse all node elements */
	std::vector<int> element;

	Ensemble(const int& index): index(index)
	{}

	Ensemble()
	{}
};


// remove two elements in template vector
template <class T>
void deleteVecElements(std::vector<T>& origine, const T& first, const T& second);


/* we will use a min-heap to perserve sorted distance for hirarchical clustering */
struct DistNode
{
	int first = -1, second = -1;
	float distance = -1.0;

	DistNode(const int& first, const int& second, const float& dist):first(first), second(second), distance(dist)
	{}

	DistNode()
	{}
};



class AHC
{

public:

/* default constructor */	
	AHC();

/* argument constructor */
	AHC(const int& argc, char* argv[]);

/* destructor */	
	~AHC();

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

/* preset cluster number */
	int numOfClusters;

/* run k-means clustering */
	void hierarchicalMerging(std::unordered_map<int, Ensemble>& nodeMap, std::vector<DistNode>& dNodeVec,
							 std::vector<Ensemble>& nodeVec);

/* set the map and vector */
	void setValue(std::vector<DistNode>& dNodeVec);

/* get distance between clusters given linkage type */
	const float getDistAtNodes(const vector<int>& firstList, const vector<int>& secondList);

/* activityList vector to store event */
	std::vector<string> activityList;

/* timeList vector to store time information */
	std::vector<string> timeList;

/* linkage option, 1.single, 2.complete, 3.average */
	int linkageType;

};


#endif