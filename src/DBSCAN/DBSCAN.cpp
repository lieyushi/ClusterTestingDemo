#include "DBSCAN.h"


DBSCAN::DBSCAN()
{

}

/* argument constructor */
DBSCAN::DBSCAN(const int& argc, char* argv[])
{
	if(argc!=2)
	{
		std::cout << "Error for argument input! Should be ./kmeans dataset" << std::endl;
		exit(1);
	}

	IOHandler::readPoint(argv[1], name, coordinates, numOfNodes, distanceMatrix);

	group = std::vector<int>(numOfNodes,-1);

	nodeVec = vector<PointNode>(numOfNodes, PointNode());
}


/* destructor */	
DBSCAN::~DBSCAN()
{

}


/* run k-means clustering */
void DBSCAN::performClustering()
{
	int minPts;
	std::cout << "Input the minPts for DBSCAN in [0" << ", "
			<< numOfNodes << "], 6 preferred: " << std::endl;
	std::cin >> minPts;
	if (minPts <= 0 || minPts >= numOfNodes) {
		std::cout << "Error for out-of-range minPts!" << std::endl;
		exit(1);
	}

	float radius_eps;

	std::cout << "Want to have radius option? 1.minPts-th average dist, 2.user input." << std::endl;
	int radiusOption;
	std::cin >> radiusOption;
	assert(radiusOption==1||radiusOption==2);

	if(radiusOption==1)
		radius_eps = getAverageDist(minPts);
	else if(radiusOption==2)
	{
		float minDist, maxDist;
		getDistRange(minDist, maxDist);
		std::cout << "Distance range is [" << minDist << ", " << maxDist << "]." << std::endl;

		std::cout << "Input a ratio between [0,1.0] of max " << maxDist << ":" << std::endl;
		std::cin >> radius_eps;
		assert(radius_eps>0 && radius_eps<1.0);
		radius_eps*=maxDist;
	}

	DensityClustering(radius_eps, minPts);

	IOHandler::printVTK(numOfNodes, coordinates, group, name, string("dbscan"));

	Analysis analysis;
	analysis.computeValue(coordinates, distanceMatrix, group);
	std::cout << "Silhouette is " << analysis.getSilhouette() << ", db index is " << analysis.getDBIndex()
	          << ", gamma statistics is " << analysis.getGamma() << std::endl;
	IOHandler::writeReadMe(analysis, name , "DBSCAN");
}


void DBSCAN::DensityClustering(const float& radius_eps, const int& minPts) {
	int C = 0;
	for (int i = 0; i < numOfNodes; ++i) {
		if (nodeVec[i].visited)
			continue;
		nodeVec[i].visited = true;
		vector<int> neighbor = regionQuery(i, radius_eps);
		if (neighbor.size() < minPts)
			nodeVec[i].type = NOISE;
		else {
			expandCluster(i, neighbor, C, radius_eps, minPts);
			++C;
		}
	}

	numOfClusters = -1;
	for (int i = 0; i < numOfNodes; ++i)
	{
		group[i] = nodeVec[i].group;
		numOfClusters = std::max(group[i], numOfClusters);
	}

	++numOfClusters;

	std::cout << "DBSCAN finally forms " << numOfClusters << " groups!" << std::endl;

}



/* compute minPts-th dist for all candidates */
const float DBSCAN::getAverageDist(const int& minPts) {
	float result = 0.0;
#pragma omp parallel num_threads(8)
	{
#pragma omp for nowait
		for (int i = 0; i < numOfNodes; ++i) {
			/* use a priority_queue<float> with n*logk time complexity */
			std::priority_queue<float> minDistArray;
			float tempDist;
			for (int j = 0; j < numOfNodes; ++j) {
				if (i == j)
					continue;
				tempDist = distanceMatrix(i,j);

				minDistArray.push(tempDist);
				if (minDistArray.size() > minPts)
					minDistArray.pop();
			}

#pragma omp critical
			result += minDistArray.top();
		}
	}
	return result / numOfNodes;
}



const vector<int> DBSCAN::regionQuery(const int& index, const float& radius_eps) {
	vector<int> neighborArray;
	neighborArray.push_back(index);
	float tempDist;
	for (int i = 0; i < numOfNodes; ++i) {
		if (i == index)
			continue;

		/* in case somebody uses distance matrix */

		tempDist = distanceMatrix(index, i);
		if (tempDist <= radius_eps)
			neighborArray.push_back(i);
	}
	return neighborArray;
}



void DBSCAN::expandCluster(const int& index, vector<int>& neighbor, const int& cluster_id, 
	                       const float& radius_eps, const int& minPts) 
{
	nodeVec[index].group = cluster_id;
	int insideElement;
	for (int i = 0; i < neighbor.size(); ++i) {
		insideElement = neighbor[i];
		if (!nodeVec[insideElement].visited) {
			nodeVec[insideElement].visited = true;
			vector<int> newNeighbor = regionQuery(insideElement, radius_eps);
			if (newNeighbor.size() >= minPts) {
				neighbor.insert(neighbor.end(), newNeighbor.begin(),
						newNeighbor.end());
			}
		}
		if (nodeVec[insideElement].group == -1)
			nodeVec[insideElement].group = cluster_id;
	}
}


void DBSCAN::getDistRange(float& minDist, float& maxDist) {
	const float& Percentage = 0.1;
	const int& chosen = int(Percentage * numOfNodes);
	minDist = FLT_MAX;
	maxDist = -1.0;
#pragma omp parallel num_threads(8)
	{
	#pragma omp for nowait
		for (int i = 0; i < chosen; ++i) {
			float tempDist;
			for (int j = 0; j < numOfNodes; ++j) {
				if (i == j)
					continue;
				tempDist = distanceMatrix(i,j);
	#pragma omp critical
				{
					if (tempDist < minDist)
						minDist = tempDist;
					if (tempDist > maxDist)
						maxDist = tempDist;
				}
			}
		}
	}
	std::cout << minDist << " " << maxDist << std::endl;
}
