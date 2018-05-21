#include "Kmeans.h"


Kmeans::Kmeans()
{

}

/* argument constructor */
Kmeans::Kmeans(const int& argc, char* argv[])
{
	if(argc!=2)
	{
		std::cout << "Error for argument input! Should be ./kmeans dataset" << std::endl;
		exit(1);
	}

	IOHandler::readPoint(argv[1], name, coordinates, numOfNodes, distanceMatrix);

	group = std::vector<int>(numOfNodes);

	std::cout << "Input preset number of clusters: " << std::endl;
	std::cin >> numOfClusters;

	assert(numOfClusters>0 && numOfClusters<=numOfNodes);
}


/* destructor */	
Kmeans::~Kmeans()
{

}


/* run k-means clustering */
void Kmeans::performClustering()
{
	performKmeans();
	IOHandler::printVTK(numOfNodes, coordinates, group, name, string("kmeans"));
}


/* run k-means clustering */
void Kmeans::performKmeans()
{
	const int& Column = coordinates.cols();

	float moving=1000, tempMoving, before;

	/* centerTemp is temporary term for storing centroid position, clusterCenter is permanent */
	MatrixXf centerTemp, clusterCenter;

	/* chosen from sample for initialization of k-means */

	int sampleInitialization;
	std::cout << "Choosen initial sampling strategy? 1.directly from samples, 2.k-means++: ";
	std::cin >> sampleInitialization;
	assert(sampleInitialization==1 || sampleInitialization==2);

	if(sampleInitialization==1)
		Initialization::generateFromSamples(clusterCenter,Column,coordinates,numOfClusters);
	else if(sampleInitialization==2)
		Initialization::generateFarSamples(clusterCenter,Column,coordinates,numOfClusters);

	int tag = 0;

	std::vector<std::vector<int> > neighborVec=std::vector< std::vector<int> >(numOfClusters);

	std::vector<int> storage(numOfClusters);

	float PCA_KMeans_delta, KMeans_delta;

	std::cout << "...k-means started!" << std::endl;

	struct timeval start, end;
	gettimeofday(&start, NULL);

	do
	{
		before = moving;

		/* preset cluster number recorder */
		std::fill(storage.begin(), storage.end(), 0);
		centerTemp = MatrixXf::Zero(numOfClusters, Column);

	#pragma omp parallel for schedule(dynamic) num_threads(8)
		for (int i = 0; i < numOfClusters; ++i)
		{
			neighborVec[i].clear();
		}

	#pragma omp parallel num_threads(8)
		{
		#pragma omp for nowait
			for (int i = 0; i < numOfNodes; ++i)
			{
				float dist = FLT_MAX;
				float temp;
				int clusTemp = -1;
				for (int j = 0; j < numOfClusters; ++j)
				{
					temp = (coordinates.row(i)-clusterCenter.row(j)).norm();
					if(temp<dist)
					{
						dist = temp;
						clusTemp = j;
					}
				}

			#pragma omp critical
				{
					++storage[clusTemp];
					neighborVec[clusTemp].push_back(i);
					group[i] = clusTemp;
					centerTemp.row(clusTemp)+=coordinates.row(i);
				}
			}
		}

		moving = FLT_MIN;

	#pragma omp parallel for reduction(max:moving) num_threads(8)
		for (int i = 0; i < numOfClusters; ++i)
		{
			if(storage[i]>0)
			{
				centerTemp.row(i)/=storage[i];
				tempMoving = (centerTemp.row(i)-clusterCenter.row(i)).norm();
				clusterCenter.row(i) = centerTemp.row(i);
				if(moving<tempMoving)
					moving = tempMoving;
			}
		}
		std::cout << "K-means iteration " << ++tag << " completed, and moving is "<< moving << "!" << std::endl;
	}while(abs(moving-before)/before >= 1.0e-5 && tag < 50);

	gettimeofday(&end, NULL);
	float timeTemp = ((end.tv_sec-start.tv_sec)*1000000u+end.tv_usec-start.tv_usec)/1.e6;
	activityList.push_back("K-means takes: ");
	timeList.push_back(to_string(timeTemp)+" s");
}
