#include "Kmedoids.h"


Kmedoids::Kmedoids()
{

}

/* argument constructor */
Kmedoids::Kmedoids(const int& argc, char* argv[])
{
	if(argc!=2)
	{
		std::cout << "Error for argument input! Should be ./kmedoids dataset" << std::endl;
		exit(1);
	}

	IOHandler::readPoint(argv[1], name, coordinates, numOfNodes, distanceMatrix);

	group = std::vector<int>(numOfNodes);

	std::cout << "Input preset number of clusters: " << std::endl;
	std::cin >> numOfClusters;

	assert(numOfClusters>0 && numOfClusters<=numOfNodes);

	std::cout << "Choose a way to compute medoids: 1.direct samples, 2.iteration: " << std::endl;
	int medoidOption;
	std::cin >> medoidOption;
	assert(medoidOption==1 || medoidOption==2);

	isSample = (medoidOption==1);
}


/* destructor */	
Kmedoids::~Kmedoids()
{

}


/* run k-means clustering */
void Kmedoids::performClustering()
{
	performKmeans();
	IOHandler::printVTK(numOfNodes, coordinates, group, name, "kmedoids");

	Analysis analysis;
	analysis.computeValue(coordinates, distanceMatrix, group);
	std::cout << "Silhouette is " << analysis.getSilhouette() << ", db index is " << analysis.getDBIndex()
	          << ", gamma statistics is " << analysis.getGamma() << std::endl;
	IOHandler::writeReadMe(analysis, name , "K-medoids");
}


/* run k-means clustering */
void Kmedoids::performKmeans()
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
				}
			}
		}

		computeMedoids(centerTemp, neighborVec);

		moving = FLT_MIN;

	#pragma omp parallel for reduction(max:moving) num_threads(8)
		for (int i = 0; i < numOfClusters; ++i)
		{
			if(storage[i]>0)
			{
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



void Kmedoids::computeMedoids(MatrixXf& centerTemp, 
							  const vector<vector<int> >& neighborVec)
{
	centerTemp = MatrixXf(numOfClusters,coordinates.cols());
	if(isSample)//centroid is from samples with minimal L1 summation
				//use Voronoi iteration https://en.wikipedia.org/wiki/K-medoids
	{
	#pragma omp parallel for schedule(dynamic) num_threads(8)
		for(int i=0;i<neighborVec.size();++i)
		{
			const vector<int>& clusMember = neighborVec[i];
			const int& clusSize = clusMember.size();
			MatrixXf mutualDist = MatrixXf::Zero(clusSize, clusSize);
			/*mutualDist to store mutual distance among lines of each cluster */
			for(int j=0;j<clusSize;++j)
			{
				for(int k=j+1;k<clusSize;++k)
				{
					mutualDist(j,k) = distanceMatrix(clusMember[j],
						clusMember[k]);
					mutualDist(k,j) = mutualDist(j,k);
				}
			}

			float minL1_norm = FLT_MAX, rowSummation;
			int index = -1;
			for(int j=0;j<clusSize;++j)
			{
				rowSummation = mutualDist.row(j).sum();
				if(rowSummation<minL1_norm)
				{
					minL1_norm = rowSummation;
					index = j;
				}
			}
			centerTemp.row(i)=coordinates.row(clusMember[index]); 
		}
	}

	else//use Weiszfeld's algorithm to get geometric median
		//reference at https://en.wikipedia.org/wiki/Geometric_median
	{
		MatrixXf originCenter = centerTemp;
	#pragma omp parallel for schedule(dynamic) num_threads(8)
		for(int i=0;i<numOfClusters;++i)
		{
			const vector<int>& clusMember = neighborVec[i];
			const int& clusSize = clusMember.size();
			float distToCenter, distInverse, percentage = 1.0;
			int tag = 0;
			while(tag<=20&&percentage>=0.01)
			{
				VectorXf numerator = VectorXf::Zero(coordinates.cols());
				VectorXf previous = centerTemp.row(i);
				float denominator = 0;
				for(int j=0;j<clusSize;++j)
				{
					distToCenter = (centerTemp.row(i)-coordinates.row(clusMember[j])).norm();
					distInverse = (distToCenter>1.0e-8)?1.0/distToCenter:1.0e8;
					numerator += coordinates.row(clusMember[j])*distInverse;
					denominator += distInverse;
				}
				centerTemp.row(i) = numerator/denominator;
				percentage = (centerTemp.row(i)-previous).norm()/previous.norm();
				tag++;
			}
		}
	}
}
