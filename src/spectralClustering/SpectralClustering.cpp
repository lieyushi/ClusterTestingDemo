#include "SpectralClustering.h"

/* default constructor */
SpectralClustering::SpectralClustering()
{

}

/* argument constructor with argc and argv */
SpectralClustering::SpectralClustering(const int& argc, char **argv, const Para& p, bool& automatic)
{
	setDataset(argc, argv);

	if(automatic)
		setParameterAutomatic(p);

	else
		getParameterUserInput();

}

/* destructor */
SpectralClustering::~SpectralClustering()
{

}

/* perform clustering function */
void SpectralClustering::performClustering(const int& presetCluster)
{
	//distance metric type
	/*  0: Euclidean Norm
		1: Fraction Distance Metric
		2: piece-wise angle average
		3: Bhattacharyya metric for rotation
		4: average rotation
		5: signed-angle intersection
		6: normal-direction multivariate distribution
		7: Bhattacharyya metric with angle to a fixed direction
		8: Piece-wise angle average \times standard deviation
		9: normal-direction multivariate un-normalized distribution
		10: x*y/|x||y| borrowed from machine learning
		11: cosine similarity
		12: Mean-of-closest point distance (MCP)
		13: Hausdorff distance min_max(x_i,y_i)
		14: Signature-based measure from http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6231627
		15: Procrustes distance take from http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6787131
	*/
	numberOfClusters = presetCluster;

	activityList.push_back("Preset numOfClusters is: ");
	timeList.push_back(to_string(presetCluster));

	clusterByNorm();

	std::cout << std::endl;
}


/* run clustering based on different norm */
void SpectralClustering::clusterByNorm()
{
	getSigmaList();

	Eigen::MatrixXf adjacencyMatrix, laplacianMatrix;
	Eigen::DiagonalMatrix<float,Dynamic> degreeMatrix;

	/* get weighted adjacency matrix by Gaussian kernel */
	getAdjacencyMatrix(adjacencyMatrix);

	/* get degree matrix */
	getDegreeMatrix(adjacencyMatrix, degreeMatrix);

	/* get Laplacian matrix */
	getLaplacianMatrix(adjacencyMatrix, degreeMatrix, laplacianMatrix);

	getEigenClustering(laplacianMatrix);
}


/* set dataset from user command */
void SpectralClustering::setDataset(const int& argc, char **argv)
{
	if(argc!=2)
	{
		std::cout << "Input argument should have 2!" << endl
		          << "./cluster inputFile_name(in dataset folder) " << std::endl;
		exit(1);
	}

	//IOHandler::readCSV(argv[1], distanceMatrix, numOfNodes);

	IOHandler::readPoint(argv[1], name, coordinates, numOfNodes, distanceMatrix);

	//distanceMatrix = Eigen::MatrixXf::Random(100,100);

	//numOfNodes = distanceMatrix.rows();

	SCALING = 0.05*numOfNodes;

	//SCALING = 7;
}


/* get local scaling from NIPS 2002 paper */
void SpectralClustering::getSigmaList()
{
	sigmaVec = std::vector<float>(numOfNodes);

	std::cout << "SCALING value is " << SCALING << std::endl;

	if(isDistSorted)
	{
		/* get SCALING-th smallest dist */
	#pragma omp parallel for schedule(dynamic) num_threads(8)
		for(int i=0;i<numOfNodes;++i)
		{
			/* this is a n*k implementation by linear scan */
			/*
			std::vector<float> limitVec(SCALING, FLT_MAX);
			float tempDist;
			for(int j=0;j<Row;++j)
			{
				if(i==j)
					continue;
				if(distanceMatrix)
					tempDist = distanceMatrix[i][j];
				else
					tempDist = getDisimilarity(ds.dataMatrix, i, j, normOption, object);
				// element is even larger than the biggest
				if(tempDist>=limitVec.back())
					continue;

				// update the SCALING smallest vec elements
				if(tempDist<limitVec.back())
					limitVec.back() = tempDist;
				for(int k=limitVec.size()-1;k>0;--k)
				{
					if(limitVec[k]<limitVec[k-1])
						std::swap(limitVec[k],limitVec[k-1]);
				}
			}
			*/

			/* instead we implement a n*logk priority_queue method for finding k-th largest element */
			//std::priority_queue<float> limitQueue;

			std::priority_queue<float> limitQueue;
			float tempDist;
			for(int j=0;j<numOfNodes;++j)
			{
				tempDist = distanceMatrix(i,j);

				limitQueue.push(tempDist);
				if(limitQueue.size()>SCALING)
					limitQueue.pop();
			}

			sigmaVec[i] = limitQueue.top();
		}
	}
	else
	{
		/* directly by index since in both papers only mention i-th neighboring point */
	#pragma omp parallel for schedule(dynamic) num_threads(8)
		for(int i=0;i<numOfNodes;++i)
		{
			if(i<SCALING)
			{
				sigmaVec[i]=distanceMatrix(i,SCALING);
			}
			else
			{
				sigmaVec[i]=distanceMatrix(i,SCALING-1);
			}
		}
	}
	std::cout << "Finish local scaling..." << std::endl;

}

/* get weighted adjacency matrix by Gaussian kernel */
void SpectralClustering::getAdjacencyMatrix(Eigen::MatrixXf& adjacencyMatrix)
{
	//in case of diagonal matrix element is not assigned
	adjacencyMatrix = Eigen::MatrixXf::Zero(numOfNodes, numOfNodes);
#pragma omp parallel for schedule(dynamic) num_threads(8)
	for(int i=0;i<numOfNodes;++i)
	{
		for(int j=0;j<numOfNodes;++j)
		{
			float dist_ij;
			if(i==j)
				continue;
			else
			{
				dist_ij = distanceMatrix(i,j);
			}

			if(dist_ij==0)
				continue;
			adjacencyMatrix(i,j)=exp(-dist_ij*dist_ij/sigmaVec[i]/sigmaVec[j]);
		}
	}

	std::cout << "Finish computing adjacency matrix!" << std::endl;
}


/* get degree matrix */
void SpectralClustering::getDegreeMatrix(const Eigen::MatrixXf& adjacencyMatrix, Eigen::DiagonalMatrix<float,Dynamic>& degreeMatrix)
{
	degreeMatrix = Eigen::DiagonalMatrix<float,Dynamic>(numOfNodes);
	Eigen::VectorXf v = VectorXf::Zero(numOfNodes);
#pragma omp parallel for schedule(dynamic) num_threads(8)
	for(int i=0;i<numOfNodes;++i)
	{
		float summation = 0;
		for(int j=0;j<numOfNodes;++j)
		{
			summation+=adjacencyMatrix(i,j);
		}
		v(i) = summation;
	}

	degreeMatrix.diagonal() = v;

	std::cout << "Fnish computing degree matrix!" << std::endl;
}


/* get Laplacian matrix */
void SpectralClustering::getLaplacianMatrix(const Eigen::MatrixXf& adjacencyMatrix,
		                                    Eigen::DiagonalMatrix<float,Dynamic>& degreeMatrix,
											Eigen::MatrixXf& laplacianMatrix)
{
	switch(LaplacianOption)
	{
	default:
	case 1:
	/* L = D^(-1)A */
		getMatrixPow(degreeMatrix, -1.0);
		laplacianMatrix=degreeMatrix*adjacencyMatrix;
		break;

	case 2:
		Eigen::MatrixXf dMatrix = Eigen::MatrixXf(adjacencyMatrix.rows(),adjacencyMatrix.cols());
		const Eigen::VectorXf& m_v = degreeMatrix.diagonal();
		for(int i=0;i<dMatrix.rows();++i)
			dMatrix(i,i) = m_v(i);
		laplacianMatrix = dMatrix-adjacencyMatrix;
		break;
	}
}


/* decide optimal cluster number by eigenvectors of Laplacian matrix */
void SpectralClustering::getEigenClustering(const Eigen::MatrixXf& laplacianMatrix)
{
	struct timeval start, end;
	gettimeofday(&start, NULL);

	/* eigen decomposition for Hermite matrix (real and symmetric matrix) */
	std::cout << "Eigen decomposition starts!..." << std::endl;
	SelfAdjointEigenSolver<MatrixXf> eigensolver(laplacianMatrix);
	std::cout << "Eigen decomposition ends!..." << std::endl;

	gettimeofday(&end, NULL);
	float timeTemp = ((end.tv_sec-start.tv_sec)*1000000u+end.tv_usec-start.tv_usec)/1.e6;
	activityList.push_back("Eigen decomposition takes: ");
	timeList.push_back(to_string(timeTemp)+" s");

	const int& eigenRows = numberOfClusters;
	//const int& eigenRows = 5;

	Eigen::MatrixXf eigenVec(eigenRows, numOfNodes);

	const int& Row = laplacianMatrix.rows();

	/* from paper we know it should get largest eigenvalues, and from eigen library we know it's latter */
	for(int i=Row-1;i>Row-eigenRows-1;--i)
		eigenVec.row(Row-1-i) = eigensolver.eigenvectors().col(i).transpose();

	eigenVec.transposeInPlace();

	/* how many elements in each cluster */
	std::vector<int> storage;

	/* which elements stored in each cluster */
	std::vector<std::vector<int> > neighborVec;

	/* centroid cluster */
	Eigen::MatrixXf clusterCenter;

	/* k-means as a post-processing */
	if(postProcessing==1)
	{
		normalizeEigenvec(eigenVec);

		performKMeans(eigenVec,storage,neighborVec);
	}
	/* eigenvector rotation */
	else if(postProcessing==2)
	{
		getEigvecRotation(storage,neighborVec,clusterCenter,eigenVec);

		setGroup(neighborVec);

		if(neighborVec.empty())
			return;
	}

	string groupName;

	if(postProcessing==1)
		groupName = "sc_kmeans";
	else if(postProcessing==2)
		groupName = "sc_eigen";

	IOHandler::printVTK(numOfNodes, coordinates, group, name, groupName);
}


void getMatrixPow(Eigen::DiagonalMatrix<float,Dynamic>& matrix, const float& powNumber)
{
	Eigen::VectorXf& m_v = matrix.diagonal();
#pragma omp parallel for schedule(dynamic) num_threads(8)
	for(int i=0;i<m_v.size();++i)
		m_v(i) = pow(m_v(i), powNumber);
}


/* normalize the matrix */
void SpectralClustering::normalizeEigenvec(Eigen::MatrixXf& eigenVec)
{
	const int& rows = eigenVec.rows();
#pragma omp parallel for schedule(dynamic) num_threads(8)
	for(int i=0;i<rows;++i)
	{
		eigenVec.row(i)/=eigenVec.row(i).norm();
	}
}


/* perform k-means clustering */
void SpectralClustering::performKMeans(const Eigen::MatrixXf& eigenVec,
									   std::vector<int>& storage,
									   std::vector<std::vector<int> >& neighborVec)
{

	const int& Row = eigenVec.rows();
	const int& Column = eigenVec.cols();

	float moving=1000, tempMoving, before;

	storage = std::vector<int>(numberOfClusters);

	/* centerTemp is temporary term for storing centroid position, clusterCenter is permanent */
	MatrixXf centerTemp, clusterCenter;

	/* chosen from sample for initialization of k-means */

	int sampleInitialization;
	std::cout << "Choosen initial sampling strategy? 1.directly from samples, 2.k-means++: ";
	std::cin >> sampleInitialization;
	assert(sampleInitialization==1 || sampleInitialization==2);

	if(sampleInitialization==1)
		Initialization::generateFromSamples(clusterCenter,Column,eigenVec,numberOfClusters);
	else if(sampleInitialization==2)
		Initialization::generateFarSamples(clusterCenter,Column,eigenVec,numberOfClusters);

	int tag = 0;

	neighborVec=std::vector< std::vector<int> >(numberOfClusters);

	float PCA_KMeans_delta, KMeans_delta;

	std::cout << "...k-means started!" << std::endl;

	struct timeval start, end;
	gettimeofday(&start, NULL);

	do
	{
		before = moving;
		/* preset cluster number recorder */
		std::fill(storage.begin(), storage.end(), 0);

		centerTemp = MatrixXf::Zero(numberOfClusters, Column);

	#pragma omp parallel for schedule(dynamic) num_threads(8)
		for (int i = 0; i < numberOfClusters; ++i)
		{
			neighborVec[i].clear();
		}

	//#pragma omp parallel num_threads(8)
		{
	//	#pragma omp for nowait
			for (int i = 0; i < Row; ++i)
			{
				float dist = FLT_MAX;
				float temp;
				int clusTemp;
				for (int j = 0; j < numberOfClusters; ++j)
				{
					temp = (eigenVec.row(i)-clusterCenter.row(j)).norm();
					if(temp<dist)
					{
						dist = temp;
						clusTemp = j;
					}
				}

	//		#pragma omp critical
				{
					storage[clusTemp]++;
					neighborVec[clusTemp].push_back(i);
					group[i] = clusTemp;
					centerTemp.row(clusTemp)+=eigenVec.row(i);
				}
			}
		}

		moving = FLT_MIN;

	#pragma omp parallel for reduction(max:moving) num_threads(8)
		for (int i = 0; i < numberOfClusters; ++i)
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
		std::cout << "K-means iteration " << ++tag << " completed, and moving is "
		<< moving << "!" << std::endl;
	}while(abs(moving-before)/before >= 1.0e-5 && tag < 50);

	gettimeofday(&end, NULL);
	float timeTemp = ((end.tv_sec-start.tv_sec)*1000000u+end.tv_usec-start.tv_usec)/1.e6;
	activityList.push_back("K-means takes: ");
	timeList.push_back(to_string(timeTemp)+" s");
}


/* get cluster information based on eigenvector rotation */
void SpectralClustering::getEigvecRotation(std::vector<int>& storage, std::vector<std::vector<int> >& neighborVec,
        								   Eigen::MatrixXf& clusterCenter, const Eigen::MatrixXf& X)
{
	mMaxQuality = 0;
	Eigen::MatrixXf vecRot;
	Eigen::MatrixXf vecIn = X.block(0,0,X.rows(),2);
	Evrot *e = NULL;

	struct timeval start, end;
	gettimeofday(&start, NULL);

	const int& xCols = X.cols();

	std::cout << "Eigenvector rotation starts within " << xCols << " columns..." << std::endl;
	for (int g=2; g <= xCols; g++)
	{
		// make it incremental (used already aligned vectors)
		std::cout << "column " << g << ":";
		if( g > 2 )
		{
			vecIn.resize(X.rows(),g);
			vecIn.block(0,0,vecIn.rows(),g-1) = e->getRotatedEigenVectors();
			vecIn.block(0,g-1,X.rows(),1) = X.block(0,g-1,X.rows(),1);
			delete e;
		}
		//perform the rotation for the current number of dimensions
		e = new Evrot(vecIn, mMethod);

		//save max quality
		if (e->getQuality() > mMaxQuality)
		{
			mMaxQuality = e->getQuality();
		}

		if(isnan(e->getQuality())||isinf(e->getQuality()))
		{
			std::cout << "Meet with nan or inf! Stop! " << std::endl;
			return;
		}

		std::cout << " max quality is " << mMaxQuality << ", Evrot has quality " << e->getQuality() << std::endl;
		//save cluster data for max cluster or if we're near the max cluster (so prefer more clusters)
		if ((e->getQuality() > mMaxQuality) || (mMaxQuality - e->getQuality() <= 0.001))
		{
			neighborVec = e->getClusters();
			vecRot = e->getRotatedEigenVectors();
		}
	}

	gettimeofday(&end, NULL);
	float timeTemp = ((end.tv_sec-start.tv_sec)*1000000u+end.tv_usec-start.tv_usec)/1.e6;
	activityList.push_back("Eigenvector rotation takes: ");
	timeList.push_back(to_string(timeTemp)+" s");

	if(neighborVec.empty())
		return;

	numberOfClusters = neighborVec.size();

	std::cout << "Finally " << numberOfClusters << " clusters formed!" << std::endl;
}

/* set automatic parameter */
void SpectralClustering::setParameterAutomatic(const Para& p)
{
	group = std::vector<int>(numOfNodes);

	/* the default value for streamline clustering is 2 normalized Laplacian */
	LaplacianOption = p.LaplacianOption;

	isDistSorted = p.isDistSorted;

	numberOfClusters = p.numberOfClusters;

	postProcessing = p.postProcessing;

	mMethod = p.mMethod;
}



/* set parameter */
void SpectralClustering::getParameterUserInput()
{
	group = std::vector<int>(numOfNodes);

	/* the default value for streamline clustering is 2 normalized Laplacian */
	std::cout << "---------------------------" << std::endl;
	std::cout << "Laplacian option: 1.Normalized Laplacian, 2.Unsymmetric Laplacian" << std::endl;
	std::cout << "..And in streamline clustering people tend to choose 1.Normalized Laplacian!-----------" << std::endl;
	std::cin >> LaplacianOption;
	assert(LaplacianOption==1||LaplacianOption==2);


	int sortedOption;
	std::cout << "Please choose whether local scaling by sorted distance: 1. yes, 2. no: " << std::endl;
	std::cin >> sortedOption;
	assert(sortedOption==1||sortedOption==2);
	if(sortedOption==1)
		sortedOption = true;
	else if(sortedOption==2)
		sortedOption = false;

	std::cout << "-----------------------------------------------------------------------" << std::endl;
	std::cout << "Input a desired cluster number among [1, " << numOfNodes << "]: ";
	std::cin >> numberOfClusters;
	assert(numberOfClusters>1 && numberOfClusters<numOfNodes/10);

	std::cout << "-----------------------------------------------------------------------" << std::endl;
	std::cout << "Input a post-processing method: 1.k-means, 2.eigenvector rotation: " << std::endl;
	std::cin >> postProcessing;
	assert(postProcessing==1||postProcessing==2);

	if(postProcessing==2)
	{
		std::cout << "------------------------------------------------" << std::endl;
		std::cout << "Please input derivative method: 1.numerical derivative, 2.true derivative." << std::endl;
		std::cin >> mMethod;
		assert(mMethod==1 || mMethod==2);
	}

}


/* set group information */
void SpectralClustering::setGroup(const std::vector<std::vector<int> >& neighborVec)
{
#pragma omp parallel for schedule(dynamic) num_threads(8)
	for (int i = 0; i < neighborVec.size(); ++i)
	{
		const std::vector<int>& eachVec = neighborVec[i];
		const std::size_t& eachSize = eachVec.size();
		for (int j = 0; j < eachSize; ++j)
		{
			group[eachVec[j]] = i;
		}
	}
}