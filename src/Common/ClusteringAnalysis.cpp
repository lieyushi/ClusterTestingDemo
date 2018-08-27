#include "ClusteringAnalysis.h"


Analysis::Analysis()
{

}


Analysis::~Analysis()
{

}


/* compute silhouette, DB index and GammaStatistics matrix */
void Analysis::computeValue(const Eigen::MatrixXf& coordinates,
				  			const Eigen::MatrixXf& distanceMatrix,
				  			const std::vector<int>& group)
{
	int groupNumber = -1;

	const int& groupSize = group.size();
	for (int i = 0; i < groupSize; ++i)
	{
		groupNumber = std::max(groupNumber, group[i]);
	}

	/* store them in a storage, and omit -1 (noise group) */
	std::vector<std::vector<int> > storage(groupNumber+1);
	for (int i = 0; i < groupSize; ++i)
	{
		if(group[i]>=0)
			storage[group[i]].push_back(i);
	}

	computeSilhouette(group, distanceMatrix, storage);

	computeDBIndex(coordinates, group, storage);

	Eigen::MatrixXf idealDistM;

	getIdealM(idealDistM, storage, group.size());

	computeGamma(distanceMatrix, idealDistM);
}


const float& Analysis::getSilhouette() const
{
	return silhouette;
}


const float& Analysis::getDBIndex() const
{
	return dbIndex;
}


const float& Analysis::getGamma() const
{
	return gamma;
}


void Analysis::computeSilhouette(const std::vector<int>& group, const Eigen::MatrixXf& distanceMatrix,
					   			 const std::vector<std::vector<int> >& storage)
{
	const int& Row = distanceMatrix.rows();

// compute Silhouette value for each data
	float sSummation = 0;

#pragma omp parallel num_threads(8)
	{
	#pragma omp for nowait
		for (int i = 0; i < Row; ++i)
		{
			const float& a_i = getA_i(distanceMatrix, storage, group, i);
			const float& b_i = getB_i(distanceMatrix, storage, group, i);

			float s_i;
			if(abs(a_i-b_i)<1.0e-8)
				s_i = 0;
			else if(a_i<b_i)
				s_i = 1 - a_i/b_i;
			else
				s_i = b_i/a_i - 1;
			if(std::isnan(s_i))
			{
				std::cout << "Error for nan number!" << std::endl;
				exit(1);
			}

		#pragma omp critical
			sSummation += s_i;
		}
	}
	silhouette = sSummation/Row;

}


void Analysis::computeDBIndex(const Eigen::MatrixXf& coordinates,
							  const std::vector<int>& group,
							  const std::vector<std::vector<int> >& storage)
{
	dbIndex = 0.0;

	const int& groupNumber = storage.size();

	const int& Column = coordinates.cols();

	/* calculated the projected-space cenroid */
	Eigen::MatrixXf centroid(groupNumber, Column);

	/* average distance of all elements in cluster to its centroid */
	Eigen::VectorXf averageDist(groupNumber);

#pragma omp parallel for schedule(dynamic) num_threads(8)
	for(int i=0;i<groupNumber;++i)
	{
		Eigen::VectorXf tempCentroid = Eigen::VectorXf::Zero(Column);

		const std::vector<int>& clusterVec = storage[i];
		const int& clusterSize = clusterVec.size();

		for(int j=0;j<clusterSize;++j)
			tempCentroid+=coordinates.row(clusterVec[j]);

		/* get the centroid coordinates */
		centroid.row(i) = tempCentroid/clusterSize;

		float inClusterSum = 0.0;
		for(int j=0;j<clusterSize;++j)
			inClusterSum+=(coordinates.row(clusterVec[j])-centroid.row(i)).norm();

		averageDist(i) = inClusterSum/clusterSize;
	}

#pragma omp parallel num_threads(8)
	{
	#pragma omp for nowait
		for (int i = 0; i < groupNumber; ++i)
		{
			float maxValue = (float)INT_MIN, ratioDist;
			for (int j=0;j<groupNumber;++j)
			{
				if(i==j)
					continue;
				ratioDist = (averageDist(i)+averageDist(j))/(centroid.row(i)-centroid.row(j)).norm();

				if(maxValue<ratioDist)
					maxValue=ratioDist;
			}

		#pragma omp critical
			dbIndex += maxValue;
		}
	}
	dbIndex/=groupNumber;
}


void Analysis::computeGamma(const Eigen::MatrixXf& distanceMatrix,
				  			const Eigen::MatrixXf& idealDistM)
{
	const int& Row = distanceMatrix.rows();

	const int& totalNum = Row*(Row-1)/2;

	/* mean of values */
	float u_1 = 0.0, u_2 = 0.0;

	/* E(X*X) */
	float s_1 = 0.0, s_2 = 0.0, numerator = 0.0;

	for(int i=0;i<Row-1;++i)
	{
		for(int j=i+1;j<Row;++j)
		{
			/* update the mean u_1, u_2 */
			u_1+=distanceMatrix(i,j);
			u_2+=idealDistM(i,j);

			/* update the numerator */
			numerator+=distanceMatrix(i,j)*idealDistM(i,j);

			/* update the deviation */
			s_1+=distanceMatrix(i,j)*distanceMatrix(i,j);
			s_2+=idealDistM(i,j)*idealDistM(i,j);
		}
	}

	/* get mean of distM and idealDistM */
	u_1/=totalNum;
	u_2/=totalNum;

	/* get numerator for the computing */
	numerator-=totalNum*u_1*u_2;

	/* get standard deviation */
	s_1=sqrt(s_1/totalNum-u_1*u_1);
	s_2=sqrt(s_2/totalNum-u_2*u_2);

	if(std::isnan(s_1) || std::isnan(s_2))
	{
		std::cout << "standard deviation has nan error!" << std::endl;
		exit(1);
	}

	gamma = numerator/s_1/s_2/totalNum;
}


void Analysis::getIdealM(Eigen::MatrixXf& idealDistM, const std::vector<std::vector<int> >& storage, const int& Row)
{
	idealDistM = Eigen::MatrixXf::Constant(Row,Row,1.0);
#pragma omp parallel for schedule(dynamic) num_threads(8)
	for (int i = 0; i < storage.size(); ++i)
	{
		const std::vector<int>& eachVec = storage[i];
		const int& eachSize = eachVec.size();
		for (int j = 0; j < eachSize; ++j)
		{
			for (int k = 0; k < eachSize; ++k)
			{
				idealDistM(eachVec[j], eachVec[k]) = 0.0;
			}
		}
	}
}


const float Analysis::getA_i(const Eigen::MatrixXf& distanceMatrix,
							 const std::vector<std::vector<int> >& storage,
							 const std::vector<int>& group,
						 	 const int& index)
{
	const std::vector<int>& clusterSet = storage[group[index]];
	float inClusterDist = 0.0;
	for (int j = 0; j < clusterSet.size(); ++j)
	{
		if(clusterSet[j]!=index)
		{
			inClusterDist += distanceMatrix(index,clusterSet[j]);
		}
	}
	if(std::isnan(inClusterDist))
	{
		std::cout << "a_i has nan error!" << std::endl;
		exit(1);
	}
	float a_i;
	if(clusterSet.size()==1)
		a_i = 0;
	else
		a_i = inClusterDist/(clusterSet.size()-1);
	return a_i;
}


const float Analysis::getB_i(const Eigen::MatrixXf& distanceMatrix,
							 const std::vector<std::vector<int> >& storage,
							 const std::vector<int>& group,
							 const int& index)
{
	float outClusterDist = FLT_MAX, perClusterDist = 0;
	std::vector<int> outClusterSet;
	if(storage.size()==1)
		return 0;
	for (int j = 0; j < storage.size(); ++j) //j is group no.
	{
		if(j!=group[index]) //the other cluster
		{
			outClusterSet = storage[j];//get integer list of this group
			perClusterDist = 0;
			for (int k = 0; k < outClusterSet.size(); ++k)
			{
				perClusterDist+=distanceMatrix(index,outClusterSet[k]);
			}
			if(perClusterDist<0)
			{
				std::cout << "Error for negative distance!" << std::endl;
				exit(1);
			}
			perClusterDist/=outClusterSet.size();
			if(outClusterDist>perClusterDist)
				outClusterDist=perClusterDist;
		}
	}
	return outClusterDist;
}
