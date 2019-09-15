#include "Initialization.h"


void Initialization::generateFromSamples(MatrixXf& clusterCenter,
								    	 const int& column,
								    	 const MatrixXf& cArray,
								    	 const int& Cluster)
{
	clusterCenter = MatrixXf(Cluster,column);
	std::vector<int> number(Cluster);
	srand(time(0));

	const int& MaxNum = cArray.rows();

	std::cout << MaxNum << std::endl;

	number[0] = rand()%MaxNum;
	int randNum, chosen = 1;
	bool found;
	for (int i = 1; i < Cluster; ++i)
	{
		do
		{
			randNum = rand()%MaxNum;
			found = false;
			for(int j=0;j<chosen;j++)
			{
				if(randNum==number[j])
				{
					found = true;
					break;
				}
			}
		}while(found!=false);
		number[i] = randNum;
		++chosen;
	}
	assert(chosen==Cluster);
	assert(column==cArray.cols());

#pragma omp parallel for schedule(dynamic) num_threads(8)
	for (int i = 0; i < Cluster; ++i)
	{
		clusterCenter.row(i) = cArray.row(number[i]);
	}

}


/* k-means++ for better initialization of k-means */
void Initialization::generateFarSamples(MatrixXf& clusterCenter,
								   	    const int& column,
								   		const MatrixXf& cArray,
								   		const int& Cluster)
{
	assert(column==cArray.cols());
	const int Total = cArray.rows();
	clusterCenter = MatrixXf(Cluster,column);
	int number[Cluster], selection;
	srand(time(0));
	const int& MaxNum = cArray.rows();
	number[0] = rand()%MaxNum;
	int chosen = 1;

	float percentage, nearest, toCentroid;
	VectorXf distance(Total);
	float squredSummation, left, right;
	while(chosen<Cluster)
	{
		percentage = float(rand()/(float)RAND_MAX);
		std::cout << percentage << std::endl;
		for (int i = 0; i < Total; ++i)
		{
			nearest = FLT_MAX;
			for (int j = 0; j < chosen; ++j)
			{
				toCentroid = (cArray.row(i)-cArray.row(number[j])).norm();
				if(nearest>toCentroid)
					nearest=toCentroid;
			}
			distance(i)=nearest*nearest;
		}

		distance = distance/distance.sum();

		left = 0.0, right = 0.0;
		for (int i = 0; i < Total; ++i)
		{
			left = right;
			right += distance(i);
			if(left < percentage && percentage <= right)
			{
				selection = i;
				break;
			}
		}
		number[chosen] = selection;
		chosen++;
	}

#pragma omp parallel for schedule(dynamic) num_threads(8)
	for (int i = 0; i < Cluster; ++i)
	{
		clusterCenter.row(i) = cArray.row(number[i]);
	}
}