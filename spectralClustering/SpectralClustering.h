/* Spectral clustering is a graph-based technique to map original streamlines to a spectral embedding space.
 * The problem itself is a NP-hard and we used instead a relaxed versions with Graph Laplacians.
 * Detailed procedures can be referred at https://tarekmamdouh.wordpress.com/2014/09/28/spectral-clustering/
 * and TVCG paper http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6702500
 */


#ifndef _SPECTRAL_CLUSTERING_H_
#define _SPECTRAL_CLUSTERING_H_

#include "Evrot.h"
#include <unordered_set>
#include <map>
#include <queue>
#include <string>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/SVD>
#include <climits>
#include <float.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <map>
#include <algorithm>
#include <sys/time.h>
#include <set>
#include <queue>
#include "Initialization.h"

using namespace std;
using namespace Eigen;

/* local scaling for Gaussian kernel size */
/* might be defined as 0.05*totalCount as in Blood Flow Clustering and Applications in
Virtual Stenting of Intracranial Aneurysms */


/* update date size for gradient descent */
#ifndef GradientStep
	#define GradientStep 0.3
#endif


struct Para
{
	/* Laplacian option: 1.Normalized Laplacian, 2.Unsymmetric Laplacian */
	int LaplacianOption;

	/* local scaling by sorted distance: true, false */
	bool isDistSorted;

	/* preset number of clusters */
	int numberOfClusters;

	/* post-processing method: 1.k-means, 2.eigenvector rotation*/
	int postProcessing;

	/* derivative method for eigen rotation: 1.numerical derivative, 2.true derivative */
	int mMethod;

};


struct Greater
{
	bool operator()(const float& a, const float& b)
	{
		return a>b;
	}
};


class SpectralClustering
{

public:

/* default constructor */
	SpectralClustering();

/* argument constructor with argc and argv */
	SpectralClustering(const int& argc, char **argv, const Para& p, bool& automatic);

/* destructor */
	~SpectralClustering();

/* perform clustering function */
	void performClustering(const int& presetCluster);

private:

/**********************************************************************************************************
 **************************************   Private member variables   **************************************
 **********************************************************************************************************/

/* group information */
	std::vector<int> group;

/* activityList vector to store event */
	std::vector<string> activityList;

/* timeList vector to store time information */
	std::vector<string> timeList;

/* how many clusters to be needed */
	int numberOfClusters = -1;

/* distance range vector */
	std::vector<float> distRange;

/* Gaussian kernel radius for generating adjacency matrix */
	std::vector<float> sigmaVec;

/* Laplacian option, 1: Unnormalized Laplacian, 2: normalized Laplacian, 3: Random Walk Laplacian */
	int LaplacianOption = -1;

/* what kind of 5-th neighbor point would be obtained? */
	bool isDistSorted = -1;

/* what kind of post-processing is to be chosen */
	int postProcessing = -1;

/* scaling factor for spectral clustering to decide Gaussian kernel size */
	int SCALING;

/* a distance matrix for input */
	Eigen::MatrixXf distanceMatrix;

/* node number in the undirected graph */
	int numOfNodes;

/* point coordinates */
	std::vector<Eigen::Vector3f> coordinates;	


/**********************************************************************************************************
 **************************************   Private member functions   **************************************
 **********************************************************************************************************/

/* set dataset from user command */
	void setDataset(const int& argc, char **argv);

/* set parameter */
	void getParameterUserInput();

/* set automatic parameter */
	void setParameterAutomatic(const Para& p);

/* run clustering based on different norm */
	void clusterByNorm();

/* get weighted adjacency matrix by Gaussian kernel */
	void getAdjacencyMatrix(Eigen::MatrixXf& adjacencyMatrix);

/* get degree matrix */
	void getDegreeMatrix(const Eigen::MatrixXf& adjacencyMatrix, Eigen::DiagonalMatrix<float,Dynamic>& degreeMatrix);

/* get Laplacian matrix */
	void getLaplacianMatrix(const Eigen::MatrixXf& adjacencyMatrix, Eigen::DiagonalMatrix<float,Dynamic>& degreeMatrix,
							Eigen::MatrixXf& laplacianMatrix);

/* decide optimal cluster number by eigenvectors of Laplacian matrix */
	void getEigenClustering(const Eigen::MatrixXf& laplacianMatrix);

/* get local scaling from NIPS 2002 paper */
	void getSigmaList();

/* read CSV data into matrixXf */
	void readCSV(const char* csvName);	

/* read txt file into matrixXf */
	void readTXT(const char* txtName);

/* read txt point  into matrixXf */
	void readPoint(const char* txtName);	

/* url: https://www.cs.cmu.edu/~aarti/Class/10701/readings/Luxburg06_TR.pdf */
/********************************** Perform k-means clustering *********************************************/
	/* normalize each row first */
	void normalizeEigenvec(Eigen::MatrixXf& eigenVec);

	/* perform k-means clustering */
	void performKMeans(const Eigen::MatrixXf& eigenVec,
					   std::vector<int>& storage,
					   std::vector<std::vector<int> >& neighborVec);

	/* print group information */
	void printGroup();

	/* void print vtk file */
	void printVTK();


/********************************** Vector Rotation from library *********************************************
 ********************************** from library https://github.com/pthimon/clustering ***********************/
	float mMaxQuality = 0;
	int mMethod = -1;

	/* get cluster information based on eigenvector rotation */
	void getEigvecRotation(std::vector<int>& storage, std::vector<std::vector<int> >& neighborVec,
			               Eigen::MatrixXf& clusterCenter, const Eigen::MatrixXf& X);

	/* set group information */
	void setGroup(const std::vector<std::vector<int> >& neighborVec);

};

void getMatrixPow(Eigen::DiagonalMatrix<float,Dynamic>& matrix, const float& powNumber);


#endif
