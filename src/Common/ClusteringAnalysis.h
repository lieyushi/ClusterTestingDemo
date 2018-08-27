#ifndef _CLUSTERING_ANALYSIS_H_
#define _CLUSTERING_ANALYSIS_H_

#include <iostream>
#include <vector>
#include <float.h>
#include <math.h>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

class Analysis
{

public:

	Analysis();

	~Analysis();

	/* compute silhouette, DB index and GammaStatistics matrix */
	void computeValue(const Eigen::MatrixXf& coordinates,
					  const Eigen::MatrixXf& distanceMatrix,
					  const std::vector<int>& group);

	/* public accessor for silhouette */
	const float& getSilhouette() const;

	/* public accessor for dbIndex */
	const float& getDBIndex() const;

	/* public accessor for gamma */
	const float& getGamma() const;


private:

	float silhouette;

	float gamma;

	float dbIndex;

	/* compute silhouette width */
	void computeSilhouette(const std::vector<int>& group, const Eigen::MatrixXf& distanceMatrix,
						   const std::vector<std::vector<int> >& storage);

	/* compute dbIndex */
	void computeDBIndex(const Eigen::MatrixXf& coordinates,
						const std::vector<int>& group,
						const std::vector<std::vector<int> >& storage);

	/* compute gamma */
	void computeGamma(const Eigen::MatrixXf& distanceMatrix,
					  const Eigen::MatrixXf& idealDistM);

	/* get Ideal distance matrix */
	void getIdealM(Eigen::MatrixXf& idealDistM, const std::vector<std::vector<int> >& storage, const int& Row);

	/* get A_i for silhouette computation */
	const float getA_i(const Eigen::MatrixXf& distanceMatrix, const std::vector<std::vector<int> >& storage,
					   const std::vector<int>& group, const int& index);

	/* get B_i for silhouette computation */
	const float getB_i(const Eigen::MatrixXf& distanceMatrix, const std::vector<std::vector<int> >& storage,
					   const std::vector<int>& group, const int& index);

};


#endif