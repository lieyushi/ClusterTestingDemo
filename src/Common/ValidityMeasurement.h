/*
 * ValidityMeasurement.h
 *
 *  Created on: Jun 24, 2018
 *      Author: lieyu
 */

#ifndef SRC_COMMON_VALIDITYMEASUREMENT_H_
#define SRC_COMMON_VALIDITYMEASUREMENT_H_

// A C++ implementation for the paper https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4761242
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/kruskal_min_spanning_tree.hpp>
#include <iostream>
#include <assert.h>
#include <tuple>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

using namespace std;
using namespace Eigen;

class ValidityMeasurement {
public:

	// f_c=h(DDc)*g(Sc)
	float f_c;

	ValidityMeasurement();
	virtual ~ValidityMeasurement();

	// function API for computing the validity measurement given a distance matrix 
	void computeValue(const Eigen::MatrixXf& distMatrix, const std::vector<int>& group);

private:
	// min and max of S_c
	float min_Sc, max_Sc;

	// get MST for each cluster given index and pair-wise distance, PCA case only
	void getMST_Parent_Node_distanceMatrix(std::tuple<float, float, float>& values,
				const std::vector<int>& clusterNode, const Eigen::MatrixXf& distMatrix);


	// compute the Sc by input range value for general cases
	const float get_Sc_by_range(const Eigen::MatrixXf& distMatrix, const std::vector<int>& clusterNode, 
													 const float& rangeValue, int& index);
};

#endif /* SRC_COMMON_VALIDITYMEASUREMENT_H_ */
