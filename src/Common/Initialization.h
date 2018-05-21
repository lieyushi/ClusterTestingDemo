#ifndef _INITIALIZATION_H_
#define _INITIALIZATION_H_

#include <eigen3/Eigen/Dense>
#include <vector>
#include <ctime>
#include <cassert>
#include <iostream>
#include <float.h>

using namespace std;
using namespace Eigen;


class Initialization
{
public:
	static void generateFromSamples(MatrixXf& clusterCenter,
								    const int& column,
								    const MatrixXf& cArray,
								    const int& Cluster);

	static void generateFarSamples(MatrixXf& clusterCenter,
								   const int& column,
								   const MatrixXf& cArray,
								   const int& Cluster);

};


#endif