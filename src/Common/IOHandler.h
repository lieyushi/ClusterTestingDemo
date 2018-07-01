#ifndef _IOHANDLER_H_
#define _IOHANDLER_H_

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/SVD>
#include <climits>
#include <float.h>
#include <unordered_set>
#include <sys/stat.h>
#include "ClusteringAnalysis.h"


using namespace std;
using namespace Eigen;

class IOHandler
{

public:
	
/* read CSV data into matrixXf */
	static void readCSV(const char* csvName, Eigen::MatrixXf& distanceMatrix, int& numOfNodes);	

/* read txt file into matrixXf */
	static void readTXT(const char* txtName, int& numOfNodes, Eigen::MatrixXf& distanceMatrix);

/* print group information */
	static void printGroup(const std::vector<int>& group);


/* read txt point  into matrixXf */
	static void readPoint(const char* txtName, string& name, Eigen::MatrixXf& coordinates,
						  int& numOfNodes, Eigen::MatrixXf& distanceMatrix);

/* void print vtk file */
	static void printVTK(const int& numOfNodes, const Eigen::MatrixXf& coordinates, const std::vector<int>& group,
		                 const string& name, const string& groupName);

/* print information into README */
	static void writeReadMe(const Analysis& analysis, const string& dataSet, const string& clustering);

/* print information into README */
	static void writeReadMe(const float& value, const string& dataSet, const string& clustering, const string& value_name);
};

#endif