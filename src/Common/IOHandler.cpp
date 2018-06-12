#include "IOHandler.h"


/* read CSV data into matrixXf */
void IOHandler::readCSV(const char* csvName,
						Eigen::MatrixXf& distanceMatrix,
						int& numOfNodes)
{
	std::cout << "Started reading file..." << std::endl;

	ifstream fin((string("../test_data/")+string(csvName)).c_str(), ios::in);
	if(!fin)
	{
		std::cout << "File does not exist!" << std::endl;
		exit(1);
	}

	string line;
	getline(fin, line);

	int minNode = INT_MAX, maxNode = INT_MIN, index;

	stringstream ss;
	while(getline(fin,line))
	{
		ss.str(line);
		getline(ss, line, ',');
		index = atoi(line.c_str());
		minNode = std::min(index, minNode);
		maxNode = std::max(index, maxNode);

		ss.str("");
	}

	numOfNodes = (maxNode-minNode)+1;

	distanceMatrix = Eigen::MatrixXf::Zero(numOfNodes, numOfNodes);

	std::cout << "There are " << numOfNodes << " nodes in the CSV!" << std::endl;

	fin.clear();
	fin.seekg(0, ios::beg);

	getline(fin, line);

	int source, destin, value;
	while(getline(fin, line))
	{
		ss.str(line);
		getline(ss, line, ',');
		source = atoi(line.c_str());

		getline(ss, line, ',');
		destin = atoi(line.c_str());

		getline(ss, line, ',');
		getline(ss, line, ',');
		getline(ss, line, ',');

		getline(ss, line, ',');
		value = atoi(line.c_str());

		distanceMatrix(source-minNode, destin-minNode) = value;

		ss.str("");
	}

	fin.close();

	std::cout << "Finished reading file..." << std::endl;
}


/* print group information */
void IOHandler::printGroup(const std::vector<int>& group)
{
	assert(!group.empty());

	ofstream fout("../test_data/group.txt", ios::out);
	if(!fout)
	{
		std::cout << "Error printing group information!" << std::endl;
		exit(1);
	}

	for (int i = 0; i < group.size(); ++i)
	{
		fout << group[i] << std::endl;
	}

	fout.close();
}


/* read txt file into matrixXf */
void IOHandler::readTXT(const char* txtName,
						int& numOfNodes,
						Eigen::MatrixXf& distanceMatrix)
{
	std::cout << "Start reading data..." << std::endl;

	stringstream ss;
	ss << "../test_data/" << txtName;

	std::ifstream fin(ss.str().c_str(), ios::in);
	if(!fin)
	{
		std::cout << "Error for reading the file!" << std::endl;
		exit(1);
	}

	ss.str("");
	ss.clear();

	string line;

	int tempIndex;

	std::unordered_set<int> mySet;
	while(getline(fin,line))
	{
		ss.str(line);
		getline(ss,line,',');

		tempIndex = std::atoi(line.c_str());
		mySet.insert(tempIndex);

		getline(ss,line,',');

		tempIndex = std::atoi(line.c_str());
		mySet.insert(tempIndex);

		ss.str("");
	}

	numOfNodes = mySet.size();

	std::cout << "It has totally " << numOfNodes << " vertices!" << std::endl;

	/* go to first line of the file */
	fin.clear();
	fin.seekg(0, ios::beg);

	distanceMatrix = Eigen::MatrixXf::Zero(numOfNodes,numOfNodes);

	int left, right;
	float dist;
	while(getline(fin,line))
	{
		ss.str(line);

		getline(ss,line,',');
		left = std::atoi(line.c_str());

		getline(ss,line,',');
		right = std::atoi(line.c_str());

		getline(ss,line,',');
		dist = std::atof(line.c_str());

		distanceMatrix(left,right) = dist;

		distanceMatrix(right,left) = dist;

		ss.str("");
		ss.clear();
	}

	fin.close();

	std::cout << "Finish reading data..." << std::endl;
}


/* read txt point  into matrixXf */
void IOHandler::readPoint(const char* txtName,
						  string& name,
						  Eigen::MatrixXf& coordinates,
						  int& numOfNodes,
						  Eigen::MatrixXf& distanceMatrix)
{
	stringstream ss;
	ss << txtName;

	getline(ss,name,'.');
	std::cout << name << std::endl;


	std::ifstream fin((string("../test_data/")+string(txtName)).c_str(), ios::in);
	if(!fin)
	{
		std::cout << "Error for reading the file!" << std::endl;
		exit(1);
	}
	string line, subcomponent;

	ss.str("");
	ss.clear();

	std::vector<Eigen::Vector3f> pointVec;

	Eigen::Vector3f point;
	/* should store coordinates into coordinates */
	while(getline(fin,line))
	{
		ss.str(line);

		ss>>subcomponent;
		point(0) = std::atof(subcomponent.c_str()); 

		ss>>subcomponent;
		point(1) = std::atof(subcomponent.c_str()); 

		ss>>subcomponent;
		point(2) = std::atof(subcomponent.c_str()); 

		pointVec.push_back(point);

		ss.str("");
		ss.clear();
	}

	numOfNodes = pointVec.size();

	coordinates = Eigen::MatrixXf(numOfNodes, point.size());

	distanceMatrix = Eigen::MatrixXf::Zero(numOfNodes,numOfNodes);

#pragma omp parallel for schedule(dynamic) num_threads(8)
	for (int i = 0; i < numOfNodes; ++i)
	{
		coordinates.row(i) = pointVec[i];

		for (int j = 0; j < numOfNodes; ++j)
		{
			distanceMatrix(i,j) = (pointVec[i]-pointVec[j]).norm();
		}
	}

	pointVec.clear();
}


/* void print vtk file */
void IOHandler::printVTK(const int& numOfNodes,
					     const Eigen::MatrixXf& coordinates,
					     const std::vector<int>& group,
					     const string& name,
					     const string& groupName)
{

	const string testString = string("../test_data/")+name+string("_clustering.vtk");

	ifstream fin(testString.c_str());

	/* file is not existing */
	if(fin.fail())

	{
		fin.close();

		ofstream fout(testString.c_str(), ios::out);
		if (fout.fail())
		{
			cout << "Error creating the file!" << endl;
			exit(1);
		}

		fout << "# vtk DataFile Version 3.0" << endl;
		fout << "point_cloud example" << endl;
		fout << "ASCII" << endl;
		fout << "DATASET UNSTRUCTURED_GRID" << endl;
		fout << "POINTS " << numOfNodes << " float" << endl;

		Eigen::Vector3f temp;
		for( int k = 0; k < numOfNodes; ++k)
		{
			temp = coordinates.row(k);
			fout << temp(0) << " " << temp(1) << " " << temp(2) << endl; 	 
		}

		fout << "CELLS " << numOfNodes << " " << 2*numOfNodes << endl;
		for (int k = 0; k < numOfNodes; k++)
		{
			fout << 1 << " " << k << endl;
		}

		fout << "CELL_TYPES " << numOfNodes << endl;
		for (int k = 0; k < numOfNodes; k++)
		{
			fout << 1 << endl;
		}

		fout << "POINT_DATA " << numOfNodes << endl;
		fout << "SCALARS " << groupName << "_group int 1" << endl;
		fout << "LOOKUP_TABLE groupTable" << endl;
		for ( int k = 0; k < numOfNodes; k++)
		{
			fout << group[k] << endl;
		}	
		fout.close();
	}

	/* file does exist */
	else
	{
		fin.close();

		ofstream fout(testString.c_str(), ios::out|ios::app);
		if (fout.fail())
		{
			cout << "Error creating the file!" << endl;
			exit(1);
		}

		fout << "SCALARS " << groupName << "_group int 1" << endl;
		fout << "LOOKUP_TABLE groupTable" << endl;
		for ( int k = 0; k < numOfNodes; k++)
		{
			fout << group[k] << endl;
		}	
		fout.close();
	}

}


/* print information into README */
void IOHandler::writeReadMe(const Analysis& analysis, const string& dataSet, const string& clustering)
{
	std::ofstream out_file("../test_data/README", ios::out|ios::app);
	if (!out_file)
	{
		std::cout << "Error for creating README!" << std::endl;
		exit(1);
	}

	out_file << clustering << " on dataset " << dataSet << " has following measurements: " << std::endl;
	out_file << "Silhouette is " << analysis.getSilhouette() << ", DB-Index is " << analysis.getDBIndex() 
	         << ", gamma is " << analysis.getGamma() << std::endl;
	out_file << std::endl;
	out_file.close();
}