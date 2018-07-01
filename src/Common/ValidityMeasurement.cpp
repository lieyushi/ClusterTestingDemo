/*
 * ValidityMeasurement.cpp
 *
 *  Created on: Jun 24, 2018
 *      Author: lieyu
 */

#include "ValidityMeasurement.h"

ValidityMeasurement::ValidityMeasurement() {
	// TODO Auto-generated constructor stub

}

ValidityMeasurement::~ValidityMeasurement() {
	// TODO Auto-generated destructor stub
}


// function API for computing the validity measurement given a distance matrix 
void ValidityMeasurement::computeValue(const Eigen::MatrixXf& distMatrix, const std::vector<int>& group)
{
		// get how many different groups it totally has
	int max_group = -1;
	const int& num_node = group.size();
	for(int i=0; i<num_node; ++i)
	{
		max_group = std::max(group[i], max_group);
	}
	max_group+=1;

	std::vector<std::vector<int> > storage(max_group);
	for(int i=0; i<num_node; ++i)
	{
		if(group[i]==-1)
			continue;
		storage[group[i]].push_back(i);
	}

	std::vector<std::tuple<float, float, float> > measureVec(max_group);
//#pragma omp parallel for schedule(dynamic) num_threads(8)
	for(int i=0; i<max_group; ++i)
	{
		getMST_Parent_Node_distanceMatrix(measureVec[i], storage[i], distMatrix);
	}

	float minSc = 0, maxSc = 0, aver_sigma = 0, std_sigma = 0, std_variance;
	for(int i=0; i<max_group; ++i)
	{
		// get the min Sc by summation
		minSc+=std::get<1>(measureVec[i]);
		// get the max Sc by summation
		maxSc+=std::get<2>(measureVec[i]);
		std_variance = std::get<0>(measureVec[i]);
		// get the average variance and standard variation of variance
		aver_sigma+=std_variance;
		std_sigma+=std_variance*std_variance;
	}
	aver_sigma/=float(max_group);
	std_sigma = std_sigma/float(max_group-1)-float(max_group)/float(max_group-1)*aver_sigma*aver_sigma;
	if(std_sigma <= 1.0e-10)
		std_sigma = 1.0e-10;
	std_sigma=sqrt(std_sigma);

	float h_DDc = aver_sigma+std_sigma;

	minSc/=float(max_group);
	maxSc/=float(max_group);

	// compute g1_Sc
	float g1_Sc = (1.0-minSc)*(1.0-maxSc);
	if(g1_Sc<0)
	{
		std::cout << "Negative number for g1_Sc computation!" << std::endl;
		exit(1);
	}
	g1_Sc = aver_sigma*sqrt(g1_Sc);

	// compute g2_Sc
	float g2_Sc = minSc*maxSc;
	if(g2_Sc<0)
	{
		std::cout << "Negative number for g2_Sc computation!" << std::endl;
		exit(1);
	}
	g2_Sc = aver_sigma/sqrt(g2_Sc);

	// compute g_Sc
	float g_Sc = (sqrt(g1_Sc*g2_Sc)+(g1_Sc+g2_Sc)/2.0)/2.0;

	// compoute f_c
	f_c = h_DDc*g_Sc;
}


// get MST for each cluster given index and pair-wise distance, PCA case only
void ValidityMeasurement::getMST_Parent_Node_distanceMatrix(std::tuple<float, float, float>& values,
			const std::vector<int>& clusterNode, const Eigen::MatrixXf& distMatrix)
{
	using namespace boost;
	typedef adjacency_list < vecS, vecS, undirectedS, no_property, property < edge_weight_t, float > > Graph;
	typedef graph_traits < Graph >::edge_descriptor Edge;
	typedef graph_traits < Graph >::vertex_descriptor Vertex;
	typedef std::pair<int, int> E;

	const int& num_nodes = clusterNode.size();
	// handle with singleton case
	if(num_nodes==1)
	{
		values = std::make_tuple(0.0,0.0,0.0);
		return;
	}

	const int num_edges = num_nodes*(num_nodes-1)/2;
	E *edge_array = new E[num_edges];
	float *weights = new float[num_edges];
	int temp = 0;
	for (int i = 0; i < num_nodes-1; ++i)
	{
		for (int j = i+1; j < num_nodes; ++j)
		{
			edge_array[temp] = std::make_pair(i,j);
			weights[temp] = distMatrix(clusterNode[i], clusterNode[j]);
			++temp;
		}
	}

#if defined(BOOST_MSVC) && BOOST_MSVC <= 1300
	Graph g(num_nodes);
	property_map<Graph, edge_weight_t>::type weightmap = get(edge_weight, g);
	for (std::size_t j = 0; j < num_edges; ++j) {
		Edge e; bool inserted;
		tie(e, inserted) = add_edge(edge_array[j].first, edge_array[j].second, g);
		weightmap[e] = weights[j];
	}
#else
	Graph g(edge_array, edge_array + num_edges, weights, num_nodes);
#endif
	property_map < Graph, edge_weight_t >::type weight = get(edge_weight, g);
	std::vector < Edge > spanning_tree;

	kruskal_minimum_spanning_tree(g, std::back_inserter(spanning_tree));

	if(edge_array!=NULL)
	{
		delete[] edge_array;
		edge_array = NULL;
	}

	if(weights!=NULL)
	{
		delete[] weights;
		weights = NULL;
	}

	// compute the standard deviation for the distance in MST
	float summation = 0.0, sq_summation = 0.0, average_mst_d, max_d_mst = -1.0, dist;
	const int& MST_EDGE_NUM = num_nodes-1;

	for (std::vector < Edge >::iterator ei = spanning_tree.begin(); ei != spanning_tree.end(); ++ei) 
	{
		dist = weight[*ei];
		max_d_mst = std::max(max_d_mst, dist);
		summation += dist;
		sq_summation += dist*dist;
	}

	float variance;

	if(MST_EDGE_NUM==1)
	{
		variance = 0;
		average_mst_d = summation;
	}
	else
	{
		average_mst_d = summation/float(MST_EDGE_NUM);
		variance = sq_summation/float(MST_EDGE_NUM-1)-average_mst_d*summation/float(MST_EDGE_NUM-1);
		if(variance <= 1.0e-10)
			variance = 1.0e-10;
		variance = sqrt(variance);
	}
	// compute the inner Sc value for this cluster
	int min_index, max_index;
	float min_Sc = get_Sc_by_range(distMatrix, clusterNode, max_d_mst, min_index);
	float max_Sc = get_Sc_by_range(distMatrix, clusterNode, average_mst_d, max_index);

	min_Sc/=float(min_index);
	max_Sc/=float(max_index);

	// store the standard deviation, min Sc and max Sc in the tuple
	values = std::make_tuple(variance, min_Sc, max_Sc);
}


// compute the Sc by input range value for general cases
const float ValidityMeasurement::get_Sc_by_range(const Eigen::MatrixXf& distMatrix, const std::vector<int>& clusterNode, 
												 const float& rangeValue, int& index)
{
	const int& node_number = clusterNode.size();
	float result = 0.0;

	index = 0;
	int inside_whole, inside_cluster;
	for(int i=0; i<node_number; ++i)
	{
		inside_whole = 0, inside_cluster = 0;
		// count how many points in N_epsi(P_i) for the whole dataset
	#pragma omp parallel num_threads(8)
		{
		#pragma omp for nowait
			for(int j=0; j<distMatrix.rows(); ++j)
			{
				// don't want to handle duplicates and itself
				if(clusterNode[i]==j)
					continue;
				float dist;
				dist = distMatrix(clusterNode[i], j);

			#pragma omp critical
				{
					if(dist<=rangeValue)
					{
						++inside_whole;
					}
				}
			}

		}

		// count how many points in N_epsi(P_i) for current cluster
	#pragma omp parallel num_threads(8)
		{
		#pragma omp for nowait
			for(int j=0; j<node_number; ++j)
			{
				// don't want to handle duplicates and itself
				if(i==j)
					continue;
				float dist;
				dist = distMatrix(clusterNode[i], clusterNode[j]);

			#pragma omp critical
				if(dist<=rangeValue)
					++inside_cluster;
			}

		}
		assert(inside_cluster<=inside_whole);
		if(inside_whole==0)
			continue;
		++index;

		float deviation = float(inside_cluster)/float(inside_whole);
		if(isnan(deviation))
		{
			std::cout << inside_cluster << ", " << inside_whole << std::endl;
			exit(1);
		}
		result+=float(inside_cluster)/float(inside_whole);
	}
	return result;
}

