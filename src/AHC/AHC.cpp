#include "AHC.h"


// remove two elements in template vector
template <class T>
void deleteVecElements(std::vector<T>& original, const T& first, const T& second)
{
	std::size_t size = original.size();
	assert(size>2);
	vector<T> result(size-2);
	int tag = 0;
	for(int i=0;i<size;++i)
	{
		//meet with target elements, not copied
		if(original[i]==first || original[i]==second)
			continue;
		result[tag++]=original[i];
	}
	assert(tag==size-2);
	original = result;
}


AHC::AHC()
{

}

/* argument constructor */
AHC::AHC(const int& argc, char* argv[])
{
	if(argc!=2)
	{
		std::cout << "Error for argument input! Should be ./AHC dataset" << std::endl;
		exit(1);
	}

	IOHandler::readPoint(argv[1], name, coordinates, numOfNodes, distanceMatrix);

	group = std::vector<int>(numOfNodes);

	std::cout << "Input preset number of clusters: " << std::endl;
	std::cin >> numOfClusters;
	assert(numOfClusters>0 && numOfClusters<=numOfNodes);

	std::cout << "Choose linkage type: 1.single, 2.complete, 3.average: " << std::endl;
	std::cin >> linkageType;

	assert(linkageType==1||linkageType==2||linkageType==3);
}


/* destructor */	
AHC::~AHC()
{

}



/* run k-means clustering */
void AHC::performClustering()
{
	std::unordered_map<int, Ensemble> nodeMap;
	std::vector<DistNode> dNodeVec;
	std::vector<Ensemble> nodeVec;

	setValue(dNodeVec);

	hierarchicalMerging(nodeMap, dNodeVec, nodeVec);

	string groupName;

	if(linkageType==1)
		groupName = "ahc_single";
	else if(linkageType==2)
		groupName = "ahc_complete";
	else if(linkageType==3)
		groupName = "ahc_average";

	IOHandler::printVTK(numOfNodes, coordinates, group, name, groupName);

	Analysis analysis;
	analysis.computeValue(coordinates, distanceMatrix, group);
	std::cout << "Silhouette is " << analysis.getSilhouette() << ", db index is " << analysis.getDBIndex()
	          << ", gamma statistics is " << analysis.getGamma() << std::endl;

	ValidityMeasurement vm;
	vm.computeValue(distanceMatrix, group);

	IOHandler::writeReadMe(analysis, name , groupName);

	IOHandler::writeReadMe(vm.f_c, name, groupName, "validity measurement");
}



/* set a vector for min-heap */
void AHC::setValue(std::vector<DistNode>& dNodeVec)
{
	dNodeVec = std::vector<DistNode>(numOfNodes*(numOfNodes-1)/2);
	int tag = 0;
	for(int i=0;i<numOfNodes-1;++i)
	{
		for(int j=i+1;j<numOfNodes;++j)
		{
			dNodeVec[tag].first = i;
			dNodeVec[tag].second = j;
			dNodeVec[tag].distance = distanceMatrix(i,j);
			++tag;
		}
	}
	assert(tag==dNodeVec.size());
}



/* perform AHC merging by given a distance threshold */
void AHC::hierarchicalMerging(std::unordered_map<int, Ensemble>& nodeMap, std::vector<DistNode>& dNodeVec,
							  std::vector<Ensemble>& nodeVec)
{
	/* would store distance matrix instead because it would save massive time */
	struct timeval start, end;
	double timeTemp;
	gettimeofday(&start, NULL);

	for(int i=0;i<numOfNodes;++i)
	{
		nodeMap[i].element.push_back(i);
	}

	DistNode poped;

	/* find node-pair with minimal distance */
	float minDist = FLT_MAX;
	int target = -1;
	for (int i = 0; i < dNodeVec.size(); ++i)
	{
		if(dNodeVec[i].distance<minDist)
		{
			target = i;
			minDist = dNodeVec[i].distance;
		}
	}
	poped = dNodeVec[target];

	int index = numOfNodes, currentNumber;
	do
	{
		//create new node merged and input it into hash map
		vector<int> first = (nodeMap[poped.first]).element;
		vector<int> second = (nodeMap[poped.second]).element;

		/* index would be starting from numOfNodes */
		Ensemble newNode(index);
		newNode.element = first;
		newNode.element.insert(newNode.element.end(), second.begin(), second.end());
		nodeMap.insert(make_pair(index, newNode));

		//delete two original nodes
		nodeMap.erase(poped.first);
		nodeMap.erase(poped.second);

		/* the difficulty lies how to update the min-heap with linkage
		 * This would take 2NlogN.
		 * Copy all node-pairs that are not relevant to merged nodes to new vec.
		 * For relevant, would update the mutual distance by linkage
		 */

		/* how many clusters exist */
		currentNumber = nodeMap.size();

		target = -1, minDist = FLT_MAX;

		std::vector<DistNode> tempVec(currentNumber*(currentNumber-1)/2);
		int current = 0, i_first, i_second;
		for(int i=0;i<dNodeVec.size();++i)
		{
			i_first=dNodeVec[i].first, i_second=dNodeVec[i].second;
			/* not relevant, directly copied to new vec */
			if(i_first!=poped.first&&i_first!=poped.second&&i_second!=poped.first&&i_second!=poped.second)
			{
				tempVec[current]=dNodeVec[i];
				if(tempVec[current].distance<minDist)
				{
					target = current;
					minDist = tempVec[current].distance;
				}
				++current;
			}
		}

		for (auto iter=nodeMap.begin();iter!=nodeMap.end();++iter)
		{
			if((*iter).first!=newNode.index)
			{
				tempVec[current].first = (*iter).first;
				tempVec[current].second = newNode.index;
				tempVec[current].distance=getDistAtNodes(newNode.element,(*iter).second.element);
				if(tempVec[current].distance<minDist)
				{
					target = current;
					minDist = tempVec[current].distance;
				}
				++current;
			}
		}

		poped = tempVec[target];

		/* judge whether current is assigned to right value */
		assert(current==tempVec.size());
		dNodeVec.clear();
		dNodeVec = tempVec;
		tempVec.clear();
		++index;
	}while(nodeMap.size()!=numOfClusters);	//merging happens whenever requested cluster is not met

	nodeVec=std::vector<Ensemble>(nodeMap.size());
	int tag = 0, clusSize;

	std::cout << "Final cluster number is " << nodeMap.size() << std::endl;
/* assign label to each object inside one cluster */
	std::vector<int> eachCluster;
	for(auto iter=nodeMap.begin();iter!=nodeMap.end();++iter)
	{
		nodeVec[tag]=(*iter).second;
		eachCluster = nodeVec[tag].element;
		clusSize = eachCluster.size();
		std::cout << "cluster " << tag << " has " << clusSize << " elements!" << std::endl;
		for (int i = 0; i < clusSize; ++i)
		{
			group[eachCluster[i]] = tag;
		}
		++tag;
	}

	gettimeofday(&end, NULL);
	timeTemp = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;

	activityList.push_back("Hirarchical clustering for "+
			               to_string(numOfClusters)+" groups takes: ");
	timeList.push_back(to_string(timeTemp)+" s");
	/* task completed, would delete memory contents */
	dNodeVec.clear();
	nodeMap.clear();
	/* use alpha function to sort the group by its size */
	std::sort(nodeVec.begin(), nodeVec.end(), [](const Ensemble& e1, const Ensemble& e2)
	{return e1.element.size()<e2.element.size() ||(e1.element.size()==e2.element.size()&&e1.index<e2.index);});
}



const float AHC::getDistAtNodes(const vector<int>& firstList, const vector<int>& secondList)
{
	const int& m = firstList.size();
	const int& n = secondList.size();
	assert(m!=0);
	assert(n!=0);
	/* 0: single linkage, min(x_i,y_j)
	 * 1: complete linkdage, max(x_i,y_j)
	 * 2: average linkage, sum/x_i/y_j
	 */
	float result, value;
	switch(linkageType)
	{
	case 1:	//single linkage
		{
			result = FLT_MAX;
		#pragma omp parallel for reduction(min:result) num_threads(8)
			for(int i=0;i<m;++i)
			{
				for(int j=0;j<n;++j)
				{
					value = distanceMatrix(firstList[i],secondList[j]);
					result = std::min(result, value);
				}
			}
		}
		break;

	case 2:	//complete linkage
		{
			result = FLT_MIN;
		#pragma omp parallel for reduction(max:result) num_threads(8)
			for(int i=0;i<m;++i)
			{
				for(int j=0;j<n;++j)
				{
					value = distanceMatrix(firstList[i],secondList[j]);
					result = std::max(result, value);
				}
			}
		}
		break;

	/* average linkage */
	case 3:
		{
			result = 0;
		#pragma omp parallel for reduction(+:result) num_threads(8)
			for(int i=0;i<m;++i)
			{
				for(int j=0;j<n;++j)
				{
					value = distanceMatrix(firstList[i],secondList[j]);
					result+=value;
				}
			}
			result/=m*n;
		}
		break;

	default:
		std::cout << "error!" << std::endl;
		exit(1);
	}
	return result;
}
