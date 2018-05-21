#include "Kmeans.h"

int main(int argc, char* argv[])
{
	Kmeans kmeans(argc, argv);

	kmeans.performClustering();

	return 0;
}