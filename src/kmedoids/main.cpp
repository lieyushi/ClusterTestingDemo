#include "Kmedoids.h"

int main(int argc, char* argv[])
{
	Kmedoids kmedoids(argc, argv);

	kmedoids.performClustering();

	return 0;
}