#include "DBSCAN.h"

int main(int argc, char* argv[])
{
	DBSCAN dbscan(argc, argv);

	dbscan.performClustering();

	return 0;
}