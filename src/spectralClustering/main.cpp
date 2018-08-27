#include "SpectralClustering.h"

/* in case of running for many norm, would enable automatic parameter choice */
void setPara(Para& p);


int main(int argc, char **argv)
{
	Para p;

	setPara(p);

	/* enable automatic option */
	bool automatic = true;

	SpectralClustering spectClus(argc, argv, p, automatic);

	spectClus.performClustering(p.numberOfClusters);

	return 0;
}



void setPara(Para& p)
{
	/* Laplacian option: 1.Normalized Laplacian, 2.Unsymmetric Laplacian */
	p.LaplacianOption = 1;

	/* local scaling by sorted distance: true, false */
	p.isDistSorted = true;

	/* preset number of clusters */
	std::cout << "Input a preset cluster numbers: " << std::endl;
	std::cin >> p.numberOfClusters;

	/* post-processing method: 1.k-means, 2.eigenvector rotation*/
	p.postProcessing = 1;

	/* derivative method for eigen rotation: 1.numerical derivative, 2.true derivative */
	p.mMethod = 2;
}
