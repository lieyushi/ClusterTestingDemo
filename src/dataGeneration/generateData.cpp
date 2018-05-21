#include <iostream>
#include <math.h>
#include <fstream>
#include <vector>
#include <stdlib.h> 
#include <time.h>       /* time */
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

using namespace std;
using namespace Eigen;


typedef struct Point
{
	float x, y, z;
	Point(const float& x, const float& y, const float& z): x(x), y(y), z(z)
	{} 
} point;


void printVec(const vector<point>& array, const char* name);

void generateRings3();

void generateRings4();

void generateRings2();

void generateTwoGaussinAndElliptic();

void generateDoubleRings();

void generateTwoGaussin();

void generateThreeBlobs();

void generateThreeSemicircles();

int main()
{
	generateRings2();
	generateRings3();
	generateRings4();
	generateDoubleRings();
	generateTwoGaussin();
	generateTwoGaussinAndElliptic();
	generateThreeBlobs();
	generateThreeSemicircles();
	return 0;
}

void generateRings3()
{
	std::vector<point> pointVec;
	const float& zValue = 3.0;
	const float& Radius = 10.0;
	const float& radius = 5.0;
	const float& radius2 = 3.0;

	const int& Total = 1000;
	const float& sub = 2*M_PI/Total;

	float theta;
	for (int i = 0; i < Total; ++i)
	{
		theta = i*sub;
		pointVec.push_back(point(Radius*cos(theta), Radius*sin(theta), zValue));
		pointVec.push_back(point(radius*cos(theta), radius*sin(theta), zValue));
		pointVec.push_back(point(radius2*cos(theta), radius2*sin(theta), zValue));
	}

	printVec(pointVec, "rings3.txt");
}


void generateRings4()
{
	std::vector<point> pointVec;
	const float& zValue = 3.0;
	const float& Radius = 10.0;
	const float& radius = 7.0;
	const float& radius2 = 4.0;
	const float& radius3 = 1.0;

	const int& Total = 1000;
	const float& sub = 2*M_PI/Total;

	float theta;
	for (int i = 0; i < Total; ++i)
	{
		theta = i*sub;
		pointVec.push_back(point(Radius*cos(theta), Radius*sin(theta), zValue));
		pointVec.push_back(point(radius*cos(theta), radius*sin(theta), zValue));
		pointVec.push_back(point(radius2*cos(theta), radius2*sin(theta), zValue));
		pointVec.push_back(point(radius3*cos(theta), radius3*sin(theta), zValue));
	}

	printVec(pointVec, "rings4.txt");
}


void generateRings2()
{
	std::vector<point> pointVec;
	const float& zValue = 3.0;
	const float& Radius = 10.0;
	const float& radius = 5.0;

	const int& Total = 1000;
	const float& sub = 2*M_PI/Total;

	float theta;
	for (int i = 0; i < Total; ++i)
	{
		theta = i*sub;
		pointVec.push_back(point(Radius*cos(theta), Radius*sin(theta), zValue));
		pointVec.push_back(point(radius*cos(theta), radius*sin(theta), zValue));
	}

	printVec(pointVec, "rings2.txt");
}


void generateDoubleRings()
{
	std::vector<point> pointVec;
	const float& zValue = 3.0;
	const float& Radius = 9.0;
	const float& radius = 3.0;
	const float& aberration = 2.0;
	/* want to make the radius to [9.0,11.0] and [3.0,5.0] */ 

	const int& Total = 1000;
	const float& sub = 2*M_PI/Total;

	srand (time(NULL));

	float theta, longRadius, shortRadius, coefficient;
	for (int i = 0; i < Total; ++i)
	{
		theta = i*sub;
		coefficient = ((double) rand() / (RAND_MAX));
		longRadius = Radius+aberration*coefficient;
		shortRadius = radius+aberration*coefficient;
		pointVec.push_back(point(longRadius*cos(theta), longRadius*sin(theta), zValue));
		pointVec.push_back(point(shortRadius*cos(theta), shortRadius*sin(theta), zValue));
	}

	printVec(pointVec, "twoAnnuals.txt");
}


void printVec(const vector<point>& array, const char* name)
{
	ofstream fout((string("../test_data/")+string(name)).c_str(), ios::out);
	if(!fout)
	{
		std::cout << "Error writing the file!" << std::endl;
		exit(1);
	}
	const std::size_t& size = array.size();
	for(int i=0;i<size;++i)
	{
		fout << array[i].x << " " << array[i].y << " " << array[i].z << std::endl;
	}
	fout.close();
}


void generateTwoGaussinAndElliptic()
{
	std::vector<point> pointVec;
	const float& zValue = 3.0;
	const float& radius = 2.0;
	const float& Radius = 4.0;
	const float& elip_r1 = 2.0;
	const float& elip_r2 = 8.0;

	point center_1(0,0,3);

	point center_2(10,10,3);

	point center_3(5,5,3);


	Eigen::Matrix3f rotationMatrix;
	rotationMatrix << sqrt(2)/2.0, -sqrt(2)/2.0, 0, sqrt(2)/2.0, sqrt(2)/2.0, 0, 0, 0, 1.0;

	const int& Total = 1000;
	const float& sub = 2*M_PI/Total;

	srand (time(NULL));

	float theta, longRadius, shortRadius, coefficient, r1, r2;
	Eigen::Vector3f coordinates;
	for (int i = 0; i < Total; ++i)
	{
		theta = i*sub;
		coefficient = ((double) rand() / (RAND_MAX));
		longRadius = Radius*coefficient;
		shortRadius = radius*coefficient;
		pointVec.push_back(point(longRadius*cos(theta)+center_2.x, longRadius*sin(theta)+center_2.y, zValue));
		pointVec.push_back(point(shortRadius*cos(theta)+center_1.x, shortRadius*sin(theta)+center_1.y, zValue));

		/* elliptic shape of cloud points */
		r1 = elip_r1*coefficient;
		r2 = elip_r2*coefficient;

		coordinates << r1*cos(theta)+center_3.x, r2*sin(theta)+center_3.y, zValue;

		coordinates = rotationMatrix*coordinates;

		pointVec.push_back(point(coordinates(0),coordinates(1),coordinates(2)));
	}

	printVec(pointVec, "twoGaussianElliptic.txt");
}


void generateTwoGaussin()
{
	std::vector<point> pointVec;
	const float& zValue = 3.0;
	const float& radius = 2.0;
	const float& Radius = 4.0;
	const float& elip_r1 = 2.0;
	const float& elip_r2 = 8.0;

	point center_1(0,0,3);

	point center_2(10,10,3);

	point center_3(5,5,3);


	Eigen::Matrix3f rotationMatrix;
	rotationMatrix << sqrt(2)/2.0, -sqrt(2)/2.0, 0, sqrt(2)/2.0, sqrt(2)/2.0, 0, 0, 0, 1.0;

	const int& Total = 1000;
	const float& sub = 2*M_PI/Total;

	srand (time(NULL));

	float theta, longRadius, shortRadius, coefficient;
	Eigen::Vector3f coordinates;
	for (int i = 0; i < Total; ++i)
	{
		theta = i*sub;
		coefficient = ((double) rand() / (RAND_MAX));
		longRadius = Radius*coefficient;
		shortRadius = radius*coefficient;
		pointVec.push_back(point(longRadius*cos(theta)+center_2.x, longRadius*sin(theta)+center_2.y, zValue));
		pointVec.push_back(point(shortRadius*cos(theta)+center_1.x, shortRadius*sin(theta)+center_1.y, zValue));
	}

	printVec(pointVec, "twoGaussians.txt");
}


void generateThreeBlobs()
{
	std::vector<point> pointVec;
	const float& zValue = 3.0;
	const float& radius = 1.0;
	const float& Radius = 4.0;

	const float& elip_r1 = 2.0;
	const float& elip_r2 = 8.0;

	point center_1(0,0,zValue);
	point center_2(6.5,-1.5,zValue);

	Eigen::Matrix3f rotationMatrix;
	rotationMatrix << sqrt(2.0)/2.0, -sqrt(2.0)/2.0, 0, sqrt(2.0)/2.0, sqrt(2.0)/2.0, 0, 0, 0, 1.0;

	const int& Total = 500;
	const float& sub = 2*M_PI/Total;

	srand (time(NULL));

	float theta, longRadius, shortRadius, coefficient;
	Eigen::Vector3f coordinates;
	for (int i = 0; i < Total; ++i)
	{
		theta = i*sub;
		coefficient = ((double) rand() / (RAND_MAX));

		longRadius = Radius*coefficient;
		shortRadius = radius*coefficient;

		coordinates << shortRadius*cos(theta)+center_2.x, longRadius*sin(theta)+center_2.y, zValue;
		coordinates = rotationMatrix*coordinates;
		pointVec.push_back(point(coordinates(0),coordinates(1),coordinates(2)));

		coordinates = coordinates+Eigen::Vector3f(-5.0,5.0,zValue);
		pointVec.push_back(point(coordinates(0),coordinates(1),coordinates(2)));

		longRadius = elip_r2*coefficient;
		shortRadius = elip_r1*coefficient;
		coordinates << shortRadius*cos(theta)+center_1.x, longRadius*sin(theta)+center_1.y, zValue;
		coordinates = rotationMatrix*coordinates;
		pointVec.push_back(point(coordinates(0),coordinates(1),coordinates(2)));
	}

	printVec(pointVec, "threeBlobs.txt");
}


void generateThreeSemicircles()
{
	std::vector<point> pointVec;
	const float& zValue = 3.0;
	const float& radius = 1.0;
	const float& Radius = 4.0;

	const float& elip_r1 = 3.5;
	const float& elip_r2 = 5.0;
	const float& error = 0.5;

	point center_1(0,0,zValue);
	point center_2(5.0,0,zValue);
	point center_3(10.0,0,zValue);

	const int& Total = 500;
	const float& sub = M_PI/Total;

	srand (time(NULL));

	float theta, currentRadius, coefficient;
	Eigen::Vector3f coordinates;
	for (int i = 0; i < Total; ++i)
	{
		theta = i*sub;
		coefficient = ((double) rand() / (RAND_MAX));

		currentRadius = elip_r1+error*coefficient;
		pointVec.push_back(point(currentRadius*cos(theta)+center_1.x,currentRadius*sin(theta)+center_1.y,zValue));

		pointVec.push_back(point(currentRadius*cos(theta)+center_3.x,currentRadius*sin(theta)+center_3.y,zValue));

		currentRadius = elip_r2+error*coefficient;
		theta = theta+M_PI;
		pointVec.push_back(point(currentRadius*cos(theta)+center_2.x,currentRadius*sin(theta)+center_2.y,zValue));

	}

	printVec(pointVec, "threeSemicircles.txt");
}