#include <iostream>
#include <math.h>
#include <fstream>
#include <vector>
#include <stdlib.h> 
#include <time.h>       /* time */

using namespace std;

typedef struct Point
{
	float x, y, z;
	Point(const float& x, const float& y, const float& z): x(x), y(y), z(z)
	{} 
} point;

void printVec(const vector<point>& array, const char* name);

void generateRings();

void generateDoubleRings();

int main()
{
	generateRings();
	generateDoubleRings();
	return 0;
}

void generateRings()
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

	printVec(pointVec, "rings.txt");
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

	printVec(pointVec, "doubleRings.txt");
}


void printVec(const vector<point>& array, const char* name)
{
	ofstream fout(name, ios::out);
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