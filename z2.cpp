#include "stdafx.h"

#include <omp.h>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <ctime>

using namespace std;

inline float function(float x)
{
	return (((3*x*x*x) + (2*x*x) + (x)) / ((x*x)-1));
}

inline void getIntegralReach(float* a, float* b)
{
	cout << "Enter a (float):\n>";
	cin >> *a;
	cout << "Enter b (float):\n>";
	cin >> *b;
}

inline void fillRandomsList(float* randomsPtr, int n)
{
	int i = 0;
 
	#pragma omp parallel for shared(randomsPtr, n) private(i)
	for (i = 0; i < n; i++)
	{
		randomsPtr[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);;
	}
}

inline float countPart(float a, float b, float u)
{
	return a + (b - a)*u;
}

inline void fillSumPartsList(vector<float> randoms, float* sumPartsPtr, float a, float b)
{
	int i = 0;

	#pragma omp parallel for shared(sumPartsPtr,randoms, a, b) private(i)
	for (i = 0; i < randoms.size(); i++)
		sumPartsPtr[i] = (countPart(a, b, randoms[i]));
}

inline float countIntegralApproximation(vector<float> sumParts, float a, float b)
{
	float integralApprox = 0.0;
	int i = 0;
	int counter = 0;

	#pragma omp parallel for shared(counter, sumParts) private(i) reduction(+:integralApprox)
	for (i = 0; i < sumParts.size(); i++)
		integralApprox += (function(sumParts[i]) / static_cast <float> (sumParts.size()));

	integralApprox *= static_cast <float> (b - a);
	
	return integralApprox;
}

int main()
{
	float a = 0.0, b = 0.0;
	auto n = 10000000;
	clock_t start, end;
	vector<float> randoms(n);
	vector<float> sumParts(n);

	float* randomsPtr = &randoms[0];
	float* sumPartsPtr = &sumParts[0];

	getIntegralReach(&a, &b);

	//1 thread
	omp_set_num_threads(1);

	start = clock();

	fillRandomsList(randomsPtr, n);
	fillSumPartsList(randoms, sumPartsPtr, a, b);
	cout << "Result: " << countIntegralApproximation(sumParts, a, b) << endl;

	end = clock();

	cout << "The task took " << end - start << " ms for 1 thread." << endl;

	//2 threads
	omp_set_num_threads(2);

	start = clock();

	fillRandomsList(randomsPtr, n);
	fillSumPartsList(randoms, sumPartsPtr, a, b);
	cout << "Result: " << countIntegralApproximation(sumParts, a, b) << endl;

	end = clock();

	cout << "The task took " << end - start << " ms for 2 threads." << endl;

	//4 threads
	omp_set_num_threads(4);

	start = clock();

	fillRandomsList(randomsPtr, n);
	fillSumPartsList(randoms, sumPartsPtr, a, b);
	cout << "Result: " << countIntegralApproximation(sumParts, a, b) << endl;

	end = clock();

	cout << "The task took " << end - start << " ms for 4 threads." << endl;

	//6 threads
	omp_set_num_threads(6);

	start = clock();

	fillRandomsList(randomsPtr, n);
	fillSumPartsList(randoms, sumPartsPtr, a, b);
	cout << "Result: " << countIntegralApproximation(sumParts, a, b) << endl;

	end = clock();

	cout << "The task took " << end - start << " ms for 6 threads." << endl;

	system("PAUSE");


    return 0;
}