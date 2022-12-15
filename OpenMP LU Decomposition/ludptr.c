#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int num_threads=1;
int display = 0;

void print(int n, double *arr[n])
{
	int i, j;
	for (i = 0; i < n; i++){
		for (j = 0; j < n; j++)
			// printf("%f\t", *((arr+i*n) + j));
			printf("%f\t", arr[j][i]);
		printf("\n");
	}
}

void mat_mult(int n, double *A[n], double *B[n], double *result[n]) {
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			// *(result+i*n+j) = 0;
			result[j][i] = 0;
			for (int k = 0; k < n; k++)
				result[j][i] += A[k][i] * B[j][k];
		}
	}
}

int main(int argc, char const *argv[])
{
	int n=3;
	
	if (argc>=2)
		n=atoi(argv[1]);
	if (argc>=3)
		num_threads=atoi(argv[2]);
	if (argc>=4)
		display = atoi(argv[3]);


	printf("Using size=%d and threads=%d\n", n, num_threads);
	

	double *a[n], *a_copy[n], *u[n], *l[n];
	int pi[n];

	// Sequential
	// srand(time(NULL));
	clock_t seq_start = clock();
	for (int j = 0; j < n; j++)
	{
		pi[j] = j;
		a[j] = (double*) malloc(sizeof(double)*n);
		a_copy[j] = (double*) malloc(sizeof(double)*n);
		u[j] = (double*) malloc(sizeof(double)*n);
		l[j] = (double*) malloc(sizeof(double)*n);

		for (int i = 0; i < n; i++)
		{
			a[j][i] = rand()/(double)RAND_MAX;
			a_copy[j][i] = a[j][i];
			// printf("%f\t",a[j][i]);
			if (j<i) {
				u[j][i] = 0;
				l[j][i] = 0;
			}
			else if (j>i) {
				u[j][i] = 0;
				l[j][i] = 0;
			}
			else {
				u[j][i] = 0;
				l[j][i] = 1;
			}
		}
		// printf("\n");
	}

	if (display==1) {
		printf("Matrix a:\n");
		print(n, a);
		// printf("Matrix l:\n");
		// print(n, l);
		// printf("Matrix u:\n");
		// print(n, u);
	}

	for (int k = 0; k < n; k++) {
		double max = 0;
		int k_swap;
		for (int i = k; i < n; i++)
		{
			if (max < fabs(a[k][i])) {
				max = fabs(a[k][i]);
				k_swap = i;
			}
		}

		if (max == 0) {
			printf("Error: the matrix is singular\n");
			exit(0);
		}

		if (k != k_swap) {
			// swap π[k] and π[k']
			int tempi = pi[k];
			pi[k] = pi[k_swap];
			pi[k_swap] = tempi;

			// swap a(k,:) and a(k',:)
			double temp;
			for (int j = 0; j < n; j++)
			{
				temp = a[j][k];
				a[j][k] = a[j][k_swap];
				a[j][k_swap] = temp;

				temp = a_copy[j][k];
				a_copy[j][k] = a_copy[j][k_swap];
				a_copy[j][k_swap] = temp;
			}

			// swap l(k,1:k-1) and l(k',1:k-1)
			for (int j = 0; j < k; j++)
			{
				temp = l[j][k];
				l[j][k] = l[j][k_swap];
				l[j][k_swap] = temp;
			}
		}

		u[k][k] = a[k][k];
		for (int i = k+1; i < n; i++)
		{
			l[k][i] = a[k][i]/u[k][k];
			u[i][k] = a[i][k];
		}
		for (int j = k+1; j < n; j++)
			for (int i = k+1; i < n; i++)
				a[j][i] -= l[k][i]*u[j][k];
	}

	if (display==1) {
		// printf("Matrix a:\n");
		// print(n, a);
		printf("Matrix l:\n");
		print(n, l);
		printf("Matrix u:\n");
		print(n, u);
	}
	clock_t seq_end = clock();
	printf("LU Decomposition took: %f s\n", (float)(seq_end - seq_start)/CLOCKS_PER_SEC);

	// Verification
	// double LU[n][n];
	double *LU[n];
	for (int i = 0; i < n; i++)
		LU[i] = (double*) malloc(sizeof(double)*n);

	mat_mult(n, l, u, LU);

	if (display==1) {
		printf("PA:\n");
		print(n, a_copy);
		printf("LU:\n");
		print(n, LU);
	}
	double verif_sum = 0.0; 
	for (int j = 0; j < n; j++)
	{	
		double col_l2 = 0.0;
		for (int i = 0; i < n; i++)
		{
			col_l2 += pow(a_copy[j][i] - LU[j][i] , 2);
		}
		verif_sum += sqrt(col_l2);
	}

	printf("Verification -> L_2,1 norm = %f\n", verif_sum);

	// Parallel
	return 0;
}