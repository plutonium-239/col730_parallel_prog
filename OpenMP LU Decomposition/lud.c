#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

int num_threads=1;
int display = 0;

void print(double *arr, int n)
{
	int i, j;
	for (i = 0; i < n; i++){
		for (j = 0; j < n; j++)
			printf("%f\t", *((arr+i*n) + j));
			// printf("%f\t", arr[i*n + j]);
		printf("\n");
	}
}

void mat_mult(int n, double *A, double *B, double *result) {
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			*(result+i*n+j) = 0;
			for (int k = 0; k < n; k++)
				*(result+i*n+j) += *(A+i*n+k) * *(B+j+n*k);
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
	

	double a[n][n], a_copy[n][n], u[n][n], l[n][n], *a_ptr[n];
	int pi[n];

	// Sequential
	double seqso = omp_get_wtime();

	for (int i = 0; i < n; i++)
	{
		pi[i] = i;
		a_ptr[i] = (double*) malloc(sizeof(double)*n);
		for (int j = 0; j < n; j++)
		{
			a[i][j] = rand()/(double)RAND_MAX;
			a_copy[i][j] = a[i][j];
			a_ptr[i][j] = a[i][j];
			// printf("%f\t",a[i][j]);
			if (j<i)
				u[i][j] = 0;
			else if (j>i)
				l[i][j] = 0;
			else
				l[i][j] = 1;
		}
		// printf("\n");
	}

	if (display==1) {
		printf("Matrix a:\n");
		print(a, n);
		// printf("Matrix l:\n");
		// print(l, n);
		// printf("Matrix u:\n");
		// print(u, n);
	}

	for (int k = 0; k < n; k++) {
		double max = 0;
		int k_swap;
		for (int i = k; i < n; i++)
		{
			if (max < fabs(a[i][k])) {
				max = fabs(a[i][k]);
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
				temp = a[k][j];
				a[k][j] = a[k_swap][j];
				a[k_swap][j] = temp;

				temp = a_copy[k][j];
				a_copy[k][j] = a_copy[k_swap][j];
				a_copy[k_swap][j] = temp;
			}

			// swap l(k,1:k-1) and l(k',1:k-1)
			for (int j = 0; j < k; j++)
			{
				temp = l[k][j];
				l[k][j] = l[k_swap][j];
				l[k_swap][j] = temp;
			}
		}

		u[k][k] = a[k][k];
		for (int i = k+1; i < n; i++)
		{
			l[i][k] = a[i][k]/u[k][k];
			u[k][i] = a[k][i];
		}
		for (int j = k+1; j < n; j++)
			for (int i = k+1; i < n; i++)
				a[i][j] -= l[i][k]*u[k][j];

	}
	double seq_end = omp_get_wtime();
	printf("LU Decomposition took: %f s\n", seq_end - seqso);


	if (display==1) {
		// printf("Matrix a:\n");
		// print(a, n);
		printf("Matrix l:\n");
		print(l, n);
		printf("Matrix u:\n");
		print(u, n);
	}

	// Verification
	double LU[n][n];
	mat_mult(n, l, u, LU);

	if (display==1) {
		printf("PA:\n");
		print(a_copy,n);
		printf("LU:\n");
		print(LU,n);
	}
	double verif_sum = 0.0; 
	for (int j = 0; j < n; j++)
	{	
		double col_l2 = 0.0;
		for (int i = 0; i < n; i++)
		{
			col_l2 += pow(a_copy[i][j] - LU[i][j] , 2);
		}
		verif_sum += sqrt(col_l2);
	}

	printf("Verification -> L_2,1 norm = %f\n", verif_sum);

	// Parallel
	return 0;
}