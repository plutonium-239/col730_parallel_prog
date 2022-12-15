#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
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
	unsigned seed;

	// Sequential
	// srand(time(NULL));
	clock_t seq_start = clock();
	double seqso = omp_get_wtime();
	// #pragma omp parallel for num_threads(num_threads) schedule(static, (int)(n/num_threads)) private(seed)
	for (int j = 0; j < n; j++)
	{
		// seed = j;
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
	clock_t init_end = clock();
	double ieo = omp_get_wtime();
	printf("init took %f (omp: %f)s\n", (float)(init_end - seq_start)/CLOCKS_PER_SEC, ieo-seqso);
	if (display==1) {
		printf("Matrix a:\n");
		print(n, a);
		// printf("Matrix l:\n");
		// print(n, l);
		// printf("Matrix u:\n");
		// print(n, u);
	}

	float total_max_time = 0;
	double total_max_time_omp = 0;

	for (int k = 0; k < n; k++) {
		double max = 0;
		int k_swap;

		clock_t max_start = clock();
		double mso = omp_get_wtime();
		double maxes[num_threads];
		int k_swaps[num_threads];
		# pragma omp parallel num_threads(num_threads)
		{
			double local_max = 0;
			int local_k_swap = 0;
			// # pragma omp for reduction(max: max) schedule(static, (int)(n/num_threads)) nowait
			# pragma omp for schedule(static, (int)(n/num_threads)) nowait
			for (int i = k; i < n; i++)
			{
				if (local_max < fabs(a[k][i])) {
					// max = fabs(a[k][i]);
					local_max = fabs(a[k][i]);
					local_k_swap = i;
				}
			}
			maxes[omp_get_thread_num()] = local_max;
			k_swaps[omp_get_thread_num()] = local_k_swap;
			// # pragma omp critical 
			// {
			// 	if (local_max == max)
			// 		k_swap = local_k_swap;
			// }
		}

		for (int i = 0; i < num_threads; i++)
		{
			if (max < maxes[i]) {
				max = maxes[i];
				k_swap = k_swaps[i];
			}
		}

		clock_t max_end = clock();
		double meo = omp_get_wtime();
		total_max_time += (float)(max_end - max_start)/CLOCKS_PER_SEC;
		total_max_time_omp += meo - mso;
		// printf("max took %f (omp: %f)s\n", (float)(max_end - max_start)/CLOCKS_PER_SEC, meo-mso);


		// printf("\nk = %d\n", k);
		// print(n, a);
		// printf("%f at %d\n", max, k_swap);


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
		#pragma omp parallel for num_threads(num_threads) schedule(static, (int)(n/num_threads))
		for (int i = k+1; i < n; i++)
		{
			l[k][i] = a[k][i]/u[k][k];
			u[i][k] = a[i][k];
		}
		#pragma omp parallel for num_threads(num_threads) schedule(static, (int)(n/num_threads))
		for (int j = k+1; j < n; j++)
			for (int i = k+1; i < n; i++)
				a[j][i] -= l[k][i]*u[j][k];
	}
	clock_t seq_end = clock();

	printf("max took %f s (omp: %f s)\n", total_max_time, total_max_time_omp);


	printf("LU Decomposition took: %f s\n", (float)(seq_end - seq_start)/CLOCKS_PER_SEC);


	if (display==1) {
		// printf("Matrix a:\n");
		// print(n, a);
		printf("Matrix l:\n");
		print(n, l);
		printf("Matrix u:\n");
		print(n, u);
	}

	// Verification
	// double LU[n][n];
	// double *LU[n];
	// for (int i = 0; i < n; i++)
	// 	LU[i] = (double*) malloc(sizeof(double)*n);

	// mat_mult(n, l, u, LU);

	// if (display==1) {
	// 	printf("PA:\n");
	// 	print(n, a_copy);
	// 	printf("LU:\n");
	// 	print(n, LU);
	// }
	// double verif_sum = 0.0; 
	// for (int j = 0; j < n; j++)
	// {	
	// 	double col_l2 = 0.0;
	// 	for (int i = 0; i < n; i++)
	// 	{
	// 		col_l2 += pow(a_copy[j][i] - LU[j][i] , 2);
	// 	}
	// 	verif_sum += sqrt(col_l2);
	// }

	// printf("Verification -> L_2,1 norm = %f\n", verif_sum);

	return 0;
}