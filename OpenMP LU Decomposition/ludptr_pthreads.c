#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <string.h>

int num_threads=1;
int display = 0;
int n=3;

void print(int n, double **arr)
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

struct pthread_arg_t {
	int k,n,start,end;
	double **a, **u, **l;
};

void* update_matrix(void* in) {
	struct pthread_arg_t *args = in;
	int k = args->k;
	int n = args->n;
	int start = args->start;
	int end = args->end;
	// printf("starting thread %d to %d\n", start, end);

	for (int j = start; j < end; j++)
		for (int i = k+1; i < n; i++)
			args->a[j][i] -= args->l[k][i]*args->u[j][k];
	// printf("\n");
	// print(n, args->a);
}

int main(int argc, char const *argv[])
{
	
	if (argc>=2)
		n=atoi(argv[1]);
	if (argc>=3)
		num_threads=atoi(argv[2]);
	if (argc>=4)
		display = atoi(argv[3]);


	printf("Using size=%d and threads=%d\n", n, num_threads);
	
	pthread_t thread_arr[num_threads];

	double **a, *a_copy[n], **u, **l;
	a = (double**)malloc(sizeof(double*)*n);
	u = (double**)malloc(sizeof(double*)*n);
	l = (double**)malloc(sizeof(double*)*n);
	int pi[n];
	unsigned seed;

	// Sequential
	// srand(time(NULL));
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
	double ieo = omp_get_wtime();
	printf("init took %f s\n", ieo-seqso);
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

		double mso = omp_get_wtime();
		
		for (int i = k; i < n; i++)
		{
			if (max < fabs(a[k][i])) {
				max = fabs(a[k][i]);
				k_swap = i;
			}
		}

		double meo = omp_get_wtime();
		total_max_time_omp += meo - mso;

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
		for (int i = k+1; i < n; i++)
		{
			l[k][i] = a[k][i]/u[k][k];
			u[i][k] = a[i][k];
		}

		struct pthread_arg_t args[num_threads];
		for (int i = 0; i < num_threads; i++)
		{
			args[i].a = a;
			args[i].l = l;
			args[i].u = u;
			args[i].k = k;
			args[i].n = n;
			args[i].start = (k+1) + i*(int)((n-k-1)/num_threads);
			args[i].end = (k+1) + (i+1)*(int)((n-k-1)/num_threads);
			// Assign all the work left up because of integer division to the last thread (omp does this for us)
			if ((i+1 == num_threads) && (args[i].end != n))
				args[i].end = n;
			// printf("%d : %d to %d\n", i, args.start, args.end);
			pthread_create(&thread_arr[i], NULL, update_matrix, (void *) &args[i]);
		}
		for (int i = 0; i < num_threads; i++)
			pthread_join(thread_arr[i], NULL);
		
	}
	double seq_end = omp_get_wtime();

	printf("max took %f s\n", total_max_time_omp);


	printf("LU Decomposition took: %f s\n", seq_end - seqso);


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
