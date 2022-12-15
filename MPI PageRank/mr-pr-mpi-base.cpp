#include "mpi.h"
#include "stdio.h"
#include "stdlib.h"
#include "mapreduce.h"
#include "keyvalue.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <math.h>
#include <unordered_map>
#include <algorithm>
#include <iomanip>


// using namespace MAPREDUCE_NS;
using namespace std;

// void fileread(int, char *, KeyValue *, void *);
// void sum(char *, int, char *, int, int *, KeyValue *, void *);
// int ncompare(char *, int, char *, int);
// void output(uint64_t, char *, int, char *, int, KeyValue *, void *);
bool display = false;

bool array_diff(int total_pages, vector<double> a1, vector<double> a2) {
	for (int i = 0; i < total_pages; ++i) {
		if (abs(a1[i] - a2[i]) > 1e-6)
		{
			// cout<<"Different !"<<endl;
			return true;
		}
	}
	// cout<<"SAME - ENDING ITERS !"<<endl;
	return false;
}

struct mr_arguments {
	vector<int> num_conns, left_pages, right_pages;
	int total_pages, n_proc, start, end, red_s, red_e;
	double alpha;
	vector<double> q_last, q_curr;	
	// Adj List representation 
	// vector<int> *conns, vector<int> *weights;
	float **S;
};

void map_make_list(int rank, MAPREDUCE_NS::KeyValue *kv, void *args) {
	mr_arguments *mrargs = (mr_arguments *) args;

	// for (int i = ceil((rank)/mrargs->n_proc*mrargs->left_pages.size()); i < ceil((rank+1)/mrargs->n_proc*mrargs->left_pages.size()); ++i)
	// 	kv->add((char*) &mrargs->left_pages[i], sizeof(int), (char*) &mrargs->right_pages[i], sizeof(int));
	double temp_row[mrargs->total_pages] = {};
	// cout<<rank<<" start: "<<mrargs->start<<" end: "<<mrargs->end<<endl;
	for (int j = mrargs->start; j < mrargs->end; j++)
	{
		// for (int i = 0; i < mrargs->conns[j].size(); ++i)
		// {
		// 	temp_row[i] += mrargs->alpha*(mrargs->weights[j][i])*(mrargs->q_last[j]) + (1.0 - mrargs->alpha)/mrargs->total_pages*(mrargs->q_last[j]); 
		// }
		for (int i = 0; i < mrargs->total_pages; ++i)
		{
			// cout<<"("<<rank<<") "<<j<<","<<i<<" : "<<mrargs->alpha*(mrargs->S[j][i])*(mrargs->q_last[j])<<", "<<(1.0 - mrargs->alpha)/mrargs->total_pages*(mrargs->q_last[j])<<endl;
			temp_row[i] += mrargs->alpha*(mrargs->S[j][i])*(mrargs->q_last[j]) + (1.0 - mrargs->alpha)/mrargs->total_pages*(mrargs->q_last[j]); 
		}
	}
	// cout<<"KV: ";
	for (int i = 0; i < mrargs->total_pages; i++)
	{
		if (display)
			cout<<"M: "<<i<<" -> "<<temp_row[i]<<" "<<endl;
		kv->add((char*) &i, sizeof(int), (char*) &temp_row[i], sizeof(double));
	}
	// cout<<endl;

}

void calc_q(char *key, int keybytes, char *multivalue, int nvalues, int *valuebytes, MAPREDUCE_NS::KeyValue *kv, void *args) {
	mr_arguments *mrargs = (mr_arguments *) args;
	int keyi = *((int*) key);
	// cout<<keyi<<"["<<mrargs->total_pages<<"] : ";

	double sum = 0;
	if (keyi > mrargs->red_e)
		mrargs->red_e = keyi;
	if (keyi < mrargs->red_s)
		mrargs->red_s = keyi;

	for (int i = 0; i < nvalues; ++i)
	{
		auto row = ((double *) multivalue)+i;
		// auto row = (double *) multivalue + i;
		// cout<<"\t"<<*row;
		sum += *row;
		// cout<<"k: "<<key<<", v: "<<multivalue+i<<", n: "<<nvalues<<endl;	
	}
	if (display)
		cout<<"R: "<<keyi<<" -> "<<sum<<" "<<endl;
	mrargs->q_curr[keyi] = sum;
	// mrargs->q_last[keyi] = mrargs->q_curr[keyi];
	// cout<<keyi<<" final = "<<sum<<endl;
}


int main(int argc, char **argv)
{
	if (argc < 2) {
		cout<<"ERROR: At least 1 argument required (file to run)"<<endl;
		exit(1);
	}
	if (argc == 3)
		display = (stoi(argv[2])==1)?true:false;

	MPI_Comm MPI_COMM_WORLD;
	MPI_Init(&argc, &argv);

	int rank, n_proc;
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&n_proc);

	// cout << typeid(MPI_COMM_WORLD).name()<<MPI_COMM_WORLD<<'\n';

	string fname = argv[1];
	fstream f("test/"+fname+".txt");
	// iss << f.rdbuf();

	vector<int> num_conns, left_pages, right_pages;
	
	string line;
	int prev_page = -1, total_pages, right_max = 0;
	int ymax = -1;
	while (getline(f, line)) {
		// cout<<line<<endl;
		stringstream iss(line);
		int left, right;
		iss >> left >> right;
		left_pages.push_back(left);
		right_pages.push_back(right);
		// conns[left].push_back(right)

		if (right > right_max)
			right_max = right;

		if (left == prev_page)
			num_conns[left] += 1;
		else {
			for (int i = prev_page+1; i < left; ++i)
				num_conns.push_back(0);

			num_conns.push_back(1);
			total_pages = right_max;
		}
		prev_page = left;
	}
	f.close();

	for (int i = prev_page+1; i <= right_max; ++i)
		num_conns.push_back(0);
	total_pages++;

	cout<<"Total pages = "<<total_pages<<endl;
	// for(int i =0; i <= total_pages;i++)
	// 	cout<<i<<" - "<<num_conns[i]<<endl;

	// Adj List representation 
	// vector<int> *conns, vector<int> *weights;

	float **S = (float **) malloc(total_pages*sizeof(float *));
	int start = ceil(((float)rank)/n_proc*total_pages);
	int end = min( (int) ceil(((float)rank+1)/n_proc*total_pages), (int) total_pages);	
	cout<<"proc "<<rank<<" -> start: "<<start<<" end: "<<end<<endl;
		
	vector<double> q_last(total_pages, 0.0), q_curr(total_pages, 0.0);
	for (int i = start; i < end; i++)
	{
		// q_last.push_back(0);
		// q_curr.push_back(0);
		S[i] = (float *) malloc(total_pages*sizeof(float));
	}
	q_curr[0] = 1.0;
	// q_last[0] = 1.0;

	for (int i = start; i < end; ++i)
	{
		float val;
		if (num_conns[i] != 0)
			val = 0.0;
		else
			val = 1.0/total_pages;

		for (int j = 0; j < total_pages; ++j)
			S[i][j] = val;
	}

	cout<<"proc "<<rank<<" -> CONN start: "<<ceil(((float)rank)/n_proc*left_pages.size())<<" end: "<<ceil(((float)rank+1)/n_proc*left_pages.size())<<endl;
	// for (int i = ceil(((float)rank)/n_proc*left_pages.size()); i < ceil(((float)rank+1)/n_proc*left_pages.size()); ++i)
	for (int i = 0; i < left_pages.size(); ++i)
	{
		// cout<<i<<", ";
		// cout<<left_pages[i]<<", ";
		// cout<<right_pages[i]<<", ";
		// cout<<num_conns[left_pages[i]]<<", "<<endl;
		if (left_pages[i]>= start && left_pages[i]<end)
			S[left_pages[i]][right_pages[i]] = 1.0/num_conns[left_pages[i]];
			// weights[left_pages[i]].push_back(1.0/num_conns[left_pages[i]])

	}

	// MPI_Allgather()
	if (display) {
		for (int i = start; i < end; ++i)
		{
			cout<<i<<"\t| ";
			for (int j = 0; j < total_pages; ++j)
				cout<<S[i][j]<<"  ";
			cout<<endl;
		}
	}

	// string entireFile = iss.str();
	// entireFile = entireFile.substr(0, entireFile.size()-1);

	// string last_element = entireFile.substr(entireFile.rfind("\n") + 1);
	// cout<<last_element;
	// int num_page = last_connection.substr(0, last_connection.rfind(" ")+1);


	mr_arguments args;
	args.num_conns = num_conns;
	args.left_pages = left_pages;
	args.right_pages = right_pages;
	args.total_pages = total_pages;
	args.n_proc = n_proc;
	// for (int i = start; i < end; ++i)
	// 	args.S[i] = S[i];
	args.S = S;
	args.start = start;
	args.end = end;
	args.red_s = total_pages;
	args.red_e = 0;
	args.alpha = 0.85;
	args.q_last = q_last;
	args.q_curr = q_curr;


	MPI_Barrier(MPI_COMM_WORLD);
	double tstart = MPI_Wtime();

	// double q_last[total_pages] = {} , q_curr[total_pages] = {};
	int iters_taken = 0;
	int red_starts[n_proc], red_ends[n_proc], red_counts[n_proc];

	bool different, final;
	MAPREDUCE_NS::MapReduce *mr = new MAPREDUCE_NS::MapReduce(MPI_COMM_WORLD);
	// mr->verbosity = 1;
	do {
		iters_taken += 1;
		// mr->timer = 1;
		// if (rank == 0) {
		for (int i = 0; i < total_pages; i++)
		{
			args.q_last[i] = args.q_curr[i];
		}
		mr->map(n_proc, map_make_list, &args);
		mr->collate(NULL);
		mr->reduce(calc_q, &args);
		// mr->gather(1);
		// }
		if (display) {
			cout<<iters_taken<<"\nLAST : "<<endl;
			for (int i = 0; i < total_pages; i++)
				cout<<args.q_last[i]<<" ";

			cout<<endl<<"CURR : "<<endl;
			for (int i = 0; i < total_pages; i++)
				cout<<args.q_curr[i]<<" ";
			cout<<endl;
		}
		// delete mr;
		MPI_Barrier(MPI_COMM_WORLD);

		if (display) {
			cout<<" REDUCE DONE ON KEYS"<<args.red_s<<" TO "<<args.red_e<<endl;
			
			cout<<endl<<"TEST : "<<endl;
			for (int i = args.red_s; i <= args.red_e; i++)
				cout<<*(args.q_curr.data() + i)<<" ";
		}

		// if (rank==0) {
		// 	MPI_Bcast(args.q_curr.data() + args.red_s, args.red_e - args.red_s + 1, MPI_DOUBLE, rank, MPI_COMM_WORLD);
		// }
		if (iters_taken == 1) {
			// red_starts[rank] = args.red_s;
			// red_ends[rank] = args.red_e;
			// red_counts[rank] = args.red_e - args.red_s + 1;
			int count = args.red_e - args.red_s + 1;
			MPI_Allgather(&args.red_s, 1, MPI_INT, red_starts, 1, MPI_INT, MPI_COMM_WORLD);
			// MPI_Allgather(&args.red_e, 1, MPI_INT, red_ends, 1, MPI_INT, MPI_COMM_WORLD);
			MPI_Allgather(&count, 1, MPI_INT, red_counts, 1, MPI_INT, MPI_COMM_WORLD);
		}
		if (display) {
			for (int i = 0; i < n_proc; i++){
				cout<<"RED Proc "<<i<<" : "<<red_starts[i]<<" [total="<<red_counts[i]<<"]"<<endl;
			}
		}

		MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &(args.q_curr[0]), red_counts, red_starts, MPI_DOUBLE, MPI_COMM_WORLD);
		

		if (display) {
			cout<<endl<<"AFTER ALLGATHER CURR : "<<endl;
			for (int i = 0; i < total_pages; i++)
				cout<<args.q_curr[i]<<" ";
			cout<<endl;
		}

		if (rank == 0)
			cout<<"Iter "<<iters_taken<<"\t";
		different = array_diff(total_pages, args.q_curr, args.q_last);
		if (display)
			cout<<rank<<" -> "<<different<<endl;
		MPI_Allreduce(&different, &final, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
		if (display)
			cout<<"FINALLY: "<<rank<<" -> "<<different<<endl;
	} while (final);


	// MPI_
	double tend = MPI_Wtime();

	if (display) {
		for (int i = start; i < end; i++)
		{
			cout<<args.q_curr[i]<<" ";
		}
	}

	if (rank == 0) {
		cout<<"Making output file\t"<<"outputs/"+fname+"-pr-mpi-base.txt"<<endl;
		fstream output_file;
		output_file.open("outputs/"+fname+"-pr-mpi-base.txt", fstream::out);
		double sum = 0.0;
		output_file<<setprecision(12);
		for (int i = 0; i < total_pages; i++) {
			output_file<<i<<" = "<<q_curr[i]<<endl;
			sum += q_curr[i];
		}
		output_file<<"s = "<<sum<<endl;
		output_file.close();
		cout<<endl<<"Took "<<iters_taken<<" iters in "<<tend - tstart<<" s"<<endl;
	}

	MPI_Finalize();

	return 0;
}
