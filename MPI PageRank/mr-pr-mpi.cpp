#include "mpi.h"
#include "stdio.h"
#include "stdlib.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <math.h>
#include <algorithm>
#include <iomanip>
#include <unordered_map>


using namespace std;

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

string diff_print(bool different) {
	if (different)
		return "Different";
	return "Same";
}

bool display = false;

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

	float *S[total_pages];
	int start = ceil(((float)rank)/n_proc*total_pages);
	int end = min( (int) ceil(((float)rank+1)/n_proc*total_pages), (int) total_pages);	
	cout<<"proc "<<rank<<" -> start: "<<start<<" end: "<<end<<endl;
		
	vector<double> q_last(total_pages, 0), q_curr(total_pages, 0);
	for (int i = start; i < end; i++)
	{
		// q_last.push_back(0);
		// q_curr.push_back(0);
		S[i] = new float[total_pages];
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


	double alpha = 0.85;

	MPI_Barrier(MPI_COMM_WORLD);
	double tstart = MPI_Wtime();

	int iters_taken = 0;
	int red_starts[n_proc], red_ends[n_proc], red_counts[n_proc];

	bool different, final;

	do {
		iters_taken += 1;
		for (int i = 0; i < total_pages; i++)
		{
			q_last[i] = q_curr[i];
			q_curr[i] = 0;
		}

		// MAP STARTS
		for (int j = start; j < end; j++)
		{
			// for (int i = 0; i < conns[j].size(); ++i)
			// {
			// 	q_curr[i] += alpha*weights[j][i]*q_last[j] + (1.0 - alpha)/total_pages*q_last[j]; 
			// }
			for (int i = 0; i < total_pages; ++i)
			{
				// cout<<"("<<rank<<") "<<j<<","<<i<<" : "<<alpha*S[j][i]*q_last[j]<<", "<<(1.0 - alpha)/total_pages*q_last[j]<<endl;
			}
		}

		if (display) {
			cout<<rank<<" start: "<<start<<" end: "<<end<<endl;
			cout<<iters_taken<<"\nLAST : "<<endl;
			for (int i = 0; i < total_pages; i++)
				cout<<q_last[i]<<" ";

			cout<<endl<<"CURR : "<<endl;
			for (int i = 0; i < total_pages; i++)
				cout<<q_curr[i]<<" ";
		}
		MPI_Barrier(MPI_COMM_WORLD);


		// Collate and Reduce all in one step!
		MPI_Allreduce(MPI_IN_PLACE, q_curr.data(), total_pages, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

		if (display) {
			cout<<endl<<"AFTER ALLREDUCE CURR : "<<endl;
			for (int i = 0; i < total_pages; i++)
				cout<<q_curr[i]<<" ";
			cout<<endl;
		}		

		if (rank == 0)
			cout<<"Iter "<<iters_taken<<"\t";
		different = array_diff(total_pages, q_curr, q_last);

		if (display)
			cout<<rank<<" -> "<<diff_print(different)<<endl;
		MPI_Allreduce(&different, &final, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
		if (display)
			cout<<"FINALLY: "<<rank<<" -> "<<diff_print(different)<<endl;
	} while (final);


	// MPI_
	double tend = MPI_Wtime();
	if (display) {
		for (int i = 0; i < total_pages; i++)
			cout<<q_curr[i]<<" ";
	}

	if (rank == 0) {
		cout<<"Making output file\t"<<"outputs/"+fname+"-pr-mpi.txt"<<endl;
		fstream output_file;
		output_file.open("outputs/"+fname+"-pr-mpi.txt", fstream::out);
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
