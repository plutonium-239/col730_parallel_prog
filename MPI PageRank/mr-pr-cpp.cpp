#include "stdio.h"
#include "stdlib.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <boost/config.hpp>
#include "mapreduce/include/mapreduce.hpp"
#include <algorithm>
#include <chrono>

using namespace std; 

int total_pages;
float alpha = 0.85;
vector<double> q_last, q_curr;
float **S;

bool display = false;

bool array_diff(int total_pages, vector<double> a1, vector<double> a2) {
	for (int i = 0; i < total_pages; ++i) {
		// cout<<"D: "<<a1[i]<<" - "<<a2[i]<<" = "<<a1[i] - a2[i]<<endl;
		if (abs(a1[i] - a2[i]) > 1e-6)
		{
			cout<<"Different !"<<endl;
			return true;
		}
	}
	cout<<"SAME - ENDING ITERS !"<<endl;
	return false;
}

template<typename MapTask>
class KeyValueInit : mapreduce::detail::noncopyable{
	private:
		int sequence_;
		int total_pages_;

	public:
		KeyValueInit(int total_pages_in) 
			: sequence_(0), total_pages_(total_pages) {
		}

		bool const setup_key(typename MapTask::key_type &key) {
			key = sequence_++;
			return key<total_pages_;
		}

		bool const get_data(typename MapTask::key_type const &key, typename MapTask::value_type &value) {
			value = q_curr[key];
			if (display)
				cout<<"Init "<<key<<" -> "<<value<<endl;
			return true;
		}
};

struct MapTask : public mapreduce::map_task<int, double>{
	template<typename Runtime>
	void operator()(Runtime &runtime, key_type const &key, value_type const &value) const{
		double temp = 0.0;
		for (int j = 0; j < total_pages; j++)
		{
			// cout<<j<<","<<key<<" : "<<alpha*S[j][key]*q_last[j]<<", "<<(1.0 - alpha)/total_pages*q_last[j]<<endl;
			temp += alpha*S[j][key]*q_last[j] + (1.0 - alpha)/total_pages*q_last[j]; 
		}
		if (display)
			cout<<"M: "<<key<<" -> "<<temp<<endl;
		runtime.emit_intermediate(key, temp);
	}
};

struct ReduceTask : public mapreduce::reduce_task<int, double>{
	template<typename Runtime, typename It>
	void operator()(Runtime &runtime, key_type const &key, It it, It ite) const{
		reduce_task::value_type result = 0;
		for(; it!=ite; ++it) {
			// cout<<"r: "<<key<<" -> "<<*it<<endl;
			result += *it;
		}
		if (display)
			cout<<"R: "<<key<<" -> "<<result<<endl;
		runtime.emit(key, result);
	}
};

typedef mapreduce::job<MapTask, ReduceTask, mapreduce::null_combiner, KeyValueInit<MapTask> > mapreduce_job;

int main(int argc, char **argv)
{

	if (argc < 2) {
		cout<<"ERROR: At least 1 argument required (file to run)"<<endl;
		exit(1);
	}
	if (argc == 3)
		display = (stoi(argv[2])==1)?true:false;


	string fname = argv[1];
	fstream f("test/"+fname+".txt");
	// iss << f.rdbuf();

	vector<int> num_conns, left_pages, right_pages;
	
	string line;
	int prev_page = -1, right_max = 0;
	int ymax = -1;
	while (getline(f, line)) {
		// cout<<line<<endl;
		stringstream iss(line);
		int left, right;
		iss >> left >> right;
		left_pages.push_back(left);
		right_pages.push_back(right);

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

	// int start = ceil(((float)rank)/n_proc*total_pages);
	int start = 0;
	int end = total_pages;
	// int end = min( (int) ceil(((float)rank+1)/n_proc*total_pages), (int) total_pages);	
	// cout<<"proc "<<rank<<" -> start: "<<start<<" end: "<<end<<endl;
		
	// vector<double> q_last(total_pages, 0), q_curr(total_pages, 0);
	vector<double> zerovec(total_pages, 0);
	q_last = zerovec;
	q_curr = zerovec;
	S = (float **) malloc(total_pages*sizeof(float *));
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
		double val;
		if (num_conns[i] != 0)
			val = 0.0;
		else
			val = 1.0/total_pages;

		for (int j = 0; j < total_pages; ++j)
			S[i][j] = val;
	}

	// for (int i = ceil(((float)rank)/n_proc*left_pages.size()); i < ceil(((float)rank+1)/n_proc*left_pages.size()); ++i)
	for (int i = 0; i < left_pages.size(); ++i)
	{
		// cout<<i<<", ";
		// cout<<left_pages[i]<<", ";
		// cout<<right_pages[i]<<", ";
		// cout<<num_conns[left_pages[i]]<<", "<<endl;
		if (left_pages[i]>= start && left_pages[i]<end)
			S[left_pages[i]][right_pages[i]] = 1.0/num_conns[left_pages[i]];
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

	mapreduce::specification spec;
	mapreduce::results result;
	bool final;
	int iters_taken = 0;

	auto tstart = chrono::high_resolution_clock::now();

	do {
		iters_taken += 1;
		for (int i = 0; i < total_pages; i++)
		{
			q_last[i] = q_curr[i];
		}

		mapreduce_job::datasource_type initial_values(total_pages);
		mapreduce_job mr(initial_values, spec);
		mr.run<mapreduce::schedule_policy::cpu_parallel<mapreduce_job>>(result);

		// cout<<"RESULTS"<<endl;
		for(auto it=mr.begin_results(); it!=mr.end_results(); ++it)
			q_curr[it->first] = it->second;
			// cout<<it->first<<" : "<<it->second<<endl;

		if (display) {
			cout<<iters_taken<<"\nLAST : "<<endl;
			for (int i = 0; i < total_pages; i++)
				cout<<q_last[i]<<" ";

			cout<<endl<<"CURR : "<<endl;
			for (int i = 0; i < total_pages; i++)
				cout<<q_curr[i]<<" ";
			cout<<endl;
		}



		// cout<<endl<<"AFTER ALLGATHER CURR : "<<endl;
		// for (int i = 0; i < total_pages; i++)
		// 	cout<<args.q_curr[i]<<" ";

		cout<<"Iter "<<iters_taken<<"\t";
		final = array_diff(total_pages, q_curr, q_last);

	} while (final);

	auto tend = chrono::high_resolution_clock::now();

	if (display) {
		for (int i = 0; i < total_pages; i++)
			cout<<q_curr[i]<<" ";
	}

	cout<<"Making output file\t"<<"outputs/"+fname+"-pr-cpp.txt"<<endl;
	fstream output_file;
	output_file.open("outputs/"+fname+"-pr-cpp.txt", fstream::out);
	double sum = 0.0;
	output_file<<setprecision(12);
	for (int i = 0; i < total_pages; i++) {
		output_file<<i<<" = "<<q_curr[i]<<endl;
		sum += q_curr[i];
	}
	output_file<<"s = "<<sum<<endl;
	output_file.close();

	cout<<endl<<"Took "<<iters_taken<<" iters in "<<chrono::duration_cast<chrono::microseconds>(tend - tstart).count()/(float)1e6<<" s"<<endl;

	return 0;
}
