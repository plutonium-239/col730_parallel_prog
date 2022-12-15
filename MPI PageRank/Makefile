all:
	g++ mr-pr-cpp.cpp -o mr-pr-cpp.o -lboost_filesystem -lboost_iostreams -lpthread
	mpic++ mr-pr-mpi.cpp -o mr-pr-mpi.o
	mpic++ mr-pr-mpi-base.cpp -o mr-pr-mpi-base.o -I MPI-mapreduce/ -O MPI-mapreduce/libmrmpi_mpicc.a
