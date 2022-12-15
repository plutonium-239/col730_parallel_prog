define help_msg

---------------------------------------------------------------------
		FANN_GPU
Expecting Makefile to be run from examples directory.
Run `fann_gpu.out` with the following command line arguments:
	./fann_gpu.out <dataset> [<scale_threads>]

where <dataset> is one of:
	0 -> Mushroom (default)
	1 -> Gene
	2 -> Soybean
	3 -> Pumadyn-32fm
and <scale_threads> (integer) controls how many threads are used.
num_threads = max_threads/scale_threads
By default, max threads possible are used.
---------------------------------------------------------------------

endef

export help_msg
all:
	nvcc workperthread.cu -lfann -o fann_gpu.out
	@echo "$$help_msg"
	@unset help_msg
