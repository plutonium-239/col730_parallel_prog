/*
Fast Artificial Neural Network Library (fann)
Copyright (C) 2003-2016 Steffen Nissen (steffen.fann@gmail.com)

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

#include <stdio.h>
#include "fann.h"
#include <cstdlib>

__global__ void pass_layer_i(struct fann *ann, struct fann_neuron *neurons, 
	fann_type *dev_weights, struct fann_neuron *dev_last_layer_neurons, int total_work);

__global__ void layer_activations(struct fann_neuron *dev_neurons, int total_work);

__global__ void calc_mse(struct fann *ann, struct fann_neuron *dev_last_layer_begin, fann_type *dev_output, 
	fann_type *error_begin, int last_layer_first_neuron_number, int total_work);

__device__ fann_type sigmoid_gradient(fann_type x, fann_type steepness);

__global__ void fann_backprop_layer(fann_type *dev_weights, struct fann_neuron *dev_neurons, int first_neuron_number,
	int last_layer_first_neuron_number, fann_type *error_begin, int total_work);

__global__ void fann_backprop_layer_complete(fann_type *error_begin, struct fann_neuron *dev_last_layer_neurons, 
	int last_layer_first_neuron_number, int total_work);

__global__ void update_slopes(struct fann_neuron *dev_neurons, struct fann_neuron *dev_last_layer_neurons, int first_neuron_number, 
	fann_type *dev_slopes, fann_type *dev_errors, int total_work);

__global__ void pr(fann_type *error_begin, int last_layer_first_neuron_number, int num_connections);

__global__ void vector_multiply(fann_type *neuron_sum_ptr, fann_type *dev_weights, struct fann_neuron *dev_neurons);

__global__ void gather(fann_type *arr, int N, fann_type *sum);

FANN_EXTERNAL fann_type *FANN_API fann_run(struct fann *ann, fann_type *input, struct fann *dev_ann, fann_type *dev_weights);

float fann_train_epoch_irpropm_custom(struct fann *ann, struct fann_train_data *data);

void fann_backpropagate_MSE_custom(struct fann *ann, struct fann *dev_ann, fann_type *dev_weights, fann_type *dev_errors);

void fann_compute_MSE_custom(struct fann *ann, fann_type *desired_output, fann_type *desired_output_host, fann_type *dev_errors, struct fann *dev_ann);

void fann_update_slopes_batch_custom(struct fann *ann, struct fann_layer *layer_begin,
                              struct fann_layer *layer_end, fann_type *dev_slopes, fann_type *dev_errors);

void fann_update_weights_irpropm(struct fann *ann, unsigned int first_weight,
                                 unsigned int past_end);

fann_type fann_update_MSE(struct fann *ann, struct fann_neuron *neuron, fann_type neuron_diff);


#define check(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

int scale_threads = 1; // calling all kernels with num_threads/scale_threads to see how scaling occurs

int main(int argc, char* argv[])
{
  if (argc < 2)
    printf("Needs at least dataset choice as argument!\n");
  
  int choice = atoi(argv[1]);
	unsigned int num_layers = 3;
	unsigned int num_neurons_hidden = 32;
	float desired_error = (const float) 0.0001;
  unsigned int max_epochs = 300;
	unsigned int epochs_between_reports = 1;
	struct fann *ann;
	struct fann_train_data *train_data, *test_data;

  if (choice == 0) {
    printf("Using mushroom dataset\n");
    num_layers = 3;
    num_neurons_hidden = 32;
    desired_error = 0.0001;
    max_epochs = 30;
    train_data = fann_read_train_from_file("../datasets/mushroom.train");
    test_data = fann_read_train_from_file("../datasets/mushroom.test");
    ann = fann_create_standard(num_layers,
              train_data->num_input, num_neurons_hidden, train_data->num_output);
  }
  else if (choice == 1) {
    printf("Using gene dataset\n");
    num_layers = 3;
    num_neurons_hidden = 512;
    desired_error = 0.0001;
    train_data = fann_read_train_from_file("../datasets/gene.train");
    test_data = fann_read_train_from_file("../datasets/gene.test");
    ann = fann_create_standard(num_layers,
              train_data->num_input, num_neurons_hidden, train_data->num_output);
  }
  else if (choice == 2) {
    printf("Using soybean dataset\n");
    num_layers = 3;
    num_neurons_hidden = 64;
    desired_error = 0.001;
    train_data = fann_read_train_from_file("../datasets/soybean.train");
    test_data = fann_read_train_from_file("../datasets/soybean.test");
    ann = fann_create_standard(num_layers,
              train_data->num_input, num_neurons_hidden, train_data->num_output);
  }
  else if (choice == 3) {
    printf("Using pumadyn dataset\n");
    num_layers = 3;
    num_neurons_hidden = 32;
    max_epochs = 50;
    desired_error = 0.01;
    train_data = fann_read_train_from_file("../datasets/pumadyn-32fm.train");
    test_data = fann_read_train_from_file("../datasets/pumadyn-32fm.test");
  	ann = fann_create_standard(num_layers,
  					  train_data->num_input, num_neurons_hidden, train_data->num_output);
  }
  
  if (argc == 3) {
    scale_threads = atoi(argv[2]);
    printf("Using scale_threads = %d\n", scale_threads);
  }

  unsigned int i = 0;

  
	printf("Creating network with layers (%u,%u,%u)\n", train_data->num_input, num_neurons_hidden, train_data->num_output);

	
  ann->train_errors = (fann_type *)calloc(ann->total_neurons, sizeof(fann_type));
  if (ann->train_errors == NULL) {
    fann_error((struct fann_error *)ann, FANN_E_CANT_ALLOCATE_MEM);
    return 1;
  }

  ann->train_slopes = (fann_type *)calloc(ann->total_connections_allocated, sizeof(fann_type));
  if (ann->train_slopes == NULL) {
    fann_error((struct fann_error *)ann, FANN_E_CANT_ALLOCATE_MEM);
    return -1;
  }
  
	printf("Training network.\n");

	fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
	fann_set_activation_function_output(ann, FANN_SIGMOID);

	/*fann_set_training_algorithm(ann, FANN_TRAIN_INCREMENTAL); */

	fann_train_on_data(ann, train_data, max_epochs, epochs_between_reports, desired_error);

	printf("Testing network.\n");

	fann_reset_MSE(ann);
	for(i = 0; i < fann_length_train_data(test_data); i++)
	{
		fann_test(ann, test_data->input[i], test_data->output[i]);
	}
	
	printf("MSE error on test data: %f\n", fann_get_MSE(ann));

	printf("Saving network.\n");

	fann_save(ann, "mushroom_float.net");

	printf("Cleaning up.\n");
	fann_destroy_train(train_data);
	fann_destroy_train(test_data);
	fann_destroy(ann);

	return 0;
}

FANN_EXTERNAL void FANN_API fann_train_on_data(struct fann *ann, struct fann_train_data *data,
                                               unsigned int max_epochs,
                                               unsigned int epochs_between_reports,
                                               float desired_error) {
  float error;
  unsigned int i;
  int desired_error_reached;

#ifdef DEBUG
  printf("Training with %s\n", FANN_TRAIN_NAMES[ann->training_algorithm]);
#endif

  if (epochs_between_reports && ann->callback == NULL) {
    printf("Max epochs %8d. Desired error: %.10f.\n", max_epochs, desired_error);
  }
  printf("USING CUSTOM DEFINITION\n");

  for (i = 1; i <= max_epochs; i++) {
    /*
     * train
     */
    error = fann_train_epoch_irpropm_custom(ann, data);
    desired_error_reached = fann_desired_error_reached(ann, desired_error);

    /*
     * print current output
     */
    if (epochs_between_reports && (i % epochs_between_reports == 0 || i == max_epochs || i == 1 ||
                                   desired_error_reached == 0)) {
      if (ann->callback == NULL) {
        printf("Epochs     %8d. Current error: %.10f. Bit fail %d.\n", i, error, ann->num_bit_fail);
      } else if (((*ann->callback)(ann, data, max_epochs, epochs_between_reports, desired_error,
                                   i)) == -1) {
        /*
         * you can break the training by returning -1
         */
        break;
      }
    }

    if (desired_error_reached == 0) break;
  }
}

float fann_train_epoch_irpropm_custom(struct fann *ann, struct fann_train_data *data) {
  unsigned int i;
  if (ann->prev_train_slopes == NULL) {
  	fann_clear_train_arrays(ann);
  }

  fann_reset_MSE(ann);

  struct fann *dev_ann;
	fann_type *dev_weights, *dev_errors, *dev_output, *dev_slopes;
	check(cudaMalloc((void **)&dev_ann, sizeof(struct fann)));
	check(cudaMalloc((void **)&dev_weights, ann->total_connections*sizeof(fann_type)));
	check(cudaMalloc((void **)&dev_errors, ann->total_neurons*sizeof(fann_type)));
	check(cudaMalloc((void **)&dev_output, data->num_data*data->num_output*sizeof(fann_type)));
	check(cudaMalloc((void **)&dev_slopes, ann->total_connections_allocated*sizeof(fann_type)));



	// printf("%d vs %d*%d\n", sizeof(ann->weights), ann->total_connections, sizeof(fann_type));
	check(cudaMemcpy(dev_ann, ann, sizeof(struct fann), cudaMemcpyHostToDevice));
	check(cudaMemcpy(dev_weights, ann->weights, ann->total_connections*sizeof(fann_type), cudaMemcpyHostToDevice));
	// check(cudaMemcpy(dev_weights, ann->weights, ann->total_connections*sizeof(fann_type), cudaMemcpyHostToDevice));





  for (i = 0; i < data->num_data; i++) {
  	fann_run(ann, data->input[i], dev_ann, dev_weights);
  	// fann_type *out = data->output[i];
  	// printf("\n%f %f | ", *out, *(out+1));
		check(cudaMemcpy(&dev_output[i], data->output[i], data->num_output*sizeof(fann_type), cudaMemcpyHostToDevice));
    fann_compute_MSE_custom(ann, &dev_output[i], data->output[i], dev_errors, dev_ann);
    fann_backpropagate_MSE_custom(ann, dev_ann, dev_weights, dev_errors);
    fann_update_slopes_batch_custom(ann, ann->first_layer + 1, ann->last_layer - 1, dev_slopes, dev_errors);
  }

	check(cudaMemcpy(ann->weights, dev_weights, ann->total_connections*sizeof(fann_type), cudaMemcpyDeviceToHost));
	check(cudaMemcpy(ann->train_slopes, dev_slopes, ann->total_connections_allocated*sizeof(fann_type), cudaMemcpyDeviceToHost));

  // printf("%p\n", ann->prev_steps);
  // for (i = 0; i < ann->total_connections; ++i)
  // {
	 //  printf("%d", *(ann->prev_steps+i));
  // }
  fann_update_weights_irpropm(ann, 0, ann->total_connections);

  cudaFree(dev_ann);
  cudaFree(dev_weights);
  cudaFree(dev_output);
  cudaFree(dev_errors);
  cudaFree(dev_slopes);
  
  return fann_get_MSE(ann);
}

FANN_EXTERNAL fann_type *FANN_API fann_run(struct fann *ann, fann_type *input, struct fann *dev_ann, fann_type *dev_weights) {
  struct fann_neuron *neuron_it, *last_neuron, *neurons, **neuron_pointers;
  unsigned int i, num_input, num_output;
  fann_type *output;
  struct fann_layer *layer_it, *last_layer;

  /* store some variabels local for fast access */
  struct fann_neuron *first_neuron = ann->first_layer->first_neuron;

#ifdef FIXEDFANN
  int multiplier = ann->multiplier;
  unsigned int decimal_point = ann->decimal_point;

  /* values used for the stepwise linear sigmoid function */
  fann_type r1 = 0, r2 = 0, r3 = 0, r4 = 0, r5 = 0, r6 = 0;
  fann_type v1 = 0, v2 = 0, v3 = 0, v4 = 0, v5 = 0, v6 = 0;

  fann_type last_steepness = 0;
  unsigned int last_activation_function = 0;
#else
  fann_type max_sum = 0;
#endif

  /* first set the input */
  num_input = ann->num_input;
  for (i = 0; i != num_input; i++) {
#ifdef FIXEDFANN
    if (fann_abs(input[i]) > multiplier) {
      printf(
          "Warning input number %d is out of range -%d - %d with value %d, integer overflow may "
          "occur.\n",
          i, multiplier, multiplier, input[i]);
    }
#endif
    first_neuron[i].value = input[i];
  }
  /* Set the bias neuron in the input layer */
#ifdef FIXEDFANN
  (ann->first_layer->last_neuron - 1)->value = multiplier;
#else
  (ann->first_layer->last_neuron - 1)->value = 1;
#endif

  last_layer = ann->last_layer;

  for (layer_it = ann->first_layer + 1; layer_it != last_layer; layer_it++) {
  	int neurons_in_layer = layer_it->last_neuron - layer_it->first_neuron -1;
  	int neurons_in_prev_layer = (layer_it-1)->last_neuron - (layer_it-1)->first_neuron -1;

  	// printf("Neurons in layer_it:%d, prev layer:%d\n", neurons_in_layer, neurons_in_prev_layer);
  // 	cudaError_t err1 = cudaGetLastError();
		// if(err1 != cudaSuccess)
		// 	printf("Error %s\n",cudaGetErrorString(err1));
		struct fann_neuron *dev_neurons, *dev_last_layer_neurons;
		check(cudaMalloc((void **)&dev_last_layer_neurons, neurons_in_prev_layer*sizeof(struct fann_neuron)));
  	check(cudaMemcpy(dev_last_layer_neurons, (layer_it-1)->first_neuron, neurons_in_prev_layer*sizeof(struct fann_neuron), cudaMemcpyHostToDevice));

  	check(cudaMalloc((void **)&dev_neurons, neurons_in_layer*sizeof(struct fann_neuron)));
  	check(cudaMemcpy(dev_neurons, layer_it->first_neuron, neurons_in_layer*sizeof(struct fann_neuron), cudaMemcpyHostToDevice));
		// printf("dev_n: %p to %p\n",dev_neurons, dev_neurons+neurons_in_layer);
		// cudaError_t err2 = cudaGetLastError();
		// if(err2 != cudaSuccess)
		// 	printf("Error 2 %s\n",cudaGetErrorString(err2));

  	// for (neuron_it = layer_it->first_neuron, i=0; neuron_it != last_neuron; neuron_it++) {
  	// 	cudaMemcpy(dev_neurons + i, neuron_it, sizeof(struct fann_neuron), cudaMemcpyHostToDevice);
  	// 	cudaError_t err1 = cudaGetLastError();
			// if(err1 != cudaSuccess)
			// 	printf("Error 2 %s\n",cudaGetErrorString(err1));
  	// 	i++;
  	// }
  	// printf("layer %d\n", neurons_in_layer);
    // printf("\n");
    // printf("\n");
    pass_layer_i<<<1,(int) (neurons_in_layer/scale_threads)>>>(dev_ann, dev_neurons, dev_weights, 
      dev_last_layer_neurons, neurons_in_layer);
    // pass_layer_i<<<1,1>>>(dev_ann, dev_neurons, dev_weights, dev_last_layer_neurons, neurons_in_layer);
    // cudaDeviceSynchronize();
    // printf("\nvs\n");
    // layer_activations<<<1,1>>>(dev_neurons, neurons_in_layer);
    // layer_activations<<<1, (int) (neurons_in_layer/scale_threads)>>>(dev_neurons, neurons_in_layer);
    // printf("\n");
  	check(cudaMemcpy(layer_it->first_neuron, dev_neurons, neurons_in_layer*sizeof(struct fann_neuron), cudaMemcpyDeviceToHost));
		cudaError_t err4 = cudaGetLastError();
		if(err4 != cudaSuccess)
			printf("Error 4 %s\n",cudaGetErrorString(err4));
  	
   	//  for (neuron_it = layer_it->first_neuron, i=0; neuron_it != last_neuron; neuron_it++) {
  	// 	cudaMemcpy(neuron_it, dev_neurons + i, sizeof(struct fann_neuron), cudaMemcpyDeviceToHost);
  	// 	i++;
  	// }
  	cudaFree(dev_neurons);
  	cudaFree(dev_last_layer_neurons);
    }


  /* set the output */
  output = ann->output;
  num_output = ann->num_output;
  neurons = (ann->last_layer - 1)->first_neuron;
  for (i = 0; i != num_output; i++) {
    output[i] = neurons[i].value;
  }
  return ann->output;
}

__global__ void vector_multiply(fann_type *neuron_sum_ptr, fann_type *dev_weights, struct fann_neuron *dev_last_layer_neurons) {
  neuron_sum_ptr[threadIdx.x] = fann_mult(dev_weights[threadIdx.x], dev_last_layer_neurons[threadIdx.x].value);
  // printf("at %p + %d : %.3f * %.3f = %.3f \n", neuron_sum_ptr, threadIdx.x, dev_weights[threadIdx.x], 
    // dev_last_layer_neurons[threadIdx.x].value, neuron_sum_ptr[threadIdx.x]);
  // printf("-");
}

__global__ void gather(fann_type *temp_outs, int N, fann_type *sum) {
  // fann_type temp_sum = 0;
  for (int i = 0; i < N-1; ++i)
  {
    // printf("%p (%p + %d) received = %.3f\n", temp_outs+i, temp_outs, i, temp_outs[i]);
    *sum += temp_outs[i];
  }
}

__global__ void pass_layer_i(struct fann *ann, struct fann_neuron *dev_neurons, fann_type *dev_weights,
		struct fann_neuron *dev_last_layer_neurons, int total_work) {
  int work_per_thread = total_work/blockDim.x;
  for (int work = 0; work < work_per_thread; ++work)
  {
    unsigned int activation_function;
    unsigned int i, num_connections;
  	struct fann_neuron *neuron_it, *last_neuron, *neurons, **neuron_pointers;
    fann_type steepness;
  	fann_type *weights, neuron_sum, max_sum;
    

    neuron_it = dev_neurons + threadIdx.x + work*work_per_thread; // COALESCED (also change call to vector_multiply)
    // neuron_it = dev_neurons + threadIdx.x*work_per_thread + work; // NOT COALESCED

    // printf("Kernel %p is running, %p\n", neuron_it, layer_it->first_neuron);

    neuron_it->value = 1;

    neuron_sum = 0;
    num_connections = neuron_it->last_con - neuron_it->first_con;
    weights = dev_weights + neuron_it->first_con;
    // printf("ann->wts:%p vs dev_wts:%p vs neuron_it->first_con:%p\n",ann->weights, dev_weights, neuron_it->first_con);

    // neurons = (layer_it - 1)->first_neuron;
    neurons = dev_last_layer_neurons;
    // printf("wts:%p, neur:%p, dev_neur:%p\n", weights, neurons, dev_neurons);


    // if (!use_vector) {
      for(i = 0;i != num_connections-1; i++){
        neuron_sum += fann_mult(weights[i], neurons[i].value);
      }
    // }
    // else {
    //   fann_type *temp_outs;
    //   cudaMalloc((void **)&temp_outs, (num_connections-1)*sizeof(fann_type));
    //   fann_type *temp_sum;
    //   cudaMalloc((void **)&temp_sum, sizeof(fann_type));

    //   vector_multiply<<<1, num_connections-1>>>(temp_outs, dev_weights, dev_last_layer_neurons);
    //   gather<<<1,1>>>(temp_outs, num_connections-1, temp_sum);
    //   cudaFree(temp_outs);
    //   neuron_sum = *temp_sum;
    //   cudaFree(temp_sum);
    // }

    

    // printf("%d %.3f vs %.3f | ", threadIdx.x + work*work_per_thread, neuron_sum, temp_sum);
    /* unrolled loop start */
    // i = num_connections & 3; /* same as modulo 4 */
    // // printf("starting %p at %d\n", neuron_it, i);
    // switch (i) {
    //   case 3:
    //     neuron_sum += fann_mult(weights[2], neurons[2].value);
    //   case 2:
    //     neuron_sum += fann_mult(weights[1], neurons[1].value);
    //   case 1:
    //     neuron_sum += fann_mult(weights[0], neurons[0].value);
    //   case 0:
    //     break;
    // }

    // for (; i < num_connections-4; i +=4) {
    // 	// if (i<0) {
    //   // 	printf("i %u: ", i+3);
    //   // 	printf("w %f, ", weights[i+3]);
    //   // 	printf("n %f", neurons[i+3].value);
    // 	// }
    //   neuron_sum += fann_mult(weights[i], neurons[i].value);
    //   neuron_sum += fann_mult(weights[i + 1], neurons[i + 1].value);
    //   neuron_sum += fann_mult(weights[i + 2], neurons[i + 2].value);
    //   neuron_sum += fann_mult(weights[i + 3], neurons[i + 3].value);
    // }
    // for (i=0; i != num_connections-1; i ++) {
    //   neuron_sum += fann_mult(weights[i], neurons[i].value);
    // }

    /* unrolled loop end */

    
     
    // neuron_it->sum = neuron_sum;
    
  // #else
    activation_function = neuron_it->activation_function;
    steepness = neuron_it->activation_steepness;
    neuron_sum = fann_mult(steepness, neuron_sum);

    max_sum = 150 / steepness;
    if (neuron_sum > max_sum)
      neuron_sum = max_sum;
    else if (neuron_sum < -max_sum)
      neuron_sum = -max_sum;

    neuron_it->sum = neuron_sum;

    fann_activation_switch(activation_function, neuron_sum, neuron_it->value);
  // #endif
    // printf("ended %p\n", neuron_it);
    // }
    // cudaMemcpy(neuron_it, dev_neuron_it, sizeof(fann_neuron), cudaMemcpyDeviceToHost);
    // cudaFree(dev_neuron_it);
  }
}

__global__ void layer_activations(struct fann_neuron *dev_neurons, int total_work) {
  int work_per_thread = total_work/blockDim.x;
  for (int work = 0; work < work_per_thread; ++work)
  {
    struct fann_neuron *neuron_it;
    fann_type steepness, max_sum;
    unsigned int activation_function;

    neuron_it = dev_neurons + threadIdx.x + work*work_per_thread; // COALESCED (also change call to vector_multiply)

    activation_function = neuron_it->activation_function;
    steepness = neuron_it->activation_steepness;

    neuron_it->sum = fann_mult(steepness, neuron_it->sum);
    max_sum = 150 / steepness;
    if (neuron_it->sum > max_sum)
      neuron_it->sum = max_sum;
    else if (neuron_it->sum < -max_sum)
      neuron_it->sum = -max_sum;

    fann_activation_switch(activation_function, neuron_it->sum, neuron_it->value);
    // printf("%d %.3f | ", threadIdx.x + work*work_per_thread, neuron_it->sum);

  }
}

void fann_backpropagate_MSE_custom(struct fann *ann, struct fann *dev_ann, fann_type *dev_weights, fann_type *dev_errors) {
  struct fann_layer *layer_it;
  struct fann_neuron *neuron_it, *last_neuron;
  struct fann_neuron **connections;
  unsigned int i;

  fann_type tmp_error;
  fann_type *error_begin = ann->train_errors;
  fann_type *error_prev_layer;
  fann_type *weights;
  const struct fann_neuron *first_neuron = ann->first_layer->first_neuron;
  const struct fann_layer *second_layer = ann->first_layer + 1;
  struct fann_layer *last_layer = ann->last_layer;

  /* go through all the layers, from last to first.
   * And propagate the error backwards */
  for (layer_it = last_layer - 1; layer_it > second_layer; --layer_it) {

    last_neuron = layer_it->last_neuron;
    int neurons_in_layer = layer_it->last_neuron - layer_it->first_neuron;
    int neurons_in_prev_layer = (layer_it-1)->last_neuron - (layer_it-1)->first_neuron;
    // printf("Layer with %d\n", neurons_in_layer);


    /* for each connection in this layer, propagate the error backwards */
    error_prev_layer = error_begin + ((layer_it - 1)->first_neuron - first_neuron);

    struct fann_neuron *dev_neurons, *dev_last_layer_neurons;
    check(cudaMalloc((void **)&dev_neurons, neurons_in_layer*sizeof(struct fann_neuron)));
  	check(cudaMemcpy(dev_neurons, layer_it->first_neuron, neurons_in_layer*sizeof(struct fann_neuron), cudaMemcpyHostToDevice));

  	check(cudaMalloc((void **)&dev_last_layer_neurons, neurons_in_prev_layer*sizeof(struct fann_neuron)));
  	check(cudaMemcpy(dev_last_layer_neurons, (layer_it-1)->first_neuron, neurons_in_prev_layer*sizeof(struct fann_neuron), cudaMemcpyHostToDevice));

    for (neuron_it = layer_it->first_neuron; neuron_it != last_neuron; neuron_it++) {
      tmp_error = error_begin[neuron_it - first_neuron];
      weights = ann->weights + neuron_it->first_con;
      // printf("tmp_error: %f, n->it - first_neuron: %ld\n", tmp_error, neuron_it - first_neuron);
      for (i = neuron_it->last_con - neuron_it->first_con; i--;) {
        // printf("i: %d", i);
        // printf("w: %f\n", weights[i]);
        /*printf("i = %d\n", i);
         * printf("error_prev_layer[%d] = %f\n", i, error_prev_layer[i]);
         * printf("weights[%d] = %f\n", i, weights[i]); */
        error_prev_layer[i] += tmp_error * weights[i];
        // printf("%f\n", error_prev_layer[i]);
      }
    }
    // printf("\nCPU:\n");
    neuron_it = layer_it->first_neuron;
    for (i = neuron_it->last_con - neuron_it->first_con; i--;) {
      // printf("%f ", error_prev_layer[i]);
    }

    // for (neuron_it = layer_it->first_neuron; neuron_it != last_neuron; neuron_it++) {
    
    // }
    int first_neuron_number = 0, last_layer_first_neuron_number = 0;
  	struct fann_layer *l_it;
    for (l_it = layer_it-2; l_it >= ann->first_layer; l_it--) {
    	last_layer_first_neuron_number += l_it->last_neuron - l_it->first_neuron;
    }
    first_neuron_number = last_layer_first_neuron_number + (layer_it-1)->last_neuron - (layer_it-1)->first_neuron;

    error_begin = ann->train_errors + first_neuron_number;
    // printf("\nError %d\n", first_neuron_number);
    // for (int i = 0; i < 2; ++i)
    // {
    //   printf("%f | ", *(error_begin+i));  
    // }
    // printf("\nError ends\n");
    
    // printf("current:%d, prev:%d \n",first_neuron_number,last_layer_first_neuron_number);
    fann_backprop_layer<<<1, neurons_in_layer>>>(dev_weights, dev_neurons, first_neuron_number, 
      last_layer_first_neuron_number, dev_errors, neurons_in_layer);
    // cudaDeviceSynchronize();
    // pr<<<1,1>>>(dev_errors, last_layer_first_neuron_number, neuron_it->last_con - neuron_it->first_con);
    // check(cudaMemcpy(ann->train_errors, dev_errors, ));

    /* then calculate the actual errors in the previous layer */
    // error_prev_layer = error_begin + ((layer_it - 1)->first_neuron - first_neuron);
    // last_neuron = (layer_it - 1)->last_neuron;

    // printf("%p + %d = %p\n", dev_errors, ann->total_neurons*sizeof(fann_type), dev_errors+ann->total_neurons);
    // printf("%d\n", ann->total_neurons);
    fann_backprop_layer_complete<<<1, neurons_in_prev_layer>>>(dev_errors, dev_last_layer_neurons, 
      last_layer_first_neuron_number, neurons_in_prev_layer);
    // for (neuron_it = (layer_it - 1)->first_neuron; neuron_it != last_neuron; neuron_it++) {
    //   *error_prev_layer *=
    //       fann_activation_derived(neuron_it->activation_function, neuron_it->activation_steepness,
    //                               neuron_it->value, neuron_it->sum);
    //   error_prev_layer++;
    // }
    cudaFree(dev_neurons);
    cudaFree(dev_last_layer_neurons);
  }
}

__global__ void pr(fann_type *error_begin, int last_layer_first_neuron_number, int num_connections) {
  fann_type *error_prev_layer;
  unsigned int i;
  error_prev_layer = error_begin + last_layer_first_neuron_number;
  // weights = dev_weights + neuron_it->first_con;
  // printf("\ngpu:\n");
  for (i = num_connections; i--;) {
    // printf("%f ", error_prev_layer[i]);
  }
}

__global__ void fann_backprop_layer(fann_type *dev_weights, struct fann_neuron *dev_neurons, int first_neuron_number,
	int last_layer_first_neuron_number, fann_type *error_begin, int total_work) {
  // int work_per_thread = total_work/blockDim.x;
  // for (int work = 0; work < total_work; ++work)
  // {
    fann_type tmp_error;
    unsigned int i;
    fann_type *error_prev_layer, *weights;
  	struct fann_neuron *neuron_it = dev_neurons + threadIdx.x;
    // struct fann_neuron *neuron_it = dev_neurons + threadIdx.x*work_per_thread + work;
    tmp_error = error_begin[first_neuron_number + threadIdx.x];
    // tmp_error = error_begin[first_neuron_number + threadIdx.x*work_per_thread + work];
    // printf("tmp_error: %f, nv: %f\n", tmp_error, neuron_it->value);
    weights = dev_weights + neuron_it->first_con;
    error_prev_layer = error_begin + last_layer_first_neuron_number;
    // printf("%p\n", error_prev_layer);
    // printf("%u, %u\n", neuron_it->last_con, neuron_it->first_con);
    for (i = neuron_it->last_con - neuron_it->first_con; i--;) {
      atomicAdd(&error_prev_layer[i], tmp_error * weights[i]);
    }
  // }
}

__global__ void fann_backprop_layer_complete(fann_type *error_begin, struct fann_neuron *dev_last_layer_neurons, 
	int last_layer_first_neuron_number, int total_work) {
  // int work_per_thread = total_work/blockDim.x;
  // for (int work = 0; work < total_work; ++work)
  // {
  	struct fann_neuron *neuron_it = dev_last_layer_neurons + threadIdx.x;
    // struct fann_neuron *neuron_it = dev_last_layer_neurons + threadIdx.x*work_per_thread + work;
    fann_type *error_prev_layer;
    error_prev_layer = error_begin + last_layer_first_neuron_number + threadIdx.x;
    // error_prev_layer = error_begin + last_layer_first_neuron_number + threadIdx.x*work_per_thread + work;
    *error_prev_layer *= sigmoid_gradient(neuron_it->value, neuron_it->activation_steepness);
  // }
}

void fann_compute_MSE_custom(struct fann *ann, fann_type *desired_output, fann_type *desired_output_host, fann_type *dev_errors, struct fann *dev_ann) {
	fann_type *error_it = 0, *error_begin = 0;
	fann_type neuron_diff, neuron_value;
  struct fann_neuron *last_layer_begin = (ann->last_layer - 1)->first_neuron;
  // const struct fann_neuron *last_layer_end = last_layer_begin + ann->num_output;
  // const struct fann_neuron *first_neuron = ann->first_layer->first_neuron;

  int last_layer_first_neuron_number = 0;
	struct fann_layer *l_it;
  for (l_it = ann->last_layer-2; l_it >= ann->first_layer; l_it--) {
  	last_layer_first_neuron_number += l_it->last_neuron - l_it->first_neuron;
  }
	// const struct fann_neuron *last_layer_end = last_layer_begin + ann->num_output;
  // const struct fann_neuron *first_neuron = ann->first_layer->first_neuron;

  /* if no room allocated for the error variabels, allocate it now */
  // if (ann->train_errors == NULL) {
  //   ann->train_errors = (fann_type *)calloc(ann->total_neurons, sizeof(fann_type));
  //   if (ann->train_errors == NULL) {
  //     fann_error((struct fann_error *)ann, FANN_E_CANT_ALLOCATE_MEM);
  //     return;
  //   }
  // } 
  // else {
    /* clear the error variabels */
  check(cudaMemset(dev_errors, 0, (ann->total_neurons) * sizeof(fann_type)));
  // memset(ann->train_errors, 0, (ann->total_neurons) * sizeof(fann_type));
	// }
  // error_begin = ann->train_errors;
	struct fann_neuron *dev_last_layer_begin;
  check(cudaMalloc((void **)&dev_last_layer_begin, sizeof(struct fann_neuron)*ann->num_output));
	check(cudaMemcpy(dev_last_layer_begin, last_layer_begin, sizeof(struct fann_neuron)*ann->num_output, cudaMemcpyHostToDevice));
	// printf("\n");
#ifdef DEBUGTRAIN
  printf("\ncalculate errors\n");
#endif
  /* calculate the error and place it in the output layer */
  // error_it = error_begin + (last_layer_begin - first_neuron);

  // for (; last_layer_begin != last_layer_end; last_layer_begin++) {
  //   neuron_value = last_layer_begin->value;
  //   neuron_diff = *desired_output_host - neuron_value;

  //   neuron_diff = fann_update_MSE(ann, last_layer_begin, neuron_diff);

  //   if (ann->train_error_function) { /* TODO make switch when more functions */
  //     if (neuron_diff < -.9999999)
  //       neuron_diff = -17.0;
  //     else if (neuron_diff > .9999999)
  //       neuron_diff = 17.0;
  //     else
  //       neuron_diff = (fann_type)log((1.0 + neuron_diff) / (1.0 - neuron_diff));
  //   }

  //   *error_it = fann_activation_derived(last_layer_begin->activation_function,
  //                                       last_layer_begin->activation_steepness, neuron_value,
  //                                       last_layer_begin->sum) *
  //               neuron_diff;
  //   printf("cpu e:%f | n:%f | d:%f || ", *error_it, neuron_value, neuron_diff);
  //   desired_output_host++;
  //   error_it++;

  //   ann->num_MSE++;
  // }

  // printf("\n");
  calc_mse<<<1,ann->num_output>>>(dev_ann, dev_last_layer_begin, desired_output, dev_errors, last_layer_first_neuron_number, ann->num_output);



  check(cudaMemcpy(&ann->MSE_value, &dev_ann->MSE_value, sizeof(float), cudaMemcpyDeviceToHost));
  check(cudaMemcpy(&ann->num_bit_fail, &dev_ann->num_bit_fail, sizeof(unsigned int), cudaMemcpyDeviceToHost));

  check(cudaMemcpy(ann->train_errors, dev_errors, (ann->total_neurons)*sizeof(fann_type), cudaMemcpyDeviceToHost));
  error_begin = ann->train_errors + last_layer_first_neuron_number;
  // printf("\ncorrect Error %d \n", last_layer_first_neuron_number);
  // for (int i = 0; i < 2; ++i)
  // {
  //   printf("%f | ", *(error_begin+i));  
  // }
  // printf("\nError ends\n");
  ann->num_MSE += ann->num_output;

  cudaFree(dev_last_layer_begin);
}

__global__ void calc_mse(struct fann *ann, struct fann_neuron *dev_last_layer_begin, fann_type *dev_output, 
	fann_type *error_begin, int last_layer_first_neuron_number, int total_work) {
  // int work_per_thread = total_work/blockDim.x;
  // for (int work = 0; work < total_work; ++work)
  // {
  	fann_type neuron_value, neuron_diff, *error_it;
  	fann_type *desired_output = dev_output + threadIdx.x;
    // fann_type *desired_output = dev_output + threadIdx.x*work_per_thread + work;
  	// printf("%f ", *desired_output, desired_output);
  	dev_last_layer_begin += threadIdx.x;
    // dev_last_layer_begin += threadIdx.x*work_per_thread + work;
  	
  	neuron_value = dev_last_layer_begin->value;
  	neuron_diff = *desired_output - neuron_value;
  	error_it = error_begin + last_layer_first_neuron_number + threadIdx.x;
    // error_it = error_begin + last_layer_first_neuron_number + threadIdx.x*work_per_thread + work;
  	
    // neuron_diff = fann_update_MSE(ann, last_layer_begin, neuron_diff);
    atomicAdd(&ann->MSE_value, neuron_diff*neuron_diff);
  	if (fann_abs(neuron_diff) >= ann->bit_fail_limit) {
  		atomicAdd(&ann->num_bit_fail, 1);
    }
  	
    if (ann->train_error_function) { /* TODO make switch when more functions */
      if (neuron_diff < -.9999999)
        neuron_diff = -17.0;
      else if (neuron_diff > .9999999)
        neuron_diff = 17.0;
      else
        neuron_diff = (fann_type)log((1.0 + neuron_diff) / (1.0 - neuron_diff));
    }
  	
    *error_it = sigmoid_gradient(neuron_value, dev_last_layer_begin->activation_steepness)*neuron_diff;
    // printf("kernel e:%f | n:%f | d:%f || ", *error_it, neuron_value, neuron_diff);
  // }
}

__device__ fann_type sigmoid_gradient(fann_type x, fann_type steepness) {
	const float lo = 0.01f;
	const float hi = 0.99f;
	x = (((x) < (lo)) ? (lo) : (((x) > (hi)) ? (hi) : (x)));
	return (2.0f * steepness * x * (1.0f - x));
}

void fann_update_weights_irpropm(struct fann *ann, unsigned int first_weight,
                                 unsigned int past_end) {
  fann_type *train_slopes = ann->train_slopes;
  fann_type *weights = ann->weights;
  fann_type *prev_steps = ann->prev_steps;
  fann_type *prev_train_slopes = ann->prev_train_slopes;
  fann_type prev_step, slope, prev_slope, next_step, same_sign;

  float increase_factor = ann->rprop_increase_factor; /*1.2; */
  float decrease_factor = ann->rprop_decrease_factor; /*0.5; */
  float delta_min = ann->rprop_delta_min;             /*0.0; */
  float delta_max = ann->rprop_delta_max;             /*50.0; */

  unsigned int i = first_weight;

  for (; i != past_end; i++) {
  	// printf("3\n");
  	// printf("%f\n", prev_steps[i]);
    prev_step = fann_max(
        prev_steps[i],
        (fann_type)0.0001); /* prev_step may not be zero because then the training will stop */
  	// printf("4\n");
    slope = train_slopes[i];
    prev_slope = prev_train_slopes[i];

    same_sign = prev_slope * slope;

    if (same_sign >= 0.0)
      next_step = fann_min(prev_step * increase_factor, delta_max);
    else {
      next_step = fann_max(prev_step * decrease_factor, delta_min);
      slope = 0;
    }

    if (slope < 0) {
      weights[i] -= next_step;
      if (weights[i] < -1500) weights[i] = -1500;
    } else {
      weights[i] += next_step;
      if (weights[i] > 1500) weights[i] = 1500;
    }

    /*if(i == 2){
     * printf("weight=%f, slope=%f, next_step=%f, prev_step=%f\n", weights[i], slope, next_step,
     * prev_step);
     * } */

    /* update global data arrays */
    prev_steps[i] = next_step;
    prev_train_slopes[i] = slope;
    train_slopes[i] = 0.0;
  }
}

void fann_update_slopes_batch_custom(struct fann *ann, struct fann_layer *layer_begin,
                              struct fann_layer *layer_end, fann_type *dev_slopes, fann_type *dev_errors) {
  // struct fann_neuron *last_neuron, *prev_neurons, **connections;
  // fann_type tmp_error;
  // unsigned int i, num_connections;

  /* store some variabels local for fast access */
  // struct fann_neuron *first_neuron = ann->first_layer->first_neuron;
  // fann_type *error_begin = ann->train_errors;
  // fann_type *slope_begin, *neuron_slope;

  /* if no room allocated for the slope variabels, allocate it now */

  if (layer_begin == NULL) {
    layer_begin = ann->first_layer + 1;
  }

  if (layer_end == NULL) {
    layer_end = ann->last_layer - 1;
  }

  // slope_begin = ann->train_slopes;

#ifdef DEBUGTRAIN
  printf("\nupdate slopes\n");
#endif

  // prev_neurons = first_neuron;

  for (; layer_begin <= layer_end; layer_begin++) {
#ifdef DEBUGTRAIN
    printf("layer[%d]\n", layer_begin - ann->first_layer);
#endif
    // last_neuron = layer_begin->last_neuron;

    int neurons_in_layer = layer_begin->last_neuron - layer_begin->first_neuron;
    int neurons_in_prev_layer = (layer_begin-1)->last_neuron - (layer_begin-1)->first_neuron;

    // if (ann->network_type == FANN_NETTYPE_LAYER) {
    // prev_neurons = (layer_begin - 1)->first_neuron;
    // }
    struct fann_neuron *dev_neurons, *dev_last_layer_neurons;
    check(cudaMalloc((void **)&dev_neurons, neurons_in_layer*sizeof(struct fann_neuron)));
  	check(cudaMemcpy(dev_neurons, layer_begin->first_neuron, neurons_in_layer*sizeof(struct fann_neuron), cudaMemcpyHostToDevice));

  	check(cudaMalloc((void **)&dev_last_layer_neurons, neurons_in_prev_layer*sizeof(struct fann_neuron)));
  	check(cudaMemcpy(dev_last_layer_neurons, (layer_begin-1)->first_neuron, neurons_in_prev_layer*sizeof(struct fann_neuron), cudaMemcpyHostToDevice));

  	// position of the first neuron in this layer == total neurons upto this layer
  	int first_neuron_number = 0;
		struct fann_layer *l_it;
	  for (l_it = layer_begin-1; l_it >= ann->first_layer; l_it--) {
	  	first_neuron_number += l_it->last_neuron - l_it->first_neuron;
	  }
    // printf("%d\n", neurons_in_layer);
	  update_slopes<<<1, (int)(neurons_in_layer/scale_threads)>>>(dev_neurons, dev_last_layer_neurons, first_neuron_number, dev_slopes, dev_errors, neurons_in_layer);

    // for (neuron_it = layer_begin->first_neuron; neuron_it != last_neuron; neuron_it++) {
    // }
    cudaFree(dev_neurons);
    cudaFree(dev_last_layer_neurons);
  }
}

__global__ void update_slopes(struct fann_neuron *dev_neurons, struct fann_neuron *dev_last_layer_neurons, int first_neuron_number, 
	fann_type *dev_slopes, fann_type *dev_errors, int total_work) {
  // int work_per_thread = total_work/blockDim.x;
  // for (int work = 0; work < total_work; ++work)
  // {
  	struct fann_neuron *neuron_it;
  	fann_type *neuron_slope, tmp_error;
  	unsigned int i;
  	int num_connections;
  	neuron_it = dev_neurons + threadIdx.x;
    // neuron_it = dev_neurons + threadIdx.x + work_per_thread*work;
  	tmp_error = dev_errors[first_neuron_number + threadIdx.x];
    // tmp_error = dev_errors[first_neuron_number + threadIdx.x*work_per_thread + work];
    neuron_slope = dev_slopes + neuron_it->first_con;
    num_connections = neuron_it->last_con - neuron_it->first_con;
    for (i = 0; i != num_connections; i++) {
      atomicAdd(&neuron_slope[i], tmp_error * dev_last_layer_neurons[i].value);
    }
  // }
}


fann_type fann_update_MSE(struct fann *ann, struct fann_neuron *neuron, fann_type neuron_diff) {
  float neuron_diff2;

  switch (neuron->activation_function) {
    case FANN_LINEAR_PIECE_SYMMETRIC:
    case FANN_THRESHOLD_SYMMETRIC:
    case FANN_SIGMOID_SYMMETRIC:
    case FANN_SIGMOID_SYMMETRIC_STEPWISE:
    case FANN_ELLIOT_SYMMETRIC:
    case FANN_GAUSSIAN_SYMMETRIC:
    case FANN_SIN_SYMMETRIC:
    case FANN_COS_SYMMETRIC:
      neuron_diff /= (fann_type)2.0;
      break;
    case FANN_THRESHOLD:
    case FANN_LINEAR:
    case FANN_SIGMOID:
    case FANN_SIGMOID_STEPWISE:
    case FANN_GAUSSIAN:
    case FANN_GAUSSIAN_STEPWISE:
    case FANN_ELLIOT:
    case FANN_LINEAR_PIECE:
    case FANN_SIN:
    case FANN_COS:
      break;
  }

#ifdef FIXEDFANN
  neuron_diff2 = (neuron_diff / (float)ann->multiplier) * (neuron_diff / (float)ann->multiplier);
#else
  neuron_diff2 = (float)(neuron_diff * neuron_diff);
#endif

  ann->MSE_value += neuron_diff2;

  /*printf("neuron_diff %f = (%f - %f)[/2], neuron_diff2=%f, sum=%f, MSE_value=%f, num_MSE=%d\n",
   * neuron_diff, *desired_output, neuron_value, neuron_diff2, last_layer_begin->sum,
   * ann->MSE_value, ann->num_MSE); */
  if (fann_abs(neuron_diff) >= ann->bit_fail_limit) {
    ann->num_bit_fail++;
  }

  return neuron_diff;
}