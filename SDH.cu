/* ==================================================================
	Programmer: Yicheng Tu (ytu@cse.usf.edu)
	The basic SDH algorithm implementation for 3D data
	To compile: nvcc SDH.c -o SDH in the C4 lab machines

	CUDA Portion and kernels programmed by: Brandon Yates
   ==================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>


#define BOX_SIZE	23000 /* size of the data box on one dimension            */

/* descriptors for single atom in the tree */
typedef struct atomdesc {
	double x_pos;
	double y_pos;
	double z_pos;
} atom;

typedef struct hist_entry{
	//float min;
	//float max;
	unsigned long long d_cnt;   /* need a long long type as the count might be huge */
} bucket;


bucket * histogram;		/* list of all buckets in the histogram   */
long long	PDH_acnt;	/* total number of data points            */
int num_buckets;		/* total number of buckets in the histogram */
double   PDH_res;		/* value of w                             */
atom * atom_list;		/* list of all data points                */

/* These are for an old way of tracking time */
struct timezone Idunno;	
struct timeval startTime, endTime;


/* 
	distance of two points in the atom_list 
*/
double p2p_distance(int ind1, int ind2) {
	
	double x1 = atom_list[ind1].x_pos;
	double x2 = atom_list[ind2].x_pos;
	double y1 = atom_list[ind1].y_pos;
	double y2 = atom_list[ind2].y_pos;
	double z1 = atom_list[ind1].z_pos;
	double z2 = atom_list[ind2].z_pos;
		
	return sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
}

/*Kernel of p2p distance*/
__device__ double p2p_distance_kernel(atom* atom_list, int ind1, int ind2){
	double x1 = atom_list[ind1].x_pos;
	double x2 = atom_list[ind2].x_pos;
	double y1 = atom_list[ind1].y_pos;
	double y2 = atom_list[ind2].y_pos;
	double z1 = atom_list[ind1].z_pos;
	double z2 = atom_list[ind2].z_pos;
		
	return sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
}


/* 
	brute-force SDH solution in a single CPU thread 
*/
int PDH_baseline() {
	int i, j, h_pos;
	double dist;
	
	for(i = 0; i < PDH_acnt; i++) {
		for(j = i+1; j < PDH_acnt; j++) {
			dist = p2p_distance(i,j);
			h_pos = (int) (dist / PDH_res);
			histogram[h_pos].d_cnt++;
		} 
	}
	return 0;
}

/*baseline kernel function*/
__global__ void PDH_baseline_kernel(bucket *histogram, atom *atom_list, double width, int size)
{
	int i, j, h_pos;
	double distance;
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	i = x + 1;

	for(j = i; j < size; j++)
	{
		distance = p2p_distance_kernel(atom_list, x, j);
		h_pos = (int) (distance / width);
		atomicAdd(&histogram[h_pos].d_cnt, 1);
	}	
}



/* 
	set a checkpoint and show the (natural) running time in seconds 
*/
double report_running_time(int i) {
	long sec_diff, usec_diff;
	gettimeofday(&endTime, &Idunno);
	sec_diff = endTime.tv_sec - startTime.tv_sec;
	usec_diff= endTime.tv_usec-startTime.tv_usec;
	if(usec_diff < 0) {
		sec_diff --;
		usec_diff += 1000000;
	}
	if(i == 0)
		printf("Running time for CPU version: %ld.%06ld\n", sec_diff, usec_diff);
	else
		printf("Running time for GPU version: %ld.%06ld\n", sec_diff, usec_diff);

	return (double)(sec_diff*1.0 + usec_diff/1000000.0);
}


/* 
	print the counts in all buckets of the histogram 
*/
void output_histogram(bucket *histogram){
	int i; 
	long long total_cnt = 0;
	for(i=0; i< num_buckets; i++) {
		if(i%5 == 0) /* we print 5 buckets in a row */
			printf("\n%02d: ", i);
		printf("%15lld ", histogram[i].d_cnt);
		total_cnt += histogram[i].d_cnt;
	  	/* we also want to make sure the total distance count is correct */
		if(i == num_buckets - 1)	
			printf("\n T:%lld \n", total_cnt);
		else printf("| ");
	}
}

/*searches for differences between gpu and host arrays*/
void difference(bucket *a, bucket *b, bucket *c)
{
	int i, difference, found;
	difference = 0;
	//found = 0;
	for(i = 0; i <= num_buckets; i++)
	{

		if (a[i].d_cnt != b[i].d_cnt)
		{
			//found = 1;
			difference = a[i].d_cnt - b[i].d_cnt;
			if(difference < 0)
				difference = difference * -1;
			//printf("Difference detected in bucket %d: %d", i, difference);
			c[i].d_cnt+= difference;
		}
		else
		{
			c[i].d_cnt = 0;
		}

		difference = 0;	
	}
	//if(found == 0)
		//printf("NO DIFFERENCES FOUND BETWEEN HISTOGRAMS.\n");
}


//argv[1] is number of atoms and argv[2] is distance
int main(int argc, char **argv)
{
	int i;
	int hw;//indicates if CPU or GPU version is being run (0 = CPU)
	PDH_acnt = atoi(argv[1]);
	PDH_res = atof(argv[2]);
//printf("args are %d and %f\n", PDH_acnt, PDH_res);

	num_buckets = (int)(BOX_SIZE * 1.732 / PDH_res) + 1;
	histogram = (bucket *)malloc(sizeof(bucket)*num_buckets);
	atom_list = (atom *)malloc(sizeof(atom)*PDH_acnt);

	
	srand(1);
	/* generate data following a uniform distribution */
	for(i = 0;  i < PDH_acnt; i++) {
		atom_list[i].x_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_list[i].y_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_list[i].z_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
	}
	/* start counting time */
	gettimeofday(&startTime, &Idunno);
	
	/* call CPU single thread version to compute the histogram */
	PDH_baseline();
	
	/* check the total running time */ 
	hw = 0;
	report_running_time(hw);
	
	/* print out the histogram */
	output_histogram(histogram);
	
	/*ADDED CODE BELOW*/
	atom *d_x;
	bucket *d_out;
	bucket *second_hist;
	second_hist = (bucket *)malloc(sizeof(bucket)*num_buckets);
	memcpy(second_hist, histogram, sizeof(bucket)*num_buckets);
	
	/*define cuda array*/
	cudaMalloc(&d_x, sizeof(atom)*PDH_acnt);//atom list
	cudaMalloc(&d_out, sizeof(bucket)*num_buckets);//histogram
	//cudaMalloc(&second_hist, sizeof(bucket)*num_buckets);//histogram

	/*Copies data from host to device*/
	cudaMemcpy(d_x, atom_list, sizeof(atom)*PDH_acnt, cudaMemcpyHostToDevice);
	cudaMemcpy(d_out, histogram, sizeof(bucket)*num_buckets, cudaMemcpyHostToDevice);

	/* start counting time */
	gettimeofday(&startTime, &Idunno);

	/*launch kernel*/
	PDH_baseline_kernel<<<ceil(PDH_acnt/32), 32>>>(d_out, d_x, PDH_res, PDH_acnt);

	/* check the total running time */ 
	hw = 1;
	report_running_time(hw);

	/*copy cuda array to host array*/
	cudaMemcpy(histogram, d_out, sizeof(bucket)*PDH_acnt, cudaMemcpyDeviceToHost);
	
	/* print out the histogram */
	output_histogram(histogram);

	/*Print any differences*/
	printf("DIFFERENCES IN HISTOGRAMS: \n");
	bucket * diff = (bucket *)malloc(sizeof(bucket)*num_buckets);
	difference(histogram, second_hist, diff);
	output_histogram(diff);
	
	/*Free memory*/
	cudaFree(d_x);
	cudaFree(d_out);
	free(histogram);
	free(atom_list);
	free(second_hist);
	free(diff);
	return 0;
}







