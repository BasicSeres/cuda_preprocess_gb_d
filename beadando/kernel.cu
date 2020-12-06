
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

#define BLOCK_S 1000
#define BLOCK_NUM 40

__constant__  float maskgpu[9] = { 0.0113f, 0.0838f,0.0113f,0.0838f,0.6193f,0.0838f,0.0113f,0.0838f,0.0113f };//gauss mask

//CPU version tesztekhez a maszkok
float mask[9] = { 0.0113f, 0.0838f,0.0113f,0.0838f,0.6193f,0.0838f,0.0113f,0.0838f,0.0113f };

//memory layout usage CPU code
void gaussian_cpu(const unsigned char *data, unsigned char *new_img, const int *w, const int *h, const size_t *image_size)
{

	for (size_t i = 201; i < *image_size-200; i++)
	{
		if (i % 200 == 0 && i%200 == 199)
		{
			new_img[i] = data[i];
			continue;
		}
		float sum = 0;

		int idx = i-200;
		int counter = 0;
		for (int k = -1; k <= 1; k++)
		{

			for (int h = -1; h <= 1; h++)
			{

				sum += data[ idx + h] * mask[counter];
				counter++;

			}
			idx += 200;// offset with stride
		}
		new_img[i] = sum;
	}


}
void dilation_cpu(const unsigned char *data, unsigned char *new_img, const int *w, const int *h, const size_t *image_size)
{

	for (size_t i = 201; i < *image_size - 200; i++)
	{
		if (i % 200 == 0 && i % 200 == 199)
		{
			new_img[i] = data[i];
			continue;
		}
		float max = 0;

		int idx = i - 200;
		int counter = 0;
		for (int k = -1; k <= 1; k++)
		{

			for (int h = -1; h <= 1; h++)
			{
				if (data[idx+h]> max)
				{
					max = data[idx + h];
				}
				counter++;

			}
			idx += 200;
		}
		new_img[i] = max;
	}
}

void gaussian_cpu_call(const unsigned char *data, const int *w, const int *h, const int *n, const size_t *image_size, unsigned char *new_img)
{

	gaussian_cpu(data, new_img, w, h, image_size);
	stbi_write_png("cpu_g.png", *w, *h, *n, new_img, *w);
}
void dilation_cpu_call(const unsigned char *data, const int *w, const int *h, const int *n, const size_t *image_size, unsigned char *new_img)
{
	dilation_cpu(data, new_img, w, h, image_size);
	stbi_write_png("cpu_d.png", *w, *h, *n, new_img, *w);
}

//geometriai felbontas utan shared memory hasznalata parallel
__global__ void gaussian_para(unsigned char* done_img, unsigned char* imggpu)
{
	const int index = threadIdx.x + blockIdx.x * blockDim.x;
	const int rowsize = 200;
	const int lastrow_id = 800;

	float sum = 0;
	int idx = threadIdx.x - rowsize;
	int counter = 0;
	if (index < rowsize)
	{
		done_img[index] = imggpu[index];
		return;

	}

	if (index % rowsize == 0 && index % rowsize == rowsize-1)//ha oldalt lennenk akkor return
	{
		done_img[index] = imggpu[index];
		return;
	}
	if (index >= 40000-rowsize)//utolso sor eseten return
	{
		done_img[index] = imggpu[index];
		return;
	}

	
	__shared__ unsigned char local_matrix[1000];//lokalis matrix amit az adott blokkban mukodo szalaknak biztosit shared memoryt
	__shared__ unsigned char local_helper_f[200];//felette levo sor a masik blokkbol
	__shared__ unsigned char local_helper_a[200];//alatta levo sor a masik blokkbol

	local_matrix[threadIdx.x] = imggpu[index];
	if (threadIdx.x > lastrow_id)
	{
		local_helper_a[threadIdx.x-lastrow_id] = imggpu[index+rowsize];

	}
	if (threadIdx.x<rowsize)//az elso ketszasz elem soran masolodik be még az elozo blokk utolso sora
	{
		local_helper_f[threadIdx.x] = imggpu[index - rowsize];
	}

	for (int k = -1; k <= 1; k++)
	{	
		for (int h = -1; h <= 1; h++)
		{
			if (k == -1 && idx<=0  && threadIdx.x != 0)
			{
				sum += local_helper_f[threadIdx.x + h] * maskgpu[counter];
			}
			else if (k == 1 && idx>= BLOCK_S)
			{
				sum += local_helper_a[threadIdx.x + h] * maskgpu[counter];
			}
			else
			{
				sum += local_matrix[idx + h] * maskgpu[counter];
			}
			counter++;

		}
		idx += rowsize;// offset with stride
	}

	done_img[index] = sum;


}
//dilation with shared_memory
__global__ void dilation_para(unsigned char* done_img, unsigned char* imggpu)
{
	const int index = threadIdx.x + blockIdx.x * blockDim.x;
	const int rowsize = 200;
	const int lastrow_id = 800;

	float max = 0;
	int idx = threadIdx.x - rowsize;
	int counter = 0;
	if (index < rowsize)
	{
		done_img[index] = imggpu[index];
		return;//elso sor eseten return

	}

	if (index % rowsize == 0 && index % rowsize == rowsize - 1)//ha oldalt lennenk akkor return
	{
		done_img[index] = imggpu[index];
		return;
	}
	if (index >= 40000 - rowsize)//utolso sor eseten return
	{
		done_img[index] = imggpu[index];
		return;
	}


	__shared__ unsigned char local_matrix[1000];//lokalis matrix amit az adott blokkban mukodo szalaknak biztosit shared memoryt
	__shared__ unsigned char local_helper_f[200];//felette levo sor a masik blokkbol
	__shared__ unsigned char local_helper_a[200];//alatta levo sor a masik blokkbol

	local_matrix[threadIdx.x] = imggpu[index];
	if (threadIdx.x > lastrow_id)
	{
		local_helper_a[threadIdx.x - lastrow_id] = imggpu[index + rowsize];

	}
	if (threadIdx.x<rowsize)//az elso ketszasz elem soran masolodik be még az elozo blokk utolso sora
	{
		//local_helper_a[y] = imggpu[i];
		local_helper_f[threadIdx.x] = imggpu[index - rowsize];
	}
	for (int k = -1; k <= 1; k++)
	{
		for (int h = -1; h <= 1; h++)
		{
			if (k == -1 && idx <= 0 && threadIdx.x != 0)
			{
				if (local_helper_f[threadIdx.x + h] > max) max = local_helper_f[threadIdx.x + h];
			}
			else if (k == 1 && idx >= BLOCK_S)
			{
				if (local_helper_a[threadIdx.x + h] > max) max = local_helper_a[threadIdx.x + h];
			}
			else
			{
				if (local_matrix[idx+h]> max)
				{
					max = local_matrix[idx + h];
				}
			}
			counter++;

		}
		idx += rowsize;// offset with stride
	}

	done_img[index] = max;


}
//dilation without shared memory
__global__ void dilation_para2(unsigned char* done_img, unsigned char* imggpu)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;


	if (index < 200) return;
	if (index % 200 == 0 && index % 200 == 199)
	{
		done_img[index] = imggpu[index];
		return;
	}
	if (index > 39800)
	{
		done_img[index] = imggpu[index];
		return;
	}
	float max = 0;
	int idx = index - 200;
	int counter = 0;
	for (int k = -1; k <= 1; k++)
	{

		for (int h = -1; h <= 1; h++)
		{
			if (imggpu[idx + h] > max)
			{
				max = imggpu[idx + h];
			}
			counter++;

		}
		idx += 200;// offset with stride
	}
	done_img[index] = max;


}
//shared memoryt nem hasznalo paralell code
__global__ void gaussian_para2(unsigned char* done_img, unsigned char* imggpu)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;


	if (index < 200) return;
	if (index % 200 == 0 && index % 200 == 199)
	{
		done_img[index] = imggpu[index];
		return;
	}
	if (index > 39800)
	{
		done_img[index] = imggpu[index];
		return;
	}
	float sum = 0;
	int idx = index - 200;
	int counter = 0;
	for (int k = -1; k <= 1; k++)
	{

		for (int h = -1; h <= 1; h++)
		{
			sum += imggpu[idx + h] * maskgpu[counter];

			counter++;

		}
		idx += 200;// offset with stride
	}
	done_img[index] = sum;


}

int main()
{
	char inputstr[30] = "./test_imgs/og_1.png";
	int w, h, n;
	//beolvasas
	unsigned char *data = stbi_load(inputstr, &w, &h, &n, 0);

	//device data lefoglalas, data atlmasolas
	unsigned char* dev_data;
	cudaMalloc((void**)&dev_data, w*h*n*(sizeof(unsigned char)));
	cudaMemcpy(dev_data, data, w*h*n*(sizeof(unsigned char)), cudaMemcpyHostToDevice);

	size_t image_size =(w*h*n);
	//host new img lefoglalas
	unsigned char *new_img = (unsigned char*)malloc(image_size);

	//device new img letrehozasa, lefoglalasa
	unsigned char* dev_newimg;
	cudaMalloc((void**)&dev_newimg, image_size*sizeof(unsigned char));

	//CPU TIME MEASURE
	clock_t t;
	t = clock();
	gaussian_cpu_call(data, &w, &h, &n, &image_size, new_img);
	t = clock() - t;
	double time = ((double)t) / CLOCKS_PER_SEC;
	time = time * 1000;
	printf("CPU: %f ms \n", time);

	//CUDA TIME MEASURE//////////////////////////
	cudaEvent_t start, stop;
	float elapsedTime;
	cudaEventCreate(&start);
	cudaEventRecord(start, 0);


	//dynamic version

	dilation_para2 << < BLOCK_NUM, BLOCK_S >> > (dev_newimg, dev_data);
	//futás közben már fellehet szabadítani a memoriat
	stbi_image_free(data);
	cudaFree(dev_data);
	gaussian_para << < BLOCK_NUM, BLOCK_S >> > (dev_newimg, dev_newimg);


	cudaMemcpy(new_img, dev_newimg, image_size*(sizeof(unsigned char)), cudaMemcpyDeviceToHost);

	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	printf("GPU: %f ms\n", elapsedTime);
	//////////////////////////////////////////////
	
	stbi_write_png("dilate_3.png", w, h, n, new_img, w);

	stbi_image_free(new_img);
	cudaFree(dev_newimg);
	cudaFree(dev_data);
	return(0);
}