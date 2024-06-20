
#include "Fft_lattice.h"
#include <cufft.h>
#include <helper_cuda.h>

extern cufftHandle planr2c;
extern cufftHandle planc2r;


__global__ void create_lattice_kernel(float *d_latticevol,uint NX, uint NY, uint NZ, uint size)
{


	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	uint index = x + (y * NX) + (z * (NX * NY));
	float a;

	if((x < NX) && (y < NY ) && ( z < NZ))
	{
		
		float xx = (x * 1.0)/NX;
		float yy = (y * 1.0)/NY;
		float zz = (z * 1.0)/NZ;
		a = cosf(6.28 * xx)*sinf(6.28 * yy) + cosf(6.28 * yy) * sinf(6.28 * zz) + cosf(6.28* zz) * sinf(6.28 * xx);

		d_latticevol[index] = a;
		
		__syncthreads();

	}

};

void Fft_lattice::create_lattice(float *d_latticevol, uint NX, uint NY, uint NZ, uint size)
{
	dim3 grids(ceil((NX)/float(16)),ceil((NY)/float(8)),ceil((NZ)/float(8)));
	dim3 tids(16,8,8);
	create_lattice_kernel<<<grids,tids>>>(d_latticevol,NX,NY,NZ,size);
	cudaDeviceSynchronize();
}


void normalise_bufferr(float *dataone, float *datatwo, size_t size)
{
	float *h_B;
	h_B = (float *)malloc((size) * sizeof(*dataone));
	cudaMemcpy(h_B, dataone, (size) * sizeof(*dataone), cudaMemcpyHostToHost);
	float a,b;
	for (int i=0;i<size;i++)
	{
		if(i==0)
		{
			a = h_B[i];
			b = h_B[i];
		}
		
		a=min(a,h_B[i]);
		b =max(b,h_B[i]);
	}

	for(int i=0;i<size;i++)
	{
		h_B[i] = (h_B[i] - a)/(b-a);
	}

	cudaMemcpy(datatwo, h_B, (size) * sizeof(*h_B), cudaMemcpyHostToHost);
	free(h_B);
}




void Fft_lattice::fft_func(float2 *fft_data)
{

		checkCudaErrors(cufftExecR2C(planr2c, (cufftReal *)fft_data, (cufftComplex *)fft_data));

}

void Fft_lattice::ifft_func(float2 *fft_data)
{

	checkCudaErrors(cufftExecC2R(planc2r, (cufftComplex *)fft_data, (cufftReal *)fft_data));

}


__global__ void fft_scalar_kernel(float2 *fft_data_compute,float scalar_val,int size)
{


	int tx = blockIdx.x * blockDim.x + threadIdx.x;

	float2 a ;

	if(tx < size)
	{
		a = fft_data_compute[tx];

        a.x /= scalar_val;
        a.y /= scalar_val;
	
		fft_data_compute[tx] = a;

		__syncthreads();

	}

};


void Fft_lattice::fft_scalar(float2 *fft_data_compute,float scalar_val,int size)
{
    dim3 grids(ceil((size)/float(1024)),1,1);
	dim3 tids(1024,1,1);
	fft_scalar_kernel<<<grids,tids>>>(fft_data_compute,scalar_val,size);
	cudaDeviceSynchronize();
}



__global__ void fft_fill_kernel(float2 *fft_compute, float2 *fft_compute_fill,int Nx, int Ny , int Nz ,size_t size, uint Nx2)
{

		int tx = blockIdx.x * blockDim.x + threadIdx.x;

		if(tx < Nx2*Ny*Nz)
		{
			int z = tx/(Nx2*Ny);
			int y = (tx%(Nx2*Ny))/Nx2;
			int x = (tx%(Nx2*Ny))%Nx2;

			int e,f,g;

			if((x == 0) && (y == 0) && (z == 0))
			{
				fft_compute_fill[0] = fft_compute[0];
				
			}
			else
			{
			
				if(x == 0)
				{
					e = 0;
				}

				else if(x > 0)
				{
					e = Nx - x;	
				}

				if( y == 0)
				{
					f = 0;
				}
				else if(y > 0)
				{
					f = Nx - y;
				}

				if(z == 0)
				{
					g = 0;
				}
				else if(z > 0)
				{
					g = Nx - z;
				}

				int indd =  x + y *Nx2 + z * (Nx2 *Ny);
				int indd1 = x + y *Nx + z * (Nx*Ny);
				int indd2 = e + f *Nx + g * (Nx*Ny);
	
				fft_compute_fill[indd1] = fft_compute[indd];
				fft_compute_fill[indd2] = fft_compute[indd];
				fft_compute_fill[indd2].y *= -1;
			}
        }
		
    	
}

void Fft_lattice::fft_fill(float2 *fft_compute, float2 *fft_compute_fill,int Nx, int Ny , int Nz)
{
	
	uint Nx2 = floor(Nx/2.0) +1;
	size_t size = Nx2*Ny*Nz;
	dim3 grids(ceil((size)/float(1024)),1,1);
	dim3 tids(1024,1,1);
	fft_fill_kernel<<<grids,tids>>>(fft_compute,fft_compute_fill,Nx,Ny,Nz,size,Nx2);
	cudaDeviceSynchronize();
}

