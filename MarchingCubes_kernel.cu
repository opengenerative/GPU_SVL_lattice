/*

Reference - https://paulbourke.net/geometry/polygonise/

Reference - https://github.com/NVIDIA/cuda-samples/tree/master/Samples/5_Domain_Specific/marchingCubes

*/


#include <stdio.h>
#include <string.h>
#include <helper_cuda.h>    
#include <helper_math.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include "tables.h"
#include "MarchingCubes_kernel.h"

cudaTextureObject_t triTex_s;
cudaTextureObject_t triTex_t;
cudaTextureObject_t numVertsTex_s;


void MarchingCubeCuda::allocateTextures_s(uint **d_triTable,  uint **d_numVertsTable)
{

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);

    checkCudaErrors(cudaMalloc((void **) d_triTable, 256*16*sizeof(uint)));
    checkCudaErrors(cudaMemcpy((void *)*d_triTable, (void *)triTable, 256*16*sizeof(uint), cudaMemcpyHostToDevice));

    cudaResourceDesc            texRes;
    memset(&texRes,0,sizeof(cudaResourceDesc));

    texRes.resType                = cudaResourceTypeLinear;
    texRes.res.linear.devPtr      = *d_triTable;
    texRes.res.linear.sizeInBytes = 256*16*sizeof(uint);
    texRes.res.linear.desc        = channelDesc;

    cudaTextureDesc             texDescr;
    memset(&texDescr,0,sizeof(cudaTextureDesc));

    texDescr.normalizedCoords = false;
    texDescr.filterMode       = cudaFilterModePoint;
    texDescr.addressMode[0] = cudaAddressModeClamp;
    texDescr.readMode = cudaReadModeElementType;

    checkCudaErrors(cudaCreateTextureObject(&triTex_s, &texRes, &texDescr, NULL));

    checkCudaErrors(cudaMalloc((void **) d_numVertsTable, 256*sizeof(uint)));
    checkCudaErrors(cudaMemcpy((void *)*d_numVertsTable, (void *)numVertsTable, 256*sizeof(uint), cudaMemcpyHostToDevice));

    memset(&texRes,0,sizeof(cudaResourceDesc));

    texRes.resType                = cudaResourceTypeLinear;
    texRes.res.linear.devPtr      = *d_numVertsTable;
    texRes.res.linear.sizeInBytes = 256*sizeof(uint);
    texRes.res.linear.desc        = channelDesc;

    memset(&texDescr,0,sizeof(cudaTextureDesc));

    texDescr.normalizedCoords = false;
    texDescr.filterMode       = cudaFilterModePoint;
    texDescr.addressMode[0] = cudaAddressModeClamp;
    texDescr.readMode = cudaReadModeElementType;

    checkCudaErrors(cudaCreateTextureObject(&numVertsTex_s, &texRes, &texDescr, NULL));
}


void MarchingCubeCuda::destroyAllTextureObjects()
{
    checkCudaErrors(cudaDestroyTextureObject(triTex_s));
    checkCudaErrors(cudaDestroyTextureObject(triTex_t));
    checkCudaErrors(cudaDestroyTextureObject(numVertsTex_s));
}


__device__
float sampleVolume(float *data, uint3 p, uint3 gridSize)
{
    p.x = min(p.x, gridSize.x);
    p.y = min(p.y, gridSize.y);
    p.z = min(p.z, gridSize.z);
    uint i = (p.z*gridSize.x*gridSize.y) + (p.y*gridSize.x) + p.x;
    return (float) data[i];
}


__device__
uint3 calcGridPos(uint i, uint3 gridSizeShift, uint3 gridSizeMask)
{
    uint3 gridPos;
    
    uint z_quo = i / gridSizeShift.z;
    uint z_rem = i % gridSizeShift.z;
    uint y_quo = (z_rem)/gridSizeShift.y;
    uint x_rem = (z_rem) % gridSizeShift.y;

    gridPos.x = x_rem;
    gridPos.y = y_quo;
    gridPos.z = z_quo; 

    return gridPos;
}

__global__ void
classifyVoxel(uint *voxelVerts, uint *voxelOccupied, float *volume,
              uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask, uint numVoxels,
              float3 voxelSize, float isoValue, cudaTextureObject_t numVertsTex)
{
    uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
    uint i = __mul24(blockId, blockDim.x) + threadIdx.x;

    if (i < numVoxels)
    {
        uint3 gridPos = calcGridPos(i, gridSizeShift, gridSizeMask);
    
        float field[8];
        field[0] = sampleVolume(volume, gridPos, gridSize);
        field[1] = sampleVolume(volume, gridPos + make_uint3(1, 0, 0), gridSize);
        field[2] = sampleVolume(volume, gridPos + make_uint3(1, 1, 0), gridSize);
        field[3] = sampleVolume(volume, gridPos + make_uint3(0, 1, 0), gridSize);
        field[4] = sampleVolume(volume, gridPos + make_uint3(0, 0, 1), gridSize);
        field[5] = sampleVolume(volume, gridPos + make_uint3(1, 0, 1), gridSize);
        field[6] = sampleVolume(volume, gridPos + make_uint3(1, 1, 1), gridSize);
        field[7] = sampleVolume(volume, gridPos + make_uint3(0, 1, 1), gridSize);

        float isoVal = isoValue;
     
        uint cubeindex;
        cubeindex =  uint(field[0] < (isoVal));
        cubeindex += uint(field[1] < (isoVal))*2;
        cubeindex += uint(field[2] < (isoVal))*4;
        cubeindex += uint(field[3] < (isoVal))*8;
        cubeindex += uint(field[4] < (isoVal))*16;
        cubeindex += uint(field[5] < (isoVal))*32;
        cubeindex += uint(field[6] < (isoVal))*64;
        cubeindex += uint(field[7] < (isoVal))*128;
        uint numVerts = tex1Dfetch<uint>(numVertsTex, cubeindex);

        voxelVerts[i] = numVerts;

        voxelOccupied[i] = (numVerts > 0);
 
    }
  
 
}

void MarchingCubeCuda::classifyVoxel_lattice(dim3 grid, dim3 threads, uint *voxelVerts, uint *voxelOccupied, float *volume,
                     uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask, uint numVoxels,
                     float3 voxelSize, float isoValue)
{

   
    classifyVoxel<<<grid, threads>>>(voxelVerts, voxelOccupied, volume,
                                     gridSize, gridSizeShift, gridSizeMask,
                                     numVoxels, voxelSize, isoValue, numVertsTex_s);
    cudaDeviceSynchronize();

    getLastCudaError("classifyVoxel failed");

   
}


__global__ void
compactVoxels(uint *compactedVoxelArray, uint *voxelOccupied, uint *voxelOccupiedScan, uint numVoxels)
{
    uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
    uint i = __mul24(blockId, blockDim.x) + threadIdx.x;

    if(i < numVoxels)
    {
        if (voxelOccupied[i])
        {
            compactedVoxelArray[ voxelOccupiedScan[i] ] = i;
        }
    
    }
}

void MarchingCubeCuda::compactVoxels_lattice(dim3 grid, dim3 threads, uint *compactedVoxelArray, uint *voxelOccupied, uint *voxelOccupiedScan, uint numVoxels)
{
    compactVoxels<<<grid, threads>>>(compactedVoxelArray, voxelOccupied,
                                     voxelOccupiedScan, numVoxels);
    getLastCudaError("compactVoxels failed");
}



__device__
float3  vertexInterp3(float isolevel, float3 p0, float3 p1, float f0, float f1)
{
    
    if (f1 < f0)
    {
        float3 temp;
        temp = p1;
        p1 = p0;
        p0 = temp;    

        float tm;
        tm = f1;
        f1 = f0;
        f0 = tm;
    }


    float a = isolevel - 0.1;

    float c = isolevel + 0.1;
    
    float t;

    if((f1 > a) && (f0 < a))
    {
        if (fabs(a-f0) < 0.0005)
        {
            return(p0);
        }
        if (fabs(a-f1) < 0.0005)
        {
            return(p1);
        }
        if (fabs(f1-f0) < 0.0005)
        {
            return(p0);
        }

        t = (a - f0) / (f1 - f0);
    }



    else if((f1 > c) && (f0 < c))
    {
       

        if (fabs(c-f0) < 0.0005)
        {
            return(p0);
        }
        if (fabs(c-f1) < 0.0005)
        {
            return(p1);
        }
        if (fabs(f1-f0) < 0.0005)
        {
            return(p0);
        }

        t = (c - f0) / (f1 - f0);
    }


    else if(((f1 >a ) && (f0 > a)) && ((f1 < c) && (f0 <c )))
    {
        t = 0.5;
    }
    
    return lerp(p0, p1, t);
}


__device__
float3 calcNormal(float3 *v0, float3 *v1, float3 *v2)
{
    float3 edge0 = *v1 - *v0;
    float3 edge1 = *v2 - *v0;
    return cross(edge0, edge1);
}


__global__ void
generateTriangles_lattice_kernel(float4 *pos, float4 *norm, uint *compactedVoxelArray, uint *numVertsScanned, float *volume,
                   uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask,
                   float3 voxelSize, float3 gridcenter, float isoValue, uint activeVoxels, uint maxVerts,
                   cudaTextureObject_t triTex, cudaTextureObject_t numVertsTex,uint totalverts, float *volume_one)
{
    
    
    uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
    uint i = __mul24(blockId, blockDim.x) + threadIdx.x;
    
    if (i < activeVoxels)
    {
             
        uint voxel = compactedVoxelArray[i];
        
        uint3 gridPos = calcGridPos(voxel, gridSizeShift, gridSizeMask);

        float3 p;

        p.x = (gridPos.x - gridcenter.x) *voxelSize.x ;
        p.y = (gridPos.y - gridcenter.y) *voxelSize.y ;
        p.z = (gridPos.z - gridcenter.z) *voxelSize.z ;
        
        float3 v[8];
        v[0] = p;
        v[1] = p + make_float3(voxelSize.x, 0, 0);
        v[2] = p + make_float3(voxelSize.x, voxelSize.y, 0);
        v[3] = p + make_float3(0, voxelSize.y, 0);
        v[4] = p + make_float3(0, 0, voxelSize.z);
        v[5] = p + make_float3(voxelSize.x, 0, voxelSize.z);
        v[6] = p + make_float3(voxelSize.x, voxelSize.y, voxelSize.z);
        v[7] = p + make_float3(0, voxelSize.y, voxelSize.z);


        
        float field[8];
        field[0] = sampleVolume(volume, gridPos, gridSize);
        field[1] = sampleVolume(volume, gridPos + make_uint3(1, 0, 0), gridSize);
        field[2] = sampleVolume(volume, gridPos + make_uint3(1, 1, 0), gridSize);
        field[3] = sampleVolume(volume, gridPos + make_uint3(0, 1, 0), gridSize);
        field[4] = sampleVolume(volume, gridPos + make_uint3(0, 0, 1), gridSize);
        field[5] = sampleVolume(volume, gridPos + make_uint3(1, 0, 1), gridSize);
        field[6] = sampleVolume(volume, gridPos + make_uint3(1, 1, 1), gridSize);
        field[7] = sampleVolume(volume, gridPos + make_uint3(0, 1, 1), gridSize);

    
        float isoVal = isoValue; 
        
        uint cubeindex;
        cubeindex =  uint(field[0] < isoVal);
        cubeindex += uint(field[1] < isoVal)*2;
        cubeindex += uint(field[2] < isoVal)*4;
        cubeindex += uint(field[3] < isoVal)*8;
        cubeindex += uint(field[4] < isoVal)*16;
        cubeindex += uint(field[5] < isoVal)*32;
        cubeindex += uint(field[6] < isoVal)*64;
        cubeindex += uint(field[7] < isoVal)*128;
        
        field[0] = sampleVolume(volume_one, gridPos, gridSize);
        field[1] = sampleVolume(volume_one, gridPos + make_uint3(1, 0, 0), gridSize);
        field[2] = sampleVolume(volume_one, gridPos + make_uint3(1, 1, 0), gridSize);
        field[3] = sampleVolume(volume_one, gridPos + make_uint3(0, 1, 0), gridSize);
        field[4] = sampleVolume(volume_one, gridPos + make_uint3(0, 0, 1), gridSize);
        field[5] = sampleVolume(volume_one, gridPos + make_uint3(1, 0, 1), gridSize);
        field[6] = sampleVolume(volume_one, gridPos + make_uint3(1, 1, 1), gridSize);
        field[7] = sampleVolume(volume_one, gridPos + make_uint3(0, 1, 1), gridSize);

        __shared__ float3 vertlist[12*NTHREADS];
        vertlist[threadIdx.x] = vertexInterp3(isoValue, v[0], v[1], field[0], field[1]);
        vertlist[NTHREADS+threadIdx.x] = vertexInterp3(isoValue, v[1], v[2], field[1], field[2]);
        vertlist[(NTHREADS*2)+threadIdx.x] = vertexInterp3(isoValue, v[2], v[3], field[2], field[3]);
        vertlist[(NTHREADS*3)+threadIdx.x] = vertexInterp3(isoValue, v[3], v[0], field[3], field[0]);
        vertlist[(NTHREADS*4)+threadIdx.x] = vertexInterp3(isoValue, v[4], v[5], field[4], field[5]);
        vertlist[(NTHREADS*5)+threadIdx.x] = vertexInterp3(isoValue, v[5], v[6], field[5], field[6]);
        vertlist[(NTHREADS*6)+threadIdx.x] = vertexInterp3(isoValue, v[6], v[7], field[6], field[7]);
        vertlist[(NTHREADS*7)+threadIdx.x] = vertexInterp3(isoValue, v[7], v[4], field[7], field[4]);
        vertlist[(NTHREADS*8)+threadIdx.x] = vertexInterp3(isoValue, v[0], v[4], field[0], field[4]);
        vertlist[(NTHREADS*9)+threadIdx.x] = vertexInterp3(isoValue, v[1], v[5], field[1], field[5]);
        vertlist[(NTHREADS*10)+threadIdx.x] = vertexInterp3(isoValue, v[2], v[6], field[2], field[6]);
        vertlist[(NTHREADS*11)+threadIdx.x] = vertexInterp3(isoValue, v[3], v[7], field[3], field[7]);
        


        uint numVerts = tex1Dfetch<uint>(numVertsTex, cubeindex);

        for (int j =0; j<numVerts; j += 3)
        {
            uint index;
            
            index = numVertsScanned[voxel] + j;
            
            float3 *v[3];

            uint edge;

            edge = tex1Dfetch<uint>(triTex, (cubeindex*16) + j);

            
            v[0] = &vertlist[(edge*NTHREADS)+threadIdx.x];

            edge = tex1Dfetch<uint>(triTex, (cubeindex*16) + j + 1);

            
            v[1] = &vertlist[(edge*NTHREADS)+threadIdx.x];
            

            edge = tex1Dfetch<uint>(triTex, (cubeindex*16) + j + 2);
            
            v[2] = &vertlist[(edge*NTHREADS)+threadIdx.x];
            

            float3 n = calcNormal(v[0], v[1], v[2]);
        
            if (index < (maxVerts - 3))
            {
        
                pos[index] = make_float4(*v[0], 1.0f);
                norm[index] = make_float4(n, 0.0f);

                pos[index+1] = make_float4(*v[1], 1.0f);
                norm[index+1] = make_float4(n, 0.0f);

                pos[index+2] = make_float4(*v[2], 1.0f);
                norm[index+2] = make_float4(n, 0.0f);    
            
            }
        }
        
    }


    
}

void MarchingCubeCuda::generateTriangles_lattice(dim3 grid, dim3 threads,
                          float4 *pos, float4 *norm, uint *compactedVoxelArray, uint *numVertsScanned, float *volume,
                          uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask,
                          float3 voxelSize, float3 gridcenter, float isoValue, uint activeVoxels, uint maxVerts, uint totalverts,float *volume_one)
{
    
    generateTriangles_lattice_kernel<<<grid, threads>>>(pos, norm,
                                           compactedVoxelArray,
                                           numVertsScanned, volume,
                                           gridSize, gridSizeShift, gridSizeMask,
                                           voxelSize,gridcenter, isoValue, activeVoxels,
                                           maxVerts, triTex_s, numVertsTex_s, totalverts, volume_one);
    cudaDeviceSynchronize();
    getLastCudaError("generateTriangles failed");
    cudaError_t err = cudaGetLastError();
    
}


void MarchingCubeCuda::ThrustScanWrapper_lattice(unsigned int *output, unsigned int *input, unsigned int numElements)
{
    thrust::exclusive_scan(thrust::device_ptr<unsigned int>(input),
                           thrust::device_ptr<unsigned int>(input + numElements),
                           thrust::device_ptr<unsigned int>(output));
}

