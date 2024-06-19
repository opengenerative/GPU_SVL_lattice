
#pragma once
#ifndef _MARCHING_CUBES_KERNEL_H_
#define _MARCHING_CUBES_KERNEL_H_

#include <stdint.h>
#include <cuda_runtime_api.h>

#define NTHREADS 32

class MarchingCubeCuda
{
    
    public:

        void classifyVoxel_lattice(dim3 grid, dim3 threads, uint *voxelVerts, 
                            uint *voxelOccupied, float *volume,
                            uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask, 
                            uint numVoxels,float3 voxelSize, float isoValue);

        void compactVoxels_lattice(dim3 grid, dim3 threads, uint *compactedVoxelArray, 
                            uint *voxelOccupied,uint *voxelOccupiedScan, uint numVoxels);

        void generateTriangles_lattice(dim3 grid, dim3 threads,float4 *pos, float4 *norm, 
                            uint *compactedVoxelArray, uint *numVertsScanned, float *volume,
                            uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask,
                            float3 voxelSize,float3 gridcenter, float isoValue, uint activeVoxels, uint maxVerts, uint totalverts, float *volume_one);

        void allocateTextures_s(uint **d_triTable,  uint **d_numVertsTable);

        void destroyAllTextureObjects();
        
        void ThrustScanWrapper_lattice(unsigned int *output, unsigned int *input, unsigned int numElements);
};
#endif // _MARCHING_CUBES_KERNEL_H_