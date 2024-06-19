#pragma once


#ifndef __GRATINGS_H_
#define __GRATINGS_H_



class  Gratings
{
    int NX;
    int NY;
    int NZ;

    bool GPUInitialized = false;
    dim3 MatVecGrid;
    dim3 MatVecBlock;
    dim3 Scalar4Grid;
    dim3 Scalar4Block;

    float *d_ResReduction = NULL;

    public:
    Gratings(int NX,int NY,int NZ);
    ~Gratings();
    
    void angle_data(float *d_theta,int NX, int NY,int NZ,float dx,float dy,float dz,float mean_x,float mean_y,float mean_z,
    char axis);

    void period_data(float *d_period,int NX ,int NY, int NZ,float dx, float dy, float dz, float mean_x,float mean_y,float mean_z,
    char axis);

    void VecSMultAdd_lattice (float *d_v, float a1, float *d_w, const float a2,const int NX, const int NY, const int NZ);

    void finding_phi(float *d_phi,float *d_period,int x_dim, int y_dim, int z_dim, int i, int j, int k,float dx, float dy, float dz, char latticetype_one , char latticetype_two);

    void GPUCG_lattice(float *d_phi, const int iter, const int OptIter, const float EndRes, int &FinalIter, float &FinalRes);
    
    void GPUScalar_lattice(float *d_result,float *d_vec1,float *d_vec2,int n, int block_num);
    
    void normalise_buffer(float *dataone, float *datatwo, size_t size);

    void normalise_bufferone(float *dataone, float *datatwo, size_t size,int NX,int NY, int NZ);

    void normalise_buffertwo(float *dataone, float *datatwo, size_t size);

    void normalise_bufferthree(float *dataone, float *datatwo, size_t size, float a1, float b1);

    void normalise_bufferfour(float *dataone, float *datatwo, size_t size, int Nx, int Ny, int Nz, float isovalue);

    void GPUMatvec_lattice(float *d_d,float *d_q, int NX, int NY , int NZ);

    void grating(float2 *dvol,int NX2, int NY2, int NZ2, float dx2,float dy2,float dz2);

    void svl(float *d_svl,float2 *d_grating,int NX, int NY, int NZ, int indxx,float2 *data_fft);

    void fillfrac(float *d_svl, float *d_fillfrac, int NX, int NY, int NZ);

    void setupTexture(int dx, int dy, int dz);

    void copytotexture(float *d_phi,cudaPitchedPtr data_ptr, int NX, int NY, int NZ);

    void updateTexture(cudaPitchedPtr data_ptr);

    void deleteTexture();

    void GPUCleanUp ();

};


#endif