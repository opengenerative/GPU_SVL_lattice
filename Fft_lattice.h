#pragma once



#ifndef __FFT_LATTICE_H__
#define __FFT_LATTICE_H__


class Fft_lattice
{
    public:

    Fft_lattice();
    ~Fft_lattice();
    void create_lattice(float *d_latticevol, uint NX, uint NY, uint NZ, uint size);
    void fft_func(float2 *fft_data);
    void ifft_func(float2 *fft_data);
    void fft_scalar(float2 *fft_data_compute,float scalar_val,int size);
    void fft_fill(float2 *fft_compute, float2 *fft_compute_fill,int Nx, int Ny , int Nz);


};


#endif /* __FFT_LATTICE_H__ */