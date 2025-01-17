//__________________________________________________________________//
//								      //
//		This file includes all 				      //
//		the CUDA funtions used				      //
//								      //
//__________________________________________________________________//

// C Libraries
//extern "C"{
    #include "CFD/cfd.h"
//}

#include "aux.h"
#include <iostream>
#include <math.h>

__device__ float cuda_diff_x( float a, float b, int x, int w );
__device__ float cuda_diff_y( float a, float b, int y, int h );
__global__ void  global_grad( float *imgIn, int *imgDomain, float *v1, float *v2, int w, int h, int nc, int n, int FullImage );
__device__ float cuda_div_x( float a, float b, int x, int w );
__device__ float cuda_div_y( float a, float b, int y, int h );
__global__ void  global_div( float *v1, float *v2, float *imgOut, int w, int h, int nc, int n );
__global__ void  global_norm( float *imgIn, float *imgOut, int w, int h, int n );
__device__ int   check_color( float *c, float r, float g, float b );
__global__ void  global_detect_domain1( float *imgIn, int *imgDomain, int w, int h, int n );
__global__ void  global_detect_domain2( float *imgIn, int *imgDomain, int w, int h, int n );
__global__ void  global_detect_domain3( float *imgIn, int *imgDomain, int w, int h, int n );
__global__ void  global_vorticity( float *imgU, float *imgV, float *imgVorticity,  int *imgDomain,  int w, int h, int nc, int n, int FullImage );
__global__ void  global_solve_Poisson (float *imgOut, float *imgIn, float *initVorticity, float *rhs, int *mask_toInpaint, int w, int h, int nc, int n, float sor_theta, int redOrBlack);
__global__ void  global_reverse_sign(float *Image, int n);
void aniso_diff(float *imgIn, int *imgDomain, float *imgOut, int w, int h, int nc, float tau, int N, dim3 grid, dim3 block);
