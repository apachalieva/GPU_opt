#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "helper.h"
#include "functions.h"

//__________________________________________________________________//
//								      //
//		CFD Initialization Functions			      //
//								      //
//__________________________________________________________________//

int read_parameters( const char *szFileName,       /* name of the file 			*/
		     float *Re,                   /* reynolds number   			*/
                     float *PI,                   /* pressure 				*/
                     float *GX,                   /* gravitation x-direction 		*/
                     float *GY,                   /* gravitation y-direction 		*/
                     float *t_end,                /* end time 				*/
                     float *xlength,              /* length of the domain x-dir.		*/
                     float *ylength,              /* length of the domain y-dir.		*/
                     float *dt,                   /* time step 				*/
                     float *dx,                   /* length of a cell x-dir. 		*/
                     float *dy,                   /* length of a cell y-dir. 		*/
                     int    imax,                 /* number of cells x-direction	*/
                     int    jmax,                 /* number of cells y-direction	*/
                     float *alpha,                /* uppwind differencing factor	*/
                     float *omg,                  /* relaxation factor 		*/
                     float *tau,                  /* safety factor for time step	*/
                     int    *itermax,              /* max. number of iterations  	*/
		                                   /* for pressure per time step	*/
                     float *eps,                  /* accuracy bound for pressure	*/
                     float *dt_value,             /* time for output 			*/
                     float *dp		   /* dp/dx gradient of pressure 	*/
		  )
{
   READ_DOUBLE( szFileName, *Re    );
   READ_DOUBLE( szFileName, *t_end );
   READ_DOUBLE( szFileName, *dt    );

   READ_DOUBLE( szFileName, *omg   );
   READ_DOUBLE( szFileName, *eps   );
   READ_DOUBLE( szFileName, *tau   );
   READ_DOUBLE( szFileName, *alpha );

   READ_INT   ( szFileName, *itermax );
   READ_DOUBLE( szFileName, *dt_value );

   READ_DOUBLE( szFileName, *GX );
   READ_DOUBLE( szFileName, *GY );
   READ_DOUBLE( szFileName, *PI );

   READ_DOUBLE( szFileName, *dp );
   
   *xlength = (float)(imax)/10;
   *ylength = (float)(jmax)/10;
   
   *dx = *xlength / (float)(imax);
   *dy = *ylength / (float)(jmax);

   return 1;
}


void init_uv( int imax, int jmax, float **U, float **V, float *imgU, float *imgV, int **Flag, int iter ){
  int i, j;
  for( i = 0; i <= imax; i++ ){
    for( j = 0; j <= jmax; j++ ){
      if( Flag[i][j] == 1 ){
	if( iter == 0 ){
	    U[i][j] = 0.0;
	    V[i][j] = 0.0;
	}
	else{
	   U[i][j] = 3.0*imgU[i+j*imax]; // TODO 3.0?
	   V[i][j] = 3.0*imgV[i+j*imax]; // TODO 3.0?
	}
      }
    }
  }
}


void init_flag( const int imax, const int jmax, int **Flag, int *imgDomain ){
	
  int i, j;
  for( i = 0; i <= imax; i++ ){
    for( j = 0; j <= jmax; j++ ){
      Flag[i][j] = imgDomain[i+j*imax];
    }
  }
}

//__________________________________________________________________//
//								      //
//		CFD Boundary Values Function			      //
//								      //
//__________________________________________________________________//
void boundaryvalues( int    imax,
		     int    jmax,
		     float **U,
		     float **V, 
		     float  *initBU, 
		     float  *initBV, 
		     int    **Flag
		   )
{
    int i, j;
	
    for( i = 0; i <= imax; i++ ){
      for( j = 0; j <= jmax; j++ ){
	if( Flag[i][j] == 2 ){
	// if( imgDomain[ i+j*imax ] == 2 ){
	  U[i][j] = initBU[ i+j*imax ];
	  V[i][j] = initBV[ i+j*imax ];
	}
      }
    }
}

//__________________________________________________________________//
//								      //
//			CFD UVP Functions 			      //
//								      //
//__________________________________________________________________//

#define SQ(a) ((a)*(a))

/* central difference approximation of the second derivative in x */
inline float d2dx(float **m, int i, int j, float dx){
	return (m[i+1][j] - 2*m[i][j] + m[i-1][j]) / (SQ(dx));
}

/* central difference approximation of the second derivative in y */
inline float d2dy(float **m, int i, int j, float dy){
	return (m[i][j+1] - 2*m[i][j] + m[i][j-1]) / (SQ(dy));
}

/* forward difference approximation of the first derivative in x */
inline float ddx(float **m, int i, int j, float dx){
	return (m[i+1][j] - m[i][j]) / dx;
}

/* approximation of the first derivative of the square of u in x */
inline float du2dx(float **m, int i, int j, float dx, float alpha){
	return (
			SQ(m[i][j]+m[i+1][j]) - SQ(m[i-1][j]+m[i][j])
			+ alpha * ( fabs(m[i][j]+m[i+1][j]) * (m[i][j]-m[i+1][j]) -  fabs(m[i-1][j]+m[i][j]) * (m[i-1][j]-m[i][j]) )
	                       )/dx/4.0;
}

/* approximation of the first derivative of the square of v in y */
inline float dv2dy(float **m, int i, int j, float dy, float alpha){
	return (
			SQ(m[i][j]+m[i][j+1]) - SQ(m[i][j-1]+m[i][j])
			+ alpha * ( fabs(m[i][j]+m[i][j+1]) * (m[i][j]-m[i][j+1]) -  fabs(m[i][j-1]+m[i][j]) * (m[i][j-1]-m[i][j]))
	                       )/dy/4.0;
}

inline float duvdx(float ** u, float **v, int i, int j, float dx, float alpha){
	return (
			(u[i][j]+u[i][j+1]) * (v[i][j]+v[i+1][j]) - (u[i-1][j]+u[i-1][j+1]) * (v[i-1][j]+v[i][j])
			+ alpha * ( fabs(u[i][j]+u[i][j+1]) * (v[i][j]-v[i+1][j]) - fabs(u[i-1][j]+u[i-1][j+1]) * (v[i-1][j]-v[i][j]) )
	                       )/dx/4.0;
}

inline float duvdy(float **u, float **v, int i, int j, float dy, float alpha){
	return (
			(u[i][j]+u[i][j+1]) * (v[i][j]+v[i+1][j]) - (u[i][j-1]+u[i][j]) * (v[i][j-1]+v[i+1][j-1])
			+ alpha * ( fabs(v[i][j]+v[i+1][j]) * (u[i][j]-u[i][j+1]) - fabs(v[i][j-1]+v[i+1][j-1]) * (u[i][j-1]-u[i][j]) )
	                       )/dy/4.0;
}

void calculate_fg(
  float Re,
  float GX,
  float GY,
  float alpha,
  float dt,
  float dx,
  float dy,
  int imax,
  int jmax,
  float **U,
  float **V,
  float **F,
  float **G, 
  int **Flag
){
	int i,j;
	
	/* boundary conditions */
	for(j=1; j<=jmax; j++){
		F[ 0 ][ j ] = U[ 0 ][ j ];
		F[ imax ][ j ] = U[ imax ][ j ];
	}

	for(i=1; i<=imax; i++){
		G[i][0] = V[i][0];
		G[i][jmax] = V[i][jmax];
	}
	
	/* inner values */
	for(i=1; i<=imax-1; i++)
		for(j=1; j<=jmax; j++)
			if(Flag[i][j]==C_F && Flag[i+1][j]==C_F)
				F[i][j] = U[i][j] + dt * (
						(d2dx(U,i,j,dx) + d2dy(U,i,j,dy))/Re - du2dx(U, i, j, dx, alpha) - duvdy(U,V,i,j,dy, alpha) + GX
						);

	for(i=1; i<=imax; i++)
		for(j=1; j<=jmax-1; j++)
			if(Flag[i][j]==C_F && Flag[i][j+1]==C_F)
				G[i][j] = V[i][j] + dt * (
						(d2dx(V,i,j,dx) + d2dy(V,i,j,dy))/Re - duvdx(U, V, i, j, dx, alpha) - dv2dy(V,i,j,dy, alpha) + GY
						);

}


void calculate_rs(
  float dt,
  float dx,
  float dy,
  int imax,
  int jmax,
  float **F,
  float **G,
  float **RS,
  int **Flag
){
	int i,j;

	for(i=1; i<=imax; i++)
		for(j=1; j<=jmax; j++)
			if( IS_FLUID(Flag[i][j]) )
				RS[i][j] = 1/dt*((F[i][j]-F[i-1][j])/dx + (G[i][j]-G[i][j-1])/dy) ;

}

void calculate_dt(
  float Re,
  float tau,
  float *dt,
  float dx,
  float dy,
  int imax,
  int jmax,
  float **U,
  float **V
){
	*dt = tau * fmin( fmin( Re/2/(1/(dx*dx) + 1/(dy*dy)), dx/fmatrix_max( U, 0, imax+1, 0, jmax+1 ) ),
			   dy/fmatrix_max(V,0,imax+1,0,jmax+1)
			);
}


void calculate_uv(
  float dt,
  float dx,
  float dy,
  int imax,
  int jmax,
  float **U,
  float **V,
  float **F,
  float **G,
  float **P,
  int **Flag
){
	int i,j;

	for(i=1; i<=imax-1; i++)
		for(j=1; j<=jmax; j++)
			if(Flag[i][j]==C_F && Flag[i+1][j]==C_F)
				U[i][j] = F[i][j] - dt/dx*(P[i+1][j]-P[i][j]);

	for(i=1; i<=imax; i++)
		for(j=1; j<=jmax-1; j++)
			if(Flag[i][j]==C_F && Flag[i][j+1]==C_F)
				V[i][j] = G[i][j] - dt/dy*(P[i][j+1]-P[i][j]);
}

//__________________________________________________________________//
//								      //
//			CFD SOR Functions			      //
//								      //
//__________________________________________________________________//

__global__ void global_sor (float omg,
  float dx,
  float dy,
  int    imax,
  int    jmax,
  const int fluid_cells,
  float *P,
  float *RS,
  int    *Flag,
  float *res,
  float dp, 
 float sor_theta, 
 int redOrBlack)
{

  int i,j;
  float rloc;
  float coeff = omg/(2.0*(1.0/(dx*dx)+1.0/(dy*dy)));

  int ind = threadIdx.x + blockDim.x * blockIdx.x;	

  i = ind / (imax+2);
  j = ind % (imax+2);

  if ((i>0) && (j>0) && (i<imax+1) && (j<jmax+1))
  {
    bool isActive = (((i+j)%2) == redOrBlack);
  /* SOR iteration */
  //for(i = 1; i <= imax; i++) {
      //for(j = 1; j<=jmax; j++) { 
		  /* SOR computation limited to the fluid cells ( C_F - fluid cell )*/
		  if( (isActive) && Flag[ i + j*(imax+2) ] == C_F ){
			P[i + j*(imax+2)] = (1.0-omg)*P[i + j*(imax+2)] + coeff*(( P[i+1 + j*(imax+2)]+P[i-1 + j*(imax+2)])/(dx*dx) + ( P[i + (j+1)*(imax+2)]+P[i + (j-1)*(imax+2)])/(dy*dy) - RS[i + j*(imax+1)]);
		  }
     // }
 // }


  /* compute the residual */
/*  rloc = 0;
  for(i = 1; i <= imax; i++) {
      for(j = 1; j <= jmax; j++) { */
	  /* Residual computation limited to the fluid cells ( C_F - fluid cell ) */
/*	  if( Flag[ i + j*(imax+2)] == C_F ){
	      rloc += ( (P[i+1 + j*(imax+2)]-2.0*P[i + j*(imax+2)]+P[i-1 + j*(imax+2)])/(dx*dx) + ( P[i + (j+1)*(imax+2)]-2.0*P[i + j*(imax+2)]+P[i + (j-1)*(imax+2)])/(dy*dy) - RS[i + j*imax])*
		          ( (P[i+1 + j*(imax+2)]-2.0*P[i + j*(imax+2)]+P[i-1 + j*(imax+2)])/(dx*dx) + ( P[i + (j+1)*(imax+2)]-2.0*P[i + j*(imax+2)]+P[i + (j-1)*(imax+2)])/(dy*dy) - RS[i + j*imax]);
	  
    }
  } */
  /* Residual devided only by the number of fluid cells instead of imax*jmax */
  //rloc = rloc/((float)fluid_cells);
  //rloc = sqrt(rloc);
  /* set residual */
  //*res = rloc;
}
}


void sor(
  float omg,
  float dx,
  float dy,
  int    imax,
  int    jmax,
  const int fluid_cells,
  float **P,
  float **RS,
  int    **Flag,
  float *res,
  float dp
) {
  
  int i,j;
  float rloc;
  float coeff = omg/(2.0*(1.0/(dx*dx)+1.0/(dy*dy)));

  /* SOR iteration */
  for(i = 1; i <= imax; i++) {
      for(j = 1; j<=jmax; j++) {
		  /* SOR computation limited to the fluid cells ( C_F - fluid cell )*/
		  if( Flag[ i ][ j ] == C_F ){
			P[i][j] = (1.0-omg)*P[i][j] + coeff*(( P[i+1][j]+P[i-1][j])/(dx*dx) + ( P[i][j+1]+P[i][j-1])/(dy*dy) - RS[i][j]);
		  }
      }
  }


  /* compute the residual */
  rloc = 0;
  for(i = 1; i <= imax; i++) {
      for(j = 1; j <= jmax; j++) {
	  /* Residual computation limited to the fluid cells ( C_F - fluid cell ) */
	  if( Flag[ i ][ j ] == C_F ){
	      rloc += ( (P[i+1][j]-2.0*P[i][j]+P[i-1][j])/(dx*dx) + ( P[i][j+1]-2.0*P[i][j]+P[i][j-1])/(dy*dy) - RS[i][j])*
		          ( (P[i+1][j]-2.0*P[i][j]+P[i-1][j])/(dx*dx) + ( P[i][j+1]-2.0*P[i][j]+P[i][j-1])/(dy*dy) - RS[i][j]);
	  }
    }
  }
  /* Residual devided only by the number of fluid cells instead of imax*jmax */
  rloc = rloc/((float)fluid_cells);
  rloc = sqrt(rloc);
  /* set residual */
  *res = rloc;
}
