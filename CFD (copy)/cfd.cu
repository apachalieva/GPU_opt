#include "cfd.h"
#include "functions.h"
#include "helper.h"
#include "visual.h"

/**
 * The main operation reads the configuration file, initializes the scenario and
 * contains the main loop. So here are the individual steps of the algorithm:
 *
 * - read the program configuration file using read_parameters()
 * - set up the matrices (arrays) needed using the matrix() command
 * - create the initial setup init_uvp(), init_flag(), output_uvp()
 * - perform the main loop
 * - trailer: destroy memory allocated and do some statistics
 *
 * The layout of the grid is decribed by the first figure below, the enumeration
 * of the whole grid is given by the second figure. All the unknowns corresond
 * to a two dimensional degree of freedom layout, so they are not stored in
 * arrays, but in a matrix.
 *
 * @image html grid.jpg
 *
 * @image html whole-grid.jpg
 *
 * Within the main loop the following big steps are done (for some of the 
 * operations a definition is defined already within uvp.h):
 *
 * - calculate_dt() Determine the maximal time step size.
 * - boundaryvalues() Set the boundary values for the next time step.
 * - calculate_fg() Determine the values of F and G (diffusion and confection).
 *   This is the right hand side of the pressure equation and used later on for
 *   the time step transition.
 * - calculate_rs()
 * - Iterate the pressure poisson equation until the residual becomes smaller
 *   than eps or the maximal number of iterations is performed. Within the
 *   iteration loop the operation sor() is used.
 * - calculate_uv() Calculate the velocity at the next time step.
 */


int cfd( int argc, char** args, float *imgU, float *imgV, int *imgDomain, float *initBU, float *initBV, int imax, int jmax, int iter, dim3 grid, dim3 block ){
	double Re, PI, GX, GY, t_end, xlength, ylength, dt, dx, dy, alpha, omg, tau, eps, dt_value, t, res, dp;
	double **U, **V, **P, **F, **G, **RS;
	int n, step, it, itermax;
	int fluid_cells;		/* Number of fluid cells in our geometry */
	char *fname;

	int **Flag;			/* Flagflield matrix */

	if( argc >= 2 )
		fname = args[1];
	else
		fname = PARAMF;

	read_parameters(fname, &Re, &PI, &GX, &GY, &t_end, &xlength, &ylength, &dt, &dx, &dy, imax, jmax, &alpha, &omg, &tau, &itermax, &eps, &dt_value, &dp );
	
	fluid_cells = imax*jmax;

	/* Allocate Flag matrix */
	Flag = imatrix( 0, imax+1, 0, jmax+1 );

	/* should we change the dimension of the matrices in order to save space? */
	U = matrix ( 0 , imax+1 , 0 , jmax+1 );
	V = matrix ( 0 , imax+1 , 0 , jmax+1 );
	
	P = matrix ( 0 , imax+1 , 0 , jmax+1 );

	F = matrix ( 0 , imax , 0 , jmax );
	G = matrix ( 0 , imax , 0 , jmax );
	RS = matrix ( 0 , imax , 0 , jmax );
	
	init_flag( imax, jmax, Flag, imgDomain );
	init_uv( imax, jmax, U, V, imgU, imgV, Flag, iter );
	
	t = .0;
	n = 0;
	step = 0;


	//while( t <= t_end ){
		if( tau > 0 ) calculate_dt( Re, tau, &dt, dx, dy, imax, jmax, U, V );
		//dt = 0.001;
		
		boundaryvalues( imax, jmax, U, V, initBU, initBV, Flag );
		
		/* calculate new values for F and G */
		calculate_fg( Re, GX, GY, alpha, dt, dx, dy, imax, jmax, U, V, F, G, Flag );
		/* calculate right hand side */
		calculate_rs( dt, dx, dy, imax, jmax, F, G, RS, Flag );

		it = 0;
		res = 10000.0;
/*
	float *d_P, *d_RS, *d_Flag;
	cudaMalloc(&d_P, (imax+2)*(jmax+2)*sizeof(float));CUDA_CHECK;
	cudaMalloc(&d_RS, (imax+1)*(jmax+1)*sizeof(float));CUDA_CHECK;
	cudaMalloc(&d_Flag, (imax+2)*(jmax+2)*sizeof(int));CUDA_CHECK;
*/

		while( it < itermax && fabs(res) > eps ){
			sor( omg, dx, dy, imax, jmax, fluid_cells, P, RS, Flag, &res, dp );
			// GPU implementation
			//global_sor <<<grid,block>>> (omg, dx, dy, imax, jmax, fluid_cells, d_P, d_RS, d_Flag, &res, dp, 0.7, 1);
	  		//global_sor <<<grid,block>>> (omg, dx, dy, imax, jmax, fluid_cells, d_P, d_RS, d_Flag, &res, dp, 0.7, 0);

			it++;
		}

		printf("[%d: %f] dt: %f, sor iterations: %d \n", n, t, dt, it);

		if( it == itermax )
		    printf( "    WARNING: Maximum number of iterations reached.\n" );

		calculate_uv( dt, dx, dy, imax, jmax, U, V, F, G, P, Flag );

		t += dt;
		n++;

		if(iter % 50 == 0){
			/* output vtk file for visualization */
			write_vtkFile( VISUAF, iter, xlength, ylength, imax, jmax, dx, dy, U, V, P );
			step++;
		}	
	//}

	// TODO copy U and V to imgU and imgV
	int i, j;

	//boundU = matrix ( 0 , imax+1 , 0 , jmax+1 );
	//boundV = matrix ( 0 , imax+1 , 0 , jmax+1 );
	for( i = 0; i <= imax; i++ ){
	  for( j = 0; j <= jmax; j++){
	    imgU[ i+j*imax ] = U[i][j];
	    imgV[ i+j*imax ] = V[i][j];
	  }
	}
	printf( "imax = %d, jmax = %d\n", imax, jmax );
	printf( "Number of fluid cells = %d\n", fluid_cells );
	printf( "Reynolds number: %f\n", Re );

	/* free memory */
	free_matrix( U, 0, imax+1, 0, jmax+1 );
	free_matrix( V, 0, imax+1, 0, jmax+1 );
	
	free_matrix( P, 0, imax+1, 0, jmax+1 );

	free_matrix( F, 0, imax, 0, jmax );
	free_matrix( G, 0, imax, 0, jmax );
	free_matrix( RS, 0, imax, 0, jmax );

	free_imatrix( Flag, 0, imax+1, 0, jmax+1 );

	return 0;
}
