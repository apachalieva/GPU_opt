// ###
// ###
// ### Practical Course: GPU Programming in Computer Vision
// ###
// ###
// ### Technical University Munich, Computer Vision Group
// ### Summer Semester 2014, September 8 - October 10
// ###
// ###
// ### Maria Klodt, Jan Stuehmer, Mohamed Souiai, Thomas Moellenhoff
// ###
// ###

// ###
// ###
// ### TODO: For every student of your group, please provide here:
// ###
// ### name, email, login username (for example p123)
// ###
// ###


// C Libraries

//extern "C"{
    #include "CFD/cfd.h"
//}  

#include "aux.h"
#include "functionalities.h"
#include <iostream>
#include <math.h>
using namespace std;

// uncomment to use the camera
//#define CAMERA


int main(int argc, char **argv)
{
    // Before the GPU can process your kernels, a so called "CUDA context" must be initialized
    // This happens on the very first call to a CUDA function, and takes some time (around half a second)
    // We will do it right here, so that the run time measurements are accurate
    cudaDeviceSynchronize();  CUDA_CHECK;




    // Reading command line parameters:
    // getParam("param", var, argc, argv) looks whether "-param xyz" is specified, and if so stores the value "xyz" in "var"
    // If "-param" is not specified, the value of "var" remains unchanged
    //
    // return value: getParam("param", ...) returns true if "-param" is specified, and false otherwise

#ifdef CAMERA
    bool ret;
#else
    // input image
    string image = "";
    bool ret = getParam("i", image, argc, argv);
    if (!ret) cerr << "ERROR: no image specified" << endl;
    if (argc <= 1) { cout << "Usage: " << argv[0] << " -i <image> <gamma> [-repeats <repeats>] [-gray]" << endl; return 1; }
#endif
    
    // number of computation repetitions to get a better run time measurement
    int repeats = 1;
    getParam("repeats", repeats, argc, argv);
    cout << "repeats: " << repeats << endl;
    
    // load the input image as grayscale if "-gray" is specifed
    bool gray = false;
    getParam("gray", gray, argc, argv);
    cout << "gray: " << gray << endl;

    // ### Define your own parameters here as needed  

    // input image
    string mask = "";
    ret = getParam("mask", mask, argc, argv);
    if (!ret) cerr << "ERROR: no image specified" << endl;
    if (argc <= 1) { cout << "Usage: " << argv[0] << " -mask <mask>" << endl; return 1; }  

    int poisson = 1000;
    getParam("poisson", poisson, argc, argv);
    cout << "poisson: " << poisson << endl;

    int iter = 10;
    getParam("iter", iter, argc, argv);
    cout << "iter: " << iter << endl;

    // Run option: 0 - Laplace Equation, 1 - NS
    int option = 1;
    getParam("option", option, argc, argv);
    cout << "option: " << option << endl;

    // Timestep for anisotropic diffusion
    float tau = 0.02;
    getParam("tau", tau, argc, argv);
    cout << "tau: " << tau << endl;

    // Timestep for anisotropic diffusion
    int aniso_iter = 5;
    getParam("aniso_iter", aniso_iter, argc, argv);
    cout << "aniso_iter: " << tau << endl;

    // Full image: 1 - Image without superimposed inpainting domain, 0 - image with ...
    int  FullImage = 1;
    getParam("FullImage", FullImage, argc, argv);
    cout << "FullImage: " << FullImage << endl;




    // Init camera / Load input image
#ifdef CAMERA

    // Init camera
  	cv::VideoCapture camera(0);
  	if(!camera.isOpened()) { cerr << "ERROR: Could not open camera" << endl; return 1; }
    int camW = 640;
    int camH = 480;
  	camera.set(CV_CAP_PROP_FRAME_WIDTH,camW);
  	camera.set(CV_CAP_PROP_FRAME_HEIGHT,camH);
    // read in first frame to get the dimensions
    cv::Mat mIn;
    camera >> mIn;
    
#else
    
    // Load the input image using opencv (load as grayscale if "gray==true", otherwise as is (may be color or grayscale))
    cv::Mat mIn = cv::imread(image.c_str(), (gray? CV_LOAD_IMAGE_GRAYSCALE : -1));
    // check
    if (mIn.data == NULL) { cerr << "ERROR: Could not load image " << image << endl; return 1; }
    
#endif

    // convert to float representation (opencv loads image values as single bytes by default)
    mIn.convertTo(mIn,CV_32F);
    // convert range of each channel to [0,1] (opencv default is [0,255])
    mIn /= 255.f;
    // get image dimensions
    int w = mIn.cols;         // width
    int h = mIn.rows;         // height
    int nc = mIn.channels();  // number of channels
    cout << "image: " << w << " x " << h << endl;


    // Load the mask
    cv::Mat mMask = cv::imread(mask.c_str(), (gray? CV_LOAD_IMAGE_GRAYSCALE : -1));
    // check
    if (mMask.data == NULL) { cerr << "ERROR: Could not load image " << mask << endl; return 1; }

    // convert to float representation (opencv loads image values as single bytes by default)
    mMask.convertTo(mMask,CV_32F);
    // convert range of each channel to [0,1] (opencv default is [0,255])
    mMask /= 255.f;


    // Set the output image format
    // ###
    // ###
    // ### TODO: Change the output image format as needed
    // ###
    // ###
    cv::Mat mOut(h,w,mIn.type());  // mOut will have the same number of channels as the input image, nc layers
    //cv::Mat mOut(h,w,CV_32FC3);    // mOut will be a color image, 3 layers
    //cv::Mat mOut(h,w,CV_32FC1);    // mOut will be a grayscale image, 1 layer
    // ### Define your own output images here as needed




    // Allocate arrays
    // input/output image width: w
    // input/output image height: h
    // input image number of channels: nc
    // output image number of channels: mOut.channels(), as defined above (nc, 3, or 1)

    // allocate raw input image array
    float *imgIn  = new float[(size_t)w*h*nc];

    // allocate mask image array
    float *imgMask  = new float[(size_t)w*h];

    // allocate raw output array (the computation result will be stored in this array, then later converted to mOut for displaying)
	float *imgOut = new float[(size_t)w*h*nc];
	float *v1 = new float[(size_t)w*h*nc];
	float *v2 = new float[(size_t)w*h*nc];
	float *imgVorticity = new float[(size_t)w*h*nc];
	float *initVorticity = new float[(size_t)w*h*nc];
	int *imgDomain = new int[(size_t)w*h];
	float *imgU = new float[(size_t)w*h*nc];
	float *imgV = new float[(size_t)w*h*nc];
	
	float *initU = new float[(size_t)w*h*nc];
	float *initV = new float[(size_t)w*h*nc];

	// TODO: Introduced temporarily to display the domain
	float *floatDomain = new float[(size_t)w*h];

    // For camera mode: Make a loop to read in camera frames
#ifdef CAMERA
    // Read a camera image frame every 30 milliseconds:
    // cv::waitKey(30) waits 30 milliseconds for a keyboard input,
    // returns a value <0 if no key is pressed during this time, returns immediately with a value >=0 if a key is pressed
    while (cv::waitKey(30) < 0)
    {
    // Get camera image
    camera >> mIn;//convert_layered_to_mat_int
    // convert to float representation (opencv loads image values as single bytes by default)
    mIn.convertTo(mIn,CV_32F);
    // convert range of each channel to [0,1] (opencv default is [0,255])
    mIn /= 255.f;
#endif

    // Init raw input image array
    // opencv images are interleaved: rgb rgb rgb...  (actually bgr bgr bgr...)
    // But for CUDA it's better to work with layered images: rrr... ggg... bbb...
    // So we will convert as necessary, using interleaved "cv::Mat" for loading/saving/displaying, and layered "float*" for CUDA computations
    convert_mat_to_layered (imgIn, mIn);

    // Creating the mask
    convert_mat_to_layered (imgMask, mMask);


    Timer timer; timer.start();
    // ###
    // ###
    // ### TODO: Main computation
    // ###
    // ###

    int n = w*h*nc; 

//===============================================================================================//
// TEST RGB transformation - It returns a 3-channel picture.
// In the algorithm we are working with 1-channel picture. 
// We have to execute the algorithm three times and to combine the pictures at the end.
    
//     forward_color_transf( imgIn, imgOut, w, h, nc );  
//     inverse_color_transf( imgOut, imgIn, w, h, nc );
    
//===============================================================================================//

forward_color_transf( imgIn, imgIn, w, h, nc ); 

// Trying to do everything on GPU

float *gpu_initVorticity, *gpu_Vorticity, *gpu_Mask, *gpu_v1, *gpu_v2;//, *gpu_U, *gpu_V;;
int *gpu_Domain;

// Allocate GPU memory
cudaMalloc(&gpu_initVorticity, n*sizeof(float)); CUDA_CHECK;
cudaMalloc(&gpu_Vorticity, n*sizeof(float)); CUDA_CHECK;
cudaMalloc(&gpu_Mask, w*h*sizeof(float));CUDA_CHECK;
cudaMalloc(&gpu_v1, n*sizeof(float));CUDA_CHECK;
cudaMalloc(&gpu_v2, n*sizeof(float));CUDA_CHECK;
//cudaMalloc(&gpu_U, n*sizeof(float)); CUDA_CHECK;
//cudaMalloc(&gpu_V, n*sizeof(float)); CUDA_CHECK;

  dim3 block = dim3(128,1,1);
  
  dim3 grid = dim3((n + block.x - 1) / block.x, 1, 1);

if (option == 0)
{
  for (int ind=0; ind<n; ind++)
    {
	    imgVorticity[ind] = 0.0f;
	    initVorticity[ind] = 0.0f;
    }
    cudaMemcpy(gpu_initVorticity, initVorticity, n*sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
    cudaMemcpy(gpu_Vorticity, imgVorticity, n*sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
}
/*else if (option == 1)
{
	  
		// Calculate init vorticity	
		// allocate GPU memory
		//cudaMalloc(&gpu_U, n*sizeof(float)); CUDA_CHECK;
		//cudaMalloc(&gpu_V, n*sizeof(float)); CUDA_CHECK;
		//cudaMalloc(&gpu_Vorticity, n*sizeof(float)); CUDA_CHECK;
		//cudaMalloc(&gpu_initVorticity, n*sizeof(float)); CUDA_CHECK;
		//cudaMalloc(&gpu_Domain, w*h*sizeof(int)); CUDA_CHECK;

		// copy host memory to device
		//cudaMemcpy(gpu_U, imgU, n*sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
		//cudaMemcpy(gpu_V, imgV, n*sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
		//cudaMemcpy(gpu_Domain, imgDomain, w*h*sizeof(int), cudaMemcpyHostToDevice); CUDA_CHECK;

		// launch kernel
		global_vorticity <<<grid,block>>> ( gpu_U, gpu_V, gpu_initVorticity, gpu_Domain, w, h, nc, n, FullImage);

		// copy result back to host (CPU) memory
		//cudaMemcpy(initVorticity, gpu_Vorticity, n * sizeof(float), cudaMemcpyDeviceToHost ); CUDA_CHECK;
		//cudaMemcpy(initVorticity, gpu_initVorticity, n * sizeof(float), cudaMemcpyDeviceToHost ); CUDA_CHECK;
		cudaMemcpy(initU, gpu_U, n * sizeof(float), cudaMemcpyDeviceToHost ); CUDA_CHECK;
		cudaMemcpy(initV, gpu_V, n * sizeof(float), cudaMemcpyDeviceToHost ); CUDA_CHECK;

		// free device (GPU) memory
		//cudaFree(gpu_U); CUDA_CHECK;
		//cudaFree(gpu_V); CUDA_CHECK;
		//cudaFree(gpu_Vorticity); CUDA_CHECK;
		//cudaFree(gpu_initVorticity); CUDA_CHECK;
		//cudaFree(gpu_Domain); CUDA_CHECK;


}*/

	// Calculate the inpainting domain	
	// allocate GPU memory
	//cudaMalloc(&gpu_Mask, w*h*sizeof(float));CUDA_CHECK;
	cudaMalloc(&gpu_Domain, w*h*sizeof(int));CUDA_CHECK;

	// copy host memory to device
	cudaMemcpy(gpu_Mask, imgMask, w*h*sizeof(float), cudaMemcpyHostToDevice);CUDA_CHECK;
	cudaMemcpy(gpu_Domain, imgDomain, w*h*sizeof(int), cudaMemcpyHostToDevice);CUDA_CHECK;

	// launch kernel
	global_detect_domain1 <<<grid,block>>> (gpu_Mask, gpu_Domain, w, h, w*h);
	global_detect_domain2 <<<grid,block>>> (gpu_Mask, gpu_Domain, w, h, w*h);
	global_detect_domain3 <<<grid,block>>> (gpu_Mask, gpu_Domain, w, h, w*h);

	// copy result back to host (CPU) memory
	cudaMemcpy(imgDomain, gpu_Domain, w*h * sizeof(int), cudaMemcpyDeviceToHost );CUDA_CHECK;

	// free device (GPU) memory
	cudaFree(gpu_Mask);CUDA_CHECK;
	//cudaFree(gpu_Domain);CUDA_CHECK;
	


	

// Main loop

for (int j=0; j<iter; j++)
{
	cout << "Global iteration: " << j << endl;
	// allocate GPU memory
	float *gpu_In, *gpu_Out, *gpu_U, *gpu_V;// ,*gpu_v1, *gpu_v2;//, *gpu_Vorticity;//, *gpu_Mask; //, *gpu_initVorticity;
	//int *gpu_Domain;



if (option == 1)
{

	// Calculate gradient
	cudaMalloc(&gpu_In, n*sizeof(float));CUDA_CHECK;
	//cudaMalloc(&gpu_v1, n*sizeof(float));CUDA_CHECK;
	//cudaMalloc(&gpu_v2, n*sizeof(float));CUDA_CHECK;
	//cudaMalloc(&gpu_Domain, w*h*sizeof(int));CUDA_CHECK;
	
	cudaMalloc(&gpu_U, n*sizeof(float));CUDA_CHECK;
	cudaMalloc(&gpu_V, n*sizeof(float));CUDA_CHECK;


	// copy host memory to device
	cudaMemcpy(gpu_In, imgIn, n*sizeof(float), cudaMemcpyHostToDevice);CUDA_CHECK;
	//cudaMemcpy(gpu_v1, v1, n*sizeof(float), cudaMemcpyHostToDevice);CUDA_CHECK;
	//cudaMemcpy(gpu_v2, v2, n*sizeof(float), cudaMemcpyHostToDevice);CUDA_CHECK;
	//cudaMemcpy(gpu_Domain, imgDomain, w*h*sizeof(int), cudaMemcpyHostToDevice);CUDA_CHECK;

	// launch kernel
	global_grad <<<grid,block>>> (gpu_In, gpu_Domain, gpu_v1, gpu_v2, w, h, nc, n, FullImage);

	// copy result back to host (CPU) memory
	//cudaMemcpy( v1, gpu_v1, n * sizeof(float), cudaMemcpyDeviceToHost ); CUDA_CHECK;
	//cudaMemcpy( v2, gpu_v2, n * sizeof(float), cudaMemcpyDeviceToHost ); CUDA_CHECK;
	// Invert the V values according t: V = -dI/dx
	global_reverse_sign <<<grid,block>>> (gpu_v2, n);
	cudaMemcpy( imgU, gpu_v2, n * sizeof(float), cudaMemcpyDeviceToHost ); CUDA_CHECK;
	cudaMemcpy( imgV, gpu_v1, n * sizeof(float), cudaMemcpyDeviceToHost ); CUDA_CHECK;

	// free device (GPU) memory
	cudaFree(gpu_In); CUDA_CHECK;
	//cudaFree(gpu_v1); CUDA_CHECK;
	//cudaFree(gpu_v2); CUDA_CHECK;
	cudaFree(gpu_U); CUDA_CHECK;
	cudaFree(gpu_V); CUDA_CHECK;
	//cudaFree(gpu_Domain); CUDA_CHECK;

// // 	Invert the V values according t: V = -dI/dx
// // 	TODO: Invert in parallel 
// 	for (int i=0; i<n; i++)
// 	{
// 		//imgV[i] = -imgV[i];
// 		imgU[i] = -imgU[i];
// 	}

}	


if (option == 1)
{
    if ( j == 0 )
	{	
	  
		// Calculate init vorticity	
		// allocate GPU memory
		cudaMalloc(&gpu_U, n*sizeof(float)); CUDA_CHECK;
		cudaMalloc(&gpu_V, n*sizeof(float)); CUDA_CHECK;
		//cudaMalloc(&gpu_Vorticity, n*sizeof(float)); CUDA_CHECK;
		//cudaMalloc(&gpu_initVorticity, n*sizeof(float)); CUDA_CHECK;
		//cudaMalloc(&gpu_Domain, w*h*sizeof(int)); CUDA_CHECK;

		// copy host memory to device
		cudaMemcpy(gpu_U, imgU, n*sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
		cudaMemcpy(gpu_V, imgV, n*sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
		//cudaMemcpy(gpu_Domain, imgDomain, w*h*sizeof(int), cudaMemcpyHostToDevice); CUDA_CHECK;

		// launch kernel
		global_vorticity <<<grid,block>>> ( gpu_U, gpu_V, gpu_initVorticity, gpu_Domain, w, h, nc, n, FullImage);

		// copy result back to host (CPU) memory
		//cudaMemcpy(initVorticity, gpu_Vorticity, n * sizeof(float), cudaMemcpyDeviceToHost ); CUDA_CHECK;
		//cudaMemcpy(initVorticity, gpu_initVorticity, n * sizeof(float), cudaMemcpyDeviceToHost ); CUDA_CHECK;
		cudaMemcpy(initU, gpu_U, n * sizeof(float), cudaMemcpyDeviceToHost ); CUDA_CHECK;
		cudaMemcpy(initV, gpu_V, n * sizeof(float), cudaMemcpyDeviceToHost ); CUDA_CHECK;

		// free device (GPU) memory
		cudaFree(gpu_U); CUDA_CHECK;
		cudaFree(gpu_V); CUDA_CHECK;
		//cudaFree(gpu_Vorticity); CUDA_CHECK;
		//cudaFree(gpu_initVorticity); CUDA_CHECK;
		//cudaFree(gpu_Domain); CUDA_CHECK;

	} 
	// CFD solver
	for (int channel = 0; channel < nc; channel++)
	{
		cfd( argc, argv, &(imgU[channel*w*h]), &(imgV[channel*w*h]), imgDomain, &(initU[channel*w*h]), &(initV[channel*w*h]), w, h, j , grid, block);
	}

	// Calculate vorticity	
	// allocate GPU memory

	cudaMalloc(&gpu_U, n*sizeof(float)); CUDA_CHECK;
	cudaMalloc(&gpu_V, n*sizeof(float)); CUDA_CHECK;
	//cudaMalloc(&gpu_Vorticity, n*sizeof(float)); CUDA_CHECK;
	//cudaMalloc(&gpu_Domain, w*h*sizeof(int)); CUDA_CHECK;

	// copy host memory to device
	cudaMemcpy(gpu_U, imgU, n*sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
	cudaMemcpy(gpu_V, imgV, n*sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
	//cudaMemcpy(gpu_Domain, imgDomain, w*h*sizeof(int), cudaMemcpyHostToDevice); CUDA_CHECK;

	// launch kernel
	global_vorticity <<<grid,block>>> ( gpu_U, gpu_V, gpu_Vorticity, gpu_Domain, w, h, nc, n, FullImage);

	// copy result back to host (CPU) memory
	//cudaMemcpy(imgVorticity, gpu_Vorticity, n * sizeof(float), cudaMemcpyDeviceToHost ); CUDA_CHECK;
	cudaMemcpy(imgU, gpu_U, n * sizeof(float), cudaMemcpyDeviceToHost ); CUDA_CHECK;
	cudaMemcpy(imgV, gpu_V, n * sizeof(float), cudaMemcpyDeviceToHost ); CUDA_CHECK;

	// free device (GPU) memory
	cudaFree(gpu_U); CUDA_CHECK;
	cudaFree(gpu_V); CUDA_CHECK;
	//cudaFree(gpu_Vorticity); CUDA_CHECK;
	//cudaFree(gpu_Domain); CUDA_CHECK;
}
/*else if (option == 0)
{
  for (int ind=0; ind<n; ind++)
    {
	    imgVorticity[ind] = 0.0f;
	    initVorticity[ind] = 0.0f;
    }
}*/

    if ( j == 0 )
	{	
		for ( int ind = 0; ind < n; ind++ )
		{

		  int x, y, ch;	

		  ch = (int)(ind) / (int)(w*h);
		  y = ( ind - ch*w*h ) / w;
		  x = ( ind - ch*w*h ) % w;
		  int indDomain = x + w*y;

			if ( imgDomain[indDomain] == 1 )
				imgIn[ind] = 1.0;
		}
		
		//Display the image with cut inpainting domain
	       inverse_color_transf( imgIn, imgIn, w, h, nc );
               convert_layered_to_mat(mIn, imgIn);
	       forward_color_transf( imgIn, imgIn, w, h, nc ); 
	}

	// Solve the Poisson equation - update the image

	cudaMalloc(&gpu_Out, n*sizeof(float)); CUDA_CHECK;
	cudaMalloc(&gpu_In, n*sizeof(float)); CUDA_CHECK;
	//cudaMalloc(&gpu_Vorticity, n*sizeof(float)); CUDA_CHECK;
	//cudaMalloc(&gpu_initVorticity, n*sizeof(float)); CUDA_CHECK;
	//cudaMalloc(&gpu_Domain, w*h*sizeof(int)); CUDA_CHECK;


	// copy host memory to device
	cudaMemcpy(gpu_In, imgIn, n*sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
	cudaMemcpy(gpu_Out, imgIn, n*sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
	//cudaMemcpy(gpu_Vorticity, imgVorticity, n*sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
	//cudaMemcpy(gpu_initVorticity, initVorticity, n*sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
	//cudaMemcpy(gpu_Domain, imgDomain, w*h*sizeof(int), cudaMemcpyHostToDevice); CUDA_CHECK;

	// launch kernel
	for (int i=0; i<poisson; i++)
	{	
	  global_solve_Poisson <<<grid,block>>> (gpu_In, gpu_In, gpu_initVorticity, gpu_Vorticity, gpu_Domain, w, h, nc, n, 0.7, 1);
	  global_solve_Poisson <<<grid,block>>> (gpu_In, gpu_In, gpu_initVorticity, gpu_Vorticity, gpu_Domain, w, h, nc, n, 0.7, 0);
	}
	//global_solve_Poisson <<<grid,block>>> (gpu_Out, gpu_In, gpu_Vorticity, gpu_Domain, w, h, nc, n, 0.7, 1);

	// copy result back to host (CPU) memory
	cudaMemcpy(imgOut, gpu_In, n * sizeof(float), cudaMemcpyDeviceToHost ); CUDA_CHECK;
	cudaMemcpy(imgIn, gpu_In, n * sizeof(float), cudaMemcpyDeviceToHost ); CUDA_CHECK;

	// free device (GPU) memory
	cudaFree(gpu_Out); CUDA_CHECK;
	cudaFree(gpu_In); CUDA_CHECK;
	//cudaFree(gpu_initVorticity); CUDA_CHECK;
	//cudaFree(gpu_Vorticity); CUDA_CHECK;
	//cudaFree(gpu_Domain); CUDA_CHECK;

	//if ( j+1 % 10 == 0 ) aniso_diff(imgIn, imgDomain, imgIn, w, h, nc, tau, aniso_iter, grid, block);
	

}

	// Trying to make it run just on GPU
	cudaFree(gpu_initVorticity); CUDA_CHECK;
	cudaFree(gpu_Vorticity); CUDA_CHECK;
	cudaFree(gpu_Domain);CUDA_CHECK;
	cudaFree(gpu_v1); CUDA_CHECK;
	cudaFree(gpu_v2); CUDA_CHECK;
	//cudaFree(gpu_U); CUDA_CHECK;
	//cudaFree(gpu_V); CUDA_CHECK;


inverse_color_transf( imgIn, imgOut, w, h, nc );

    timer.end();  float t = timer.get();  // elapsed time in seconds
    cout << "time: " << t*1000 << " ms" << endl;

    // show input image
    //mIn /= 255.f;
     showImage("Input", mIn, 100, 100);  // show at position (x_from_left=100,y_from_above=100)
// 
//     // TODO: Just for diagnostic purposes
 	for (int i=0; i<w*h; i++)
 	{
 		floatDomain[i] = imgDomain[i]*0.5;
 	}
 
     convert_layered_to_mat(mOut, floatDomain);
     showImage("imgDomain", mOut, 100+w+80, 100);
    
    // show output image: first convert to interleaved opencv format from the layered raw array
    convert_layered_to_mat(mOut, imgVorticity);
    mOut *=1000;
    showImage("Heating", mOut, 100+w+80, 100);

    convert_layered_to_mat(mOut, imgVorticity);
    mOut *=-1000;
    showImage("Cooling ", mOut, 100+w+80, 100);

    convert_layered_to_mat(mOut, v1);
    mOut *=1000;
    showImage("v1 ", mOut, 100+w+80, 100);

    convert_layered_to_mat(mOut, v2);
    mOut *=1000;
    showImage("v2 ", mOut, 100+w+80, 100);

    // show output image: first convert to interleaved opencv format from the layered raw array
    convert_layered_to_mat(mOut, imgOut);
    //mOut /= 255.f;
    showImage("Output", mOut, 100+w+40, 100);

    // ### Display your own output images here as needed
#ifdef CAMERA
    // end of camera loop
    }
#else
    // wait for key inputs
    cv::waitKey(0);
#endif

    // save input and result
    cv::imwrite("image_input.png",mIn*255.f);  // "imwrite" assumes channel range [0,255]
    cv::imwrite("image_result.png",mOut*255.f);

    // free allocated arrays
    delete[] imgIn;
    delete[] imgMask;
    delete[] imgVorticity;
    delete[] initVorticity;
    delete[] initU;
    delete[] initV;
    delete[] imgDomain;
    delete[] floatDomain;
    delete[] v1;
    delete[] v2;

    // close all opencv windows
    cvDestroyAllWindows();
    return 0;
}
