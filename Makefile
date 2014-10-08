main: main.cu aux.cu aux.h functionalities.cu functionalities.h CFD/cfd.cu CFD/cfd.h CFD/helper.cu CFD/helper.h CFD/visual.cu CFD/visual.h CFD/functions.cu CFD/functions.h Makefile
	nvcc -o main main.cu aux.cu functionalities.cu CFD/cfd.cu CFD/helper.cu CFD/visual.cu CFD/functions.cu --ptxas-options=-v --use_fast_math --compiler-options -Wall -lopencv_highgui -lopencv_core

