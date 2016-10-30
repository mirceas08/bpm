#########################################
## Make file for boundary point method ##
#########################################


##### Insert here location of headers and libraries

# headers
ARMADILLO_INC	= 
EIGEN_INC		= 

# libraries
OPENBLAS_LIB 	= /usr/lib/openblas
LAPACK_LIB		= /usr/lib/lapack

#################################################################

CC				= g++
MPICC			= mpic++
DEBUG			= -DARMA_NO_DEBUG -DNDEBUG
INCLUDES        = -I $(ARMADILLO_INC) -I $(EIGEN_INC)
LIBS            = -L $(OPENBLAS_LIB) -lopenblas -L $(LAPACK_LIB) -llapack
CFLAGS        	= -O3 -std=c++11 -fopenmp -march=native $(DEBUG)
PROG			= bpm_mpi bpm_serial

bpm_mpi: bpm_mpi.cpp
	$(MPICC) -o bpm_mpi bpm_mpi.cpp $(INCLUDES) $(LIBS) $(CFLAGS)
	
bpm_serial: bpm_serial.cpp
	$(CC) -o bpm_serial bpm_serial.cpp $(INCLUDES) $(LIBS) $(CFLAGS)

clean:
	rm -f $(PROG) *.o core
