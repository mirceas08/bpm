Description
===========
This is the repository for the code written during the PRACE Summer of High Performance Computing 2015 program. Summer of HPC (http://summerofhpc.prace-ri.eu/) offers summer placements at computing centres accross Europe to undergraduate and graduate students.

The code in the repository is a C++ parallel version of the boundary point method applied to a block diagonal structure of a semidefinite program. More information can be found in the docs and at https://summerofhpc.prace-ri.eu/parallel-boundary-point-method/

Code dependencies
===========
* C++ and MPI compiler with C++11 support
* Armadillo: high quality C++ linear algebra library used throughout the whole code. It integrates with LAPACK and BLAS. Use Armadillo without installation and link against BLAS and LAPACK instead
* Eigen: C++ template linear algebra library used for solving the sparse linear system. It has no dependencies, as it is defined in the headers.
* OpenBLAS: multi-threaded replacement of traditional BLAS. Recommended for signicantly higher performance. Otherwise link with traditional BLAS and LAPACK
* MATLAB/Octabe for generating the data

Executables
===========
* bpm_mpi: parallel version - MPI is used as message passing
* bpm_sequential: serial version

To compile the code type

>make

Run the code with

>./executable parameterFile

The code reads input data from 'data' folder and writes solution matrix into 'solution' folder.

Acknowledgments
===========
* PRACE for allowing me to be part of the program and enrich my background
* My supervisor Professor Janez Povh and the SoHPC coordinator Professor Leon Kos for all the guidance and help throughout the project
* All the people at LECAD Lab at the Faculty of Mechanical Engineering of University of Ljubljana.

References
===========
[1] J. Povh, F. Rendl, A. Wiegele. A boundary point method to solve semidefinite programs, Computing, 78(3):277-286, November 2006.

[2] C. Sanderson Armadillo: an open source C++ linear algebra library for fast prototyping and computationally intensive experiments Technical Report, NICTA, 2010.

[3] Gael Guennebaud and Benoit Jacob and others. Eigen v3, Retrieved from http://eigen.tuxfamily.org, 2010

License
===========
Code available under the GNU General Public License. PRACE sponsoring acknowledgment cannot be removed.
