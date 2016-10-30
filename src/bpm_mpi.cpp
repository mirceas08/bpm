/***************************************************************************************************
* Boundary point method, based on augmented Lagrangian
* C++ MPI parallel version
* Prepared for block diagonal structure
* Solves: max <C,X> s.t. A(X) = b, X positive semidefinite
*
* data:
* m             number of constraints
* dim           vector with dimensions of blocks on the diagonal
* A_GT(i)       n_i^2 by m sparse matrices ( with n_i = dim(i) )
* b             m by 1 vector
* C(i)          coefficient matrices
* AAT           m by m matrix
*
* For details see:
* J. Povh, F. Rendl and A. Wiegele. A boundary point method to solve semidefinite programs.
* Computing, 78(3):277-286, November 2006
*
* Code written by Mircea Simionica (mircea.simionica@gmail.com)
* Check docs for details
*
* Code published under the GNU General Public License
*
*********************** The following acknowledgment cannot be removed *****************************
* I acknowledge PRACE (Partnership for Advanced Computing in Europe) and the
* SoHPC team for allowing me to be part of the program and enrich my background.
****************************************************************************************************
*/

#ifndef BPM_CPP

#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <vector>
#include <algorithm>
#include <string>
#include <set>
#include <mpi.h>
#include <iomanip>
#include "helpers.h"
using namespace std;

#include <armadillo>
using namespace arma;

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
using namespace Eigen;

int main(int argc, char **argv)
{
    /* ---- MPI ---- */
    int rank;
    int size;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    bool mpiroot = (rank == 0);

    int i, k;
    int cnt = 0;
    int cnt_global = 0;
    double secs;

    /* ######################################################## */
    /* ###################### Parameters ###################### */
    /* ######################################################## */

    /* ------------------------- Data ------------------------- */

    // dimensions of the blocks
    vec dim;
    dim.load("data/dim.dat", raw_ascii);
    int p = dim.n_elem; // number of diagonal blocks
    int dimension = dim(rank); // local dimension

    // b vector
    vec b;
    b.load("data/b.dat", raw_ascii);
    int m = b.n_rows; // number of constraints

    // vector of non-zero values in AAT, A_GT's and C's
    vec nnz;
    nnz.load("data/nonzeros.dat", raw_ascii);

    // AAT matrix defined as Eigen object
    Eigen::SparseMatrix<double> AAT_eigen(m, m);
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double> > cholesky;
    if (mpiroot) {
        read_sparse_eigen(AAT_eigen, nnz(0), "data/AAT.dat");
        cholesky.compute(AAT_eigen);
    }

    // A_GT local matrix
    string A_GT_file = "data/A_GT" + to_string(rank+1) + ".dat";
    sp_mat A_GT;
    A_GT = read_sparse_arma(dimension * dimension, m, nnz(rank+1), A_GT_file);

    // C local matrix
    string C_file = "data/C" + to_string(rank+1) + ".dat";
    sp_mat C;
    C = read_sparse_arma(dimension, dimension, nnz(rank+1+p), C_file);

    /* ------------------------- Timer ------------------------- */
    // MPI timer
    double start_time = MPI_Wtime();
    double end_time;

    /* ------------------------ Input check ------------------------ */
    int diagonal = 0;
    if (A_GT.n_rows != dimension * dimension) {
        cout << "Dimensions of A_G are not consistent with dim" << endl;
        return 1;
    }
    if (C.n_rows != dimension) {
        cout << "Dimensions of A_G are not consistent with dim" << endl;
        return 1;
    }

    if (mpiroot) {
        if (AAT_eigen.rows() != m)
        {
            cout << "Dimensions of AAT are not consistent with dim" << endl;
            return 1;
        }
        if (AAT_eigen.cols() == 1)
            diagonal = 1;
        else if (AAT_eigen.cols() != m) {
            cout << "AAT has to be m by 1 or m by m" << endl;
            return 1;
        }
    }

    /* ------------------------ Set starting matrices ------------------------ */
    // X and Z
    mat X = zeros<mat>(dimension, dimension);
    mat Z = zeros<mat>(dimension, dimension);

    /* ------------------------ Set starting values ------------------------ */
    // default values
    double sigma = 0.1;         	// penalty parameter in the augmented Lagrangian
    double tol = 1e-5;          	// tolerance
    double max_iter = 100;      	// maximum outer iterations
    double in_max_init = 5;     	// starting inner iterations
    double in_max_inc = 0.5;    	// increase rate of inner iterations at each outer iteration
    double in_max_max = 30;     	// maximum number of inner iterations
    double print_iter = 10;     	// output interval

    // modify this file for different parameters
    // file read at runtime, no need to recompile
    string parameters_file = "parameters/parameters.dat";

    ifstream fIN(parameters_file.c_str());
    string line;
    getline(fIN, line); // skip header

    while (getline(fIN, line)) {
        stringstream stream(line);
        string variable;
        double value;

        stream >> variable >> value;

        if (variable == "sigma")
            sigma = value;
        else if (variable == "tol")
            tol = value;
        else if (variable == "max_iter")
            max_iter = value;
        else if (variable == "in_max_init")
            in_max_init= value;
        else if (variable == "in_max_inc")
            in_max_inc = value;
        else if (variable == "in_max_max")
            in_max_max = value;
        else if (variable == "print_iter")
            print_iter = value;
        else
            break;
    }

    /* ------------------- Auxiliary data ---------------------------*/
    double normC;
    double normC_local;

    normC_local = norm(C, "fro");
    MPI_Reduce(&normC_local, &normC, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Bcast(&normC, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double normb = norm(b);
    int ev_dec = 0;
    int ev_dec_global = 0;

    double in_max = in_max_init;
    double iter = 1;

    vec res_p(m);
    vec resp_local;

    resp_local = A_GT.t() * vectorise(X);
    MPI_Reduce(resp_local.begin(), res_p.begin(), m, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (mpiroot) {
        res_p -= b;
    }

    vec y = zeros<vec>(m);
    /* ------------------- Some more matrices -----------------------------*/
    // these are all local matrices
    mat Wp = zeros<mat>(dim(rank), dim(rank));
    mat V = zeros<mat>(dim(rank), dim(rank));
    mat ev = zeros<mat>(dim(rank), dim(rank));
    mat Aty = zeros<mat>(dim(rank), dim(rank));
    mat W = zeros<mat>(dim(rank), dim(rank));
    vec lam = zeros<mat>(dim(rank));

    /* --------------------- Some output ---------------------------*/
    if (mpiroot) {
        printElement("it", 5);
        printElement("secs", 14);
        printElement("dual", 14);
        printElement("primal", 14);
        printElement("log10(r-d)", 14);
        printElement("log10(r-p')", 14);
        printElement("log10(sigma)", 14);
        cout << endl;
    }

    /* ################################################################### */
    /* ###################### Boundary point method ###################### */
    /* ################################################################### */


    /* -------------------------- Some variables -------------------------------*/
    vec rhs1(m);
    vec rhs1_local;
    vec rhs_local;
    vec rhs(m);
    mat evp, evn;
    mat evpt, evnt;
    mat WWp, WWn, WWp2;
    uvec Ip, In;
    double j, ic;

    double err_d, err_p, err;
    double errd_local;

    double primal, dual;
    double primal_local;

    // Eigen vectors for rhs and y mapped to the corresponding Armadillo vectors
    Map<VectorXd> rhs_eigen(rhs.begin(), m);
	Map<VectorXd> y_eigen(y.begin(), m);

    MPI_Barrier(MPI_COMM_WORLD);

    /* -------------------------- Outer loop -----------------------------*/
    while (iter <= max_iter) {
        // constant part of right hand side
        rhs1_local = zeros<vec>(m);
        rhs1_local += A_GT.t() * vectorise_sparse(C);
        MPI_Reduce(rhs1_local.begin(), rhs1.begin(), m, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if (mpiroot) {
            rhs1 += res_p / sigma;
        }

        /* ----------------------- Inner loop --------------------------*/
        for (int in_it = 1; in_it <= in_max; in_it++) {
            // determine right hand side of linear system to get y
            rhs_local = A_GT.t() * vectorise(Z);
            MPI_Reduce(rhs_local.begin(), rhs.begin(), m, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

            if (mpiroot) {
                rhs += rhs1;
            }

            /* ################################################################### */
            /* ####################### Sparse linear system ###################### */
            /* ################################################################### */

            /* -------------- solve for Z and y alternatingly -------------- */

            // compute y on root => solve sigma*AA^Ty = sigma*A(C+Z) + A(X)-b
            // using Eigen sparse solver (way faster than Armadillo's)
            if (mpiroot) {
                y_eigen = cholesky.solve(rhs_eigen);
            }

            // broadcast y it to everyone
            // implicit barrier in the Bcast (no need for MPI_Barrier)
            MPI_Bcast(y.begin(), m, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            /* ################################################################### */
            /* ####################### Eigen decomposition ####################### */
            /* ################################################################### */

            // compute A^T(y)
            Aty = reshape(A_GT * y, dimension, dimension);
            Aty = (Aty + Aty.t()) * 0.5;

            // compute W(y)
            // Z is projection of W onto PSD
            W = Aty - C - X / sigma;

            // compute projections to get Wp and Wn
            ev_dec += 1;
            cnt += 1;

            // eigen decomposition
            eig_sym(lam, ev, W);

            Ip = find(lam > 0);
            j = Ip.n_elem;

            if (j > dimension / 2.0) {
                evp = zeros<mat>(dimension, j);

                for (int r = 0; r < j; r++) {
                    ic = Ip(r);
                    evp.col(r) = ev.col(ic) * sqrt(lam(ic));
                }

                WWp = evp * evp.t(); // the projection

                Wp = (WWp + WWp.t()) / 2; // should be symmetric
                V = -sigma * (W - Wp);
            }
            else {
                In = find(lam < 0);
                j = In.n_elem;

                evn = zeros<mat>(dimension, j);

                for (int r = 0; r < j; r++) {
                    ic = In(r);
                    evn.col(r) = ev.col(ic) * sqrt(-lam(ic));
                }

                WWn = -evn * evn.t(); // the projection to negative definite matrices

                // is nan stuff
                V = (WWn + WWn.t()) / 2; // should be symmetric
                Wp = W - V; // W_+
                V = -sigma * V; // -sigma*W_-
            }

            // update Z
            Z = Wp;
        }
        MPI_Barrier(MPI_COMM_WORLD);

        // update X
        X = V;

        // err_d
        errd_local = norm(Z + C - Aty, "fro");
        MPI_Reduce(&errd_local, &err_d, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        // res_p
        resp_local = A_GT.t() * vectorise(X);
        MPI_Reduce(resp_local.begin(), res_p.begin(), m, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (mpiroot) {
            res_p -= b;
        }

        // primal
        primal_local = as_scalar(vectorise_sparse(C).t() * vectorise(X));
        MPI_Reduce(&primal_local, &primal, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        // err_p & dual
        if (mpiroot) {
            err_p = norm(res_p);
            dual = as_scalar(b.t() * y);
        }

        // timer
        end_time = MPI_Wtime();
        secs = end_time - start_time;

        // output every fixed number of iterations
        if (mpiroot) {
            if (fmod(iter, print_iter) == 0) {
                printElement(iter, 5);
                printElement(secs, 14);
                printElement(dual, 14);
                printElement(primal, 14);
                printElement(log10(err_d / (1 + normC)), 14);
                printElement(log10(err_p / (1 + normb)), 14);
                printElement(log10(sigma), 14);
                cout << endl;
            }
        }

        MPI_Bcast(&err_d, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&err_p, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        // update sigma
        if (err_d > 1000 * err_p) {
            sigma *= 1.05;
        }

        iter += 1;
        in_max = min(in_max + in_max_inc, in_max_max); // increase number of inner iter slightly
        err = max(err_p / (1 + normb), err_d / (1 + normC));

        // stopping criterion
        if (err < tol) {
            // reduce cnt & ev decompositions
            MPI_Reduce(&cnt, &cnt_global, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(&ev_dec, &ev_dec_global, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

            // output
            if (mpiroot) {
                printElement(iter, 5);
                printElement(secs, 14);
                printElement(dual, 14);
                printElement(primal, 14);
                printElement(log10(err_d / (1 + normC)), 14);
                printElement(log10(err_p / (1 + normb)), 14);
                printElement(log10(sigma), 14);
                cout << endl;

                cout << "Normal termination" << endl;
                cout << "Eigen value decompositions: " << ev_dec << endl;
                cout << "Total time: " << secs << endl;
            }

            // write solution to file
            string filename = "solution/X" + to_string(rank+1) + ".dat";
            ofstream fout(filename.c_str());
            for (int j = 0; j < X.n_rows; j++) {
                for (int k = 0; k < X.n_cols; k++) {
                    fout << X(j, k) << "     ";
                }
                fout << endl;
            }

            MPI_Abort(MPI_COMM_WORLD, 1);
            return 0;
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    /* ----------------------------------------------------------------------*/
    // reduce cnt & ev decompositions
    MPI_Reduce(&cnt, &cnt_global, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&ev_dec, &ev_dec_global, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Final output
    if (mpiroot) {
        cout << "Maximum iterations reached." << endl;
        cout << "Current error: " << err << ", target: " << tol << endl;
        cout << "Total time: " << secs << endl;
        cout << cnt_global << endl;
    }

    // write solution to file
    string filename = "solution/X" + to_string(rank+1) + ".dat";
    ofstream fout(filename.c_str());
    for (int j = 0; j < X.n_rows; j++) {
        for (int k = 0; k < X.n_cols; k++) {
            fout << X(j, k) << "     ";
        }
        fout << endl;
    }

    MPI_Finalize();
    return 0;
}

#endif // BPM_CPP
