/***************************************************************************************************
* Boundary point method, based on augmented Lagrangian
* C++ serial version
* Prepared for block diagonal structure
* Solves: max <C,X> s.t. A(X) = b, X positive semidefinite
*
* data:
* m             number of constraints
* dim           a vector with dimensions of blocks on the diagonal
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
#include <math.h>
#include <vector>
#include <algorithm>
#include <armadillo>
#include <string>
#include <set>
#include <ctime>
#include <iomanip>
#include <typeinfo>
#include <sys/time.h>
#include "helpers.h"

using namespace std;
using namespace arma;


int main(int argc, char **argv)
{
    int i, k;
    int cnt = 0;
    double secs;

    /* ######################################################## */
    /* ###################### Parameters ###################### */  // to be read from file
    /* ######################################################## */

    /* ------------------------- Data ------------------------- */

    vec dim;
    dim.load("data/dim.dat", raw_ascii);
    int p = dim.n_elem; // number of diagonal blocks

    vec b;
    b.load("data/b.dat", raw_ascii);
    int m = b.n_rows;

    // vector of non-zero values in AAT, A_GT's and C's
    vec nnz;
    nnz.load("data/nonzeros.dat", raw_ascii);

    // Eigen AAT
    Eigen::SparseMatrix<double> AAT_eigen(m, m);
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double> > cholesky;
    read_sparse_eigen(AAT_eigen, nnz(0), "data/AAT.dat");
    cholesky.compute(AAT_eigen);

    // A_GT matrix
    vector<string> files;
    for (int i = 1; i <= p; i++) {
        string filename = "data/A_GT" + to_string(i) + ".dat";
        files.push_back(filename);
    }
    vector<sp_mat> A_GT;
    for (int i = 0; i < p; i++) {
        sp_mat temp;
        temp = read_sparse_arma(dim(i)*dim(i), m, nnz(i+1), files[i]);
        A_GT.push_back(temp);
    }

    // C matrix
    files.clear();
    for (int i = 1; i <= p; i++) {
        string filename = "data/C" + to_string(i) + ".dat";
        files.push_back(filename);
    }
    vector<sp_mat> C;
    for (int i = 0; i < p; i++) {
        sp_mat temp;
        temp = read_sparse_arma(dim(i), dim(i), nnz(i+1+p), files[i]);
        C.push_back(temp);
    }

    /* ------------------------- Timer ------------------------- */
    // elapsed time
    struct timespec start, finish;
    clock_gettime(CLOCK_REALTIME, &start);

    /* ------------------------ Input check ------------------------ */
    int diagonal = 0;
    for (i = 0; i < p; i++)
    {
        if (A_GT[i].n_rows != dim(i) * dim(i)) {
            cout << "Dimensions of A_G are not consistent with dim" << endl;
            return 1;
        }
        if (C[i].n_rows != dim(i)) {
            cout << "Dimensions of A_G are not consistent with dim" << endl;
            return 1;
        }
    }

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

    /* ------------------------ Set starting matrices ------------------------ */
    vector<mat> X;
    vector<mat> Z;

    for (i = 0; i < p; i++)
    {
        mat temp = zeros<mat>(dim(i), dim(i));
        X.push_back(temp);
        Z.push_back(temp);
    }

    /* ------------------------ Set starting values ------------------------ */
    // default values
    double sigma = 0.1;         // penalty parameter in the augmented Lagrangian
    double tol = 1e-5;          // tolerance
    double max_iter = 100;      // maximum outer iterations
    double in_max_init = 5;     // starting inner iterations
    double in_max_inc = 0.5;    // increase rate of inner iterations at each outer iteration
    double in_max_max = 30;     // maximum number of inner iterations
    double print_iter = 10;     // output interval

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
    double normC = 0;
    for (i = 0; i < p; i++) {
        normC += norm(C[i], "fro");
    }

    double normb = norm(b);
    double ev_dec = 0;

    double in_max = in_max_init;
    double iter = 1;

    vec res_p = -b;

    for (i = 0; i < p; i++) {
        res_p += A_GT[i].t() * vectorise(X[i]); // primal residue
    }

    vec y = zeros<vec>(m);

    /* ------------------- Some more matrices -----------------------------*/
    vector<mat> Aty;
    vector<mat> W;
    vector<mat> Wp;
    vector<mat> V;
    vector<mat> ev;
    vector<vec> lam;

    mat temp = zeros<mat>(dim(i), dim(i));
    vec temp_lam = zeros<vec>(dim(i));

    for (i = 0; i < p; i++)
    {
//        mat temp = zeros<mat>(dim(i), dim(i));
//        vec temp_lam = zeros<vec>(dim(i));
        Aty.push_back(temp);
        W.push_back(temp);
        Wp.push_back(temp);
        V.push_back(temp);
        ev.push_back(temp);
        lam.push_back(temp_lam);
    }

    /* --------------------- Some output ---------------------------*/
    printElement("it", 5);
    printElement("secs", 12);
    printElement("dual", 12);
    printElement("primal", 12);
    printElement("log10(r-d)", 12);
    printElement("log10(r-p')", 12);
    printElement("log10(sigma)", 12);
    cout << endl;


    /* ################################################################### */
    /* ###################### Boundary point method ###################### */
    /* ################################################################### */


    /* -------------------------- Variables -------------------------------*/
    vec rhs1;
	vec rhs(m);
    mat evp, evn;
    mat evpt, evnt;
    mat WWp, WWn, WWp2;
    uvec Ip, In;
    double j, ic;
    double err_d, err_p, err;
    double primal, dual;

    // Eigen vector for y and rhs mapped to Armadillo vectors
	Map<VectorXd> rhs_eigen(rhs.begin(), m);
	Map<VectorXd> y_eigen(y.begin(), m);

    /* -------------------------- Outer loop -----------------------------*/
    while (iter <= max_iter) {
        // determine right hand side of linear system to get y
        rhs1 = res_p / sigma;

        //constant part of rhs
        for (i = 0; i < p; i++) {
            rhs1 += A_GT[i].t() * vectorise_sparse(C[i]);
        }

        /* ----------------------- Inner loop --------------------------*/
        for (int in_it = 1; in_it <= in_max; in_it++) {
            // solve for Z and y alternatingly
            rhs = rhs1;

            for (i = 0; i < p; i++) {
                rhs += A_GT[i].t() * vectorise(Z[i]);
            }

            // now compute y => solve sigma*AA^Ty = sigma*A(C+Z) + A(X)-b
            // AAT is a vector representing the diagonal

            y_eigen = cholesky.solve(rhs_eigen);

            /* ---------------------------------------------------------------------*/

            // compute A^T(y)
            for (i = 0; i < p; i++) {
                Aty[i] = reshape(A_GT[i] * y, dim(i), dim(i));
                Aty[i] = (Aty[i] + Aty[i].t()) * 0.5;
            }

            // compute W(y)
            // Z is projection of W onto PSD
            for (i = 0; i < p; i++) {
                W[i] = Aty[i] - C[i] - X[i] / sigma;
            }


            // compute projections to get Wp and Wn
            ev_dec += 1;
            for (i = 0; i < p; i++) {
                cnt += 1;
                eig_sym(lam[i], ev[i], W[i]);
                Ip = find(lam[i] > 0);
                j = Ip.n_elem;

                if (j > dim(i) / 2.0) {
                    evp = zeros<mat>(dim(i), j);

                    for (int r = 0; r < j; r++) {
                        ic = Ip(r);
                        evp.col(r) = ev[i].col(ic) * sqrt(lam[i](ic));
                    }

                    WWp = evp * evp.t(); // the projection

                    Wp[i] = (WWp + WWp.t()) / 2; // should be symmetric
                    V[i] = -sigma * (W[i] - Wp[i]);
                }
                else {
                    In = find(lam[i] < 0);
                    j = In.n_elem;

                    evn = zeros<mat>(dim(i), j);

                    for (int r = 0; r < j; r++) {
                        ic = In(r);
                        evn.col(r) = ev[i].col(ic) * sqrt(-lam[i](ic));
                    }

                    WWn = -evn * evn.t(); // the projection to negative definite matrices

                    V[i] = (WWn + WWn.t()) / 2; // should be symmetric
                    Wp[i] = W[i] - V[i]; // W_+
                    V[i] = -sigma * V[i]; // -sigma*W_-
                }
                // determine W and Z
                //Z = Wp;

            }
            Z = Wp;

        }
        //update X
        X = V;
        res_p = -b;

        // some output
        err_d = 0.0;
        err_p = 0.0;
        primal = 0.0;

        for (i = 0; i < p; i++) {
            err_d = err_d + norm(Z[i] + C[i] - Aty[i], "fro");
            res_p = res_p + A_GT[i].t() * vectorise(X[i]);
            primal += as_scalar(vectorise_sparse(C[i]).t() * vectorise(X[i]));
        }

        err_p = norm(res_p);
        dual = as_scalar(b.t() * y);

        clock_gettime(CLOCK_REALTIME, &finish);
        secs = (finish.tv_sec - start.tv_sec);
        secs += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;


        if (fmod(iter, print_iter) == 0) {
            printElement(iter, 5);
            printElement(secs, 12);
            printElement(dual, 12);
            printElement(primal, 12);
            printElement(log10(err_d / (1 + normC)), 12);
            printElement(log10(err_p / (1 + normb)), 12);
            printElement(log10(sigma), 12);
            cout << endl;
        }

        if (err_d > 1000 * err_p) {
            sigma *= 1.05;
        }

        iter += 1;
        in_max = min(in_max + in_max_inc, in_max_max); // increase number of inner iter slightly
        err = max(err_p / (1 + normb), err_d / (1 + normC));

        if (err < tol) {
            printElement(iter, 5);
            printElement(secs, 12);
            printElement(dual, 12);
            printElement(primal, 12);
            printElement(log10(err_d / (1 + normC)), 12);
            printElement(log10(err_p / (1 + normb)), 12);
            printElement(log10(sigma), 12);
            cout << endl;

            cout << "Normal termination" << endl;
            cout << "Eigen value decompositions: " << ev_dec << endl;
            cout << "Total time: " << secs << endl;

            files.clear();
            for (int i = 1; i <= p; i++) {
                string filename = "solution/X" + to_string(i) + ".dat";
                files.push_back(filename);
            }

            for (int i = 0; i < p; i++) {
                ofstream fOUT(files[i].c_str());
                for (int j = 0; j < X[i].n_rows; j++) {
                    for (int k = 0; k < X[i].n_cols; k++) {
                        fOUT << X[i](j, k) << "     ";
                    }
                    fOUT << endl;
                }
            }


            return 0;
        }

    }

    /* ----------------------------------------------------------------------*/
    cout << "Maximum iterations reached." << endl;
    cout << "Current error: " << err << ", target: " << tol << endl;
    cout << "Total time: " << secs << endl;
    cout << cnt << endl;

    files.clear();
    for (int i = 1; i <= p; i++) {
        string filename = "solution/X" + to_string(i) + ".dat";
        files.push_back(filename);
    }

    for (int i = 0; i < p; i++) {
        ofstream fOUT(files[i].c_str());
        for (int j = 0; j < X[i].n_rows; j++) {
            for (int k = 0; k < X[i].n_cols; k++) {
                fOUT << X[i](j, k) << "     ";
            }
            fOUT << endl;
        }
    }

    return 0;
}

#endif // BPM_CPP
