/**********************
* Auxiliary functions
***********************
*/

#ifndef HELPERS_H
#define HELPERS_H

#include <string>
#include <iostream>
#include <fstream>
#include <math.h>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <string>
using namespace std;

#include <armadillo>
using namespace arma;

#include <Eigen/Sparse>
using namespace Eigen;

// display data in tabular format
template<typename T> void printElement(T t, const int& width)
{
    const char separator = ' ';
    cout << left << setw(width) << setfill(separator) << setprecision(8) << t;
}

// read Armadillo sparse matrix
sp_mat read_sparse_arma(int nRows, int nCols, int nnz, string filename)
{
    ifstream fIN(filename.c_str());
    string line;

    umat locations(2, nnz);
    vec values(nnz);

    int index = 0;
    while (getline(fIN, line) && !line.empty()) {
        stringstream stream(line);

        int check = 0;
        while (stream) {
            double num;
            stream >> num;
            if (!stream)
                break;

            if (check == 0)
                locations(0, index) = num - 1;
            else if (check == 1)
                locations(1, index) = num - 1;
            else if (check == 2)
                values(index) = num;

            check++;
        }
        index++;
    }

    sp_mat A(locations, values, nRows, nCols);
    return A;
}

// vectorise Armadillo sparse matrix
sp_mat vectorise_sparse(sp_mat &a)
{
    int N, M;
    N = a.n_rows;
    M = a.n_cols;

    sp_mat temp(N * M, 1);

    int counter = 0;
    for (int r = 0; r < N; r++) {
        for (int c = 0; c < M; c++) {
            if (a(c, r) != 0) {
                temp(counter, 0) = a(c, r);
            }
            counter++;
        }
    }

    return temp;
}

// read Eigen sparse matrix
void read_sparse_eigen(Eigen::SparseMatrix<double> &A, int nnz, string filename)
{
    ifstream fIN(filename.c_str());
    string line;

    typedef Eigen::Triplet<double> T;
    std::vector<T> tripletList;
    tripletList.reserve(nnz);
    int i, j;
    double v_ij;

    while (getline(fIN, line) && !line.empty()) {
        stringstream stream(line);

        int check = 0;
        while (stream) {
            double num;
            int index;

            if (check == 0) {
                stream >> index;
                i = index - 1;
            }
            else if (check == 1) {
                stream >> index;
                j = index - 1;
            }
            else if (check == 2) {
                stream >> num;
                v_ij = num;
                break;
            }

            check++;
        }
        tripletList.push_back(T(i, j, v_ij));
    }

    A.setFromTriplets(tripletList.begin(), tripletList.end());
}

// read Armadillo sparse matrix without knowing the number of non-zero entries in advance
void read_sparse_arma_nnz(sp_mat &a, string filename)
{
    ifstream fIN(filename.c_str());
    string line;

    while (getline(fIN, line) && !line.empty()) {
        stringstream stream(line);
        vector<double> coord;

        while (stream) {
            double num;
            stream >> num;
            if (!stream)
                break;
            coord.push_back(num);
        }
        a(coord[0]-1, coord[1]-1) = coord[2];
    }
}

// read Eigen sparse matrix without knowing the number of non-zero entries in advance
void read_sparse_eigen_nnz(Eigen::SparseMatrix<double> &a, string filename)
{
    ifstream fIN(filename.c_str());
    string line;

    while (getline(fIN, line) && !line.empty()) {
        stringstream stream(line);
        vector<double> coord;

        while (stream) {
            double num;
            stream >> num;
            if (!stream)
                break;
            coord.push_back(num);
        }
        a.insert(coord[0]-1, coord[1]-1) = coord[2];
    }
}

#endif // HELPERS_H
