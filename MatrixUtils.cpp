#include "MatrixUtils.h"
#include "termcolor.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <chrono>

using namespace Eigen;


// Lit une matrice dense depuis un fichier au format Matrix Market
void readDenseMatrixFromFile(Eigen::MatrixXd& Matrix, const std::string file_name, bool isSymmetric)
{
  int i(1), j(1), M(0), N(0), NNZ(0);
  std::ifstream matrix_file;
  std::string line;
  matrix_file.open(file_name);
  if (matrix_file.fail())
    {
      std::cout << termcolor::red << "Could not open " << file_name << std::endl;
      std::cout << termcolor::reset << "Please check if the file exists." << std::endl;
      exit(-11);
    }
  std::cout << "=====================================================" << std::endl;
  std::cout << "Reading Dense Matrix from file " + file_name << std::endl;
  matrix_file >> M >> N >> NNZ;
  std::cout << "Number of rows: " << M << std::endl;
  std::cout << "Number of columns: " << N << std::endl;
  std::cout << "Number of non-zeros: " << NNZ << std::endl;
  Matrix.resize(M, N);
  Matrix.setZero();
  while(getline(matrix_file,line))
    {
      std::stringstream ss(line);
      double coef;
      ss >> i >> j >> coef;
      Matrix(i-1,j-1) = coef;
      if (isSymmetric && i != j)
        {
          Matrix(j-1,i-1) = coef; 
        }
    }
  matrix_file.close();
  std::cout << "Dense Matrix successfully created." << std::endl;
  std::cout << "=====================================================" << std::endl;
}


// Lit une matrice creuse depuis un fichier au format Matrix Market
void readSparseMatrixFromFile(Eigen::SparseMatrix<double>& Matrix, const std::string file_name, bool isSymmetric)
{
  int i(1), j(1), M(0), N(0), NNZ(0);
  double coef;
  std::vector<Eigen::Triplet<double>> triplets;
  std::ifstream matrix_file;
  std::string line;
  matrix_file.open(file_name);
  if (matrix_file.fail())
    {
      std::cout << termcolor::red << "Could not open " << file_name << std::endl;
      std::cout << termcolor::reset << "Please check if the file exists." << std::endl;
      exit(-1);
    }
  std::cout << "=====================================================" << std::endl;
  std::cout << "Reading Sparse Matrix from file " + file_name << std::endl;
  matrix_file >> M >> N >> NNZ;
  std::cout << "Number of rows: " << M << std::endl;
  std::cout << "Number of columns: " << N << std::endl;
  std::cout << "Number of non-zeros: " << NNZ << std::endl;  Matrix.resize(M, N);
  Matrix.setZero();
  while(getline(matrix_file,line))
    {
      std::stringstream ss(line);
      ss >> i >> j >> coef;
      triplets.push_back({i-1,j-1,coef});
      if (isSymmetric && i != j)
        {
          triplets.push_back({j-1,i-1,coef}); 
        }
    }
  matrix_file.close();
  Matrix.setFromTriplets(triplets.begin(), triplets.end());
  std::cout << "Sparse Matrix successfully created." << std::endl;
  std::cout << "=====================================================" << std::endl;
}


// Génère une matrice symétrique définie positive aléatoirement.
void setRandomSPDDenseMatrix(Eigen::MatrixXd& Matrix, const int size)
{
  // Random generator
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);
  std::uniform_real_distribution<double> distribution(0.0,1.0);

  // Other variables
  double alpha(0.);
  Eigen::MatrixXd B(size,size), Id(size,size);
  Eigen::VectorXd temp(size); 
  // Matrix generation
  Id.setIdentity();
  for (int i(0) ; i < size ; ++i)
    {
      for (int j(0) ; j < size ; ++j)
        {
          B(i,j) = distribution(generator);
        }
    }
  Matrix = B.transpose() * B;
  for (int i(0) ; i < size ; ++i)
    {
      double sum(0.);
      for (int j(0) ; j < size ; ++j)
        {
          sum += Matrix(i,j);
        }
      temp(i) = abs(sum);
    }
  alpha = temp.maxCoeff();
  Matrix = Id + alpha * Matrix;
}


// Solveur diagonal, utile pour le préconditionneur de Jacobi
template<class T>
void solveDiag(const T& A, VectorXd& Sol, const VectorXd& b)
{
  if (b.size() != A.rows())
    {
      std::cout << termcolor::red << "ERROR::SOLVEDIAG : Matrix and RHS sizes do not match !" << std::endl;
      std::cout << termcolor::reset;
      exit(-1);
    }
  int M(b.size());
  Sol.resize(M);
  for (int i(0) ; i<M ; ++i)
    {
      Sol(i) = b(i)/A.coeff(i,i);
    }
}


// Solveur triangulaire supérieur, utile pour les solveurs de Krylov
// après la décomposition QR, pour la résolution de Rm*ym = gm.
template<class T>
void solveTriangUp(const T& A, Eigen::VectorXd& x, const Eigen::VectorXd& b)
{
  if (b.size() != A.rows())
    {
      std::cout << termcolor::red << "ERROR::SOLVETRIANGUP : Matrix and RHS sizes do not match !" << std::endl;
      std::cout << termcolor::reset;
      exit(-1);
    }
  int M(b.size());
  x.resize(M);
  x(M-1) = b(M-1)/A.coeff(M-1,M-1);
  for (int i(M-2) ; i >=0 ; --i)
    {
      double sum(0.);
      for (int j(i+1) ; j < M ; ++j)
        {
          sum += A.coeff(i,j) * x(j);
        }
      x(i) = (b(i) - sum)/A.coeff(i,i);
    }
}

// Utile pour les version SDP de FOM et GMRes.
// On ne sépare pas la décomposition de la résolution, puisque de toute manière
// on fait une décomposition Cholesky et une résolution par itération de SPDFOM/SPDGMRes
void TridiagCholeskyAndSolve(VectorXd& alpha, VectorXd& beta, VectorXd& Sol, VectorXd& RHS)
{
  // alpha est la diagonale de la matrice à décomposer (m)
  // beta est la sous-diagonale de la matrice à décomposer (m-1)
  int m(alpha.size());
  if (beta.size() != m-1)
    {
      std::cout << "CHOLESKY : Problème de taille." << std::endl;
      exit(-1);
    }

  // Décomposition de Cholesky sans modifier alpha et beta
  VectorXd diag(alpha.size()), subdiag(beta.size());
  diag(0) = sqrt(alpha(0));
  for (int i(1) ; i < m ; ++i)
    {
      subdiag(i-1) = beta(i-1)/diag(i-1);
      diag(i) = sqrt(alpha(i) - pow(subdiag(i-1),2));
    }

  // Résolution du système linéaire
  VectorXd temp(m);
  Sol.resize(m);
  // Algo de descente (L * temp = RHS)
  temp(0) = RHS(0)/diag(0);
  for (int i(1) ; i < m ; ++i)
    {
      temp(i) = (RHS(i) - subdiag(i-1) * temp(i-1))/diag(i);
    }
  // Algo de remontée (L^T * Sol = temp)
  Sol(m-1) = temp(m-1)/diag(m-1);
  for (int i(m-2) ; i >= 0 ; --i)
    {
      Sol(i) = (temp(i) - subdiag(i) * Sol(i+1))/diag(i);
    }
}


// Templates pour les solveurs diagonaux et triangulaires
// Ces méthodes doivent pouvoir prendre des matrices dense et creuses,
// ainsi que des blocks de matrices denses et creuses (Eigen fait la distinction).
template void solveDiag<MatrixXd>(const MatrixXd& A, VectorXd& Sol, const VectorXd& b);
template void solveDiag<Block<MatrixXd>>(const Block<MatrixXd>& A, VectorXd& Sol, const VectorXd& b);
template void solveDiag<SparseMatrix<double>>(const SparseMatrix<double>& A, VectorXd& Sol, const VectorXd& b);
template void solveDiag<Block<SparseMatrix<double>>>(const Block<SparseMatrix<double>>& A, VectorXd& Sol, const VectorXd& b);

template void solveTriangUp<MatrixXd>(const MatrixXd& A, VectorXd& Sol, const VectorXd& b);
template void solveTriangUp<Block<MatrixXd>>(const Block<MatrixXd>& A, VectorXd& Sol, const VectorXd& b);
template void solveTriangUp<SparseMatrix<double>>(const SparseMatrix<double>& A, VectorXd& Sol, const VectorXd& b);
template void solveTriangUp<Block<SparseMatrix<double>>>(const Block<SparseMatrix<double>>& A, VectorXd& Sol, const VectorXd& b);
