#ifndef MATRIX_UTILS_H
#define MATRIX_UTILS_H

#include "Eigen/Eigen/Dense"
#include "Eigen/Eigen/Sparse"

// Les méthodes prorotypées dans ce fichier sont dédiées à la création/manipulation
// de matrices.
// Dans l'idéal, il aurait fallu que ce soit des méthodes des classes MatrixXd et
// SparseMatrix<double> de Eigen, mais la solution proposée par Eigen pour ajouter
// des méthodes à des classes déjà existantes est relativement complexe... Nous nous
// contenterons donc de ces fonctions, qui prennent une référence vers une matrice en argument.


// Crée/remplit la matrice dense/creuse Matrix depuis un fichier au format Matrix Market.
// Pour les matrices symétriques, Matrix Market ne stocke que la diagonale et la partie triangulaire
// supérieure ou inférieure. On doit donc indiquer comment lire le fichier avec un booléen.
void readDenseMatrixFromFile(Eigen::MatrixXd& Matrix, const std::string file_name, bool isSymmetric);
void readSparseMatrixFromFile(Eigen::SparseMatrix<double>& Matrix, const std::string file_name, bool isSymmetric);


// Remplit la matrice Matrix sous la forme Id + alpha * B^T*B où B est une matrice d'éléments
// choisis aléatoirement entre 0 et 1, et alpha un coefficient calculé en fonction des
// éléments de B^T*B. 
void setRandomSPDDenseMatrix(Eigen::MatrixXd& Matrix, const int size);

// Ces méthodes résolvent le système Ax=b dans les cas simples où A est :
//   - diagonale
//   - triangulaire supérieure ou inférieure.
template<class T>
void solveDiag(const T& A, Eigen::VectorXd& x, const Eigen::VectorXd& b);
template<class T>
void solveTriangUp(const T& A, Eigen::VectorXd& x, const Eigen::VectorXd& b);

// Cette méthode effectue la décomposition de Cholesky d'une matrice
// tridiagonale SDP, dont les coefs non nuls (diagonale et
// sous/sur-diagonale) sont placés dans les vecteurs alpha et beta.
// La décomposition est stockée dans ces mêmes vecteurs afin
// d'économiser de la place en mémoire.
void TridiagCholeskyAndSolve(Eigen::VectorXd& alpha, Eigen::VectorXd& beta, Eigen::VectorXd& Sol, Eigen::VectorXd& RHS);


#endif // MATRIX_UTILS_H
