#include "Solver.h"
#include "termcolor.h"
#include "StringAddOn.h"
#include "MatrixUtils.h"
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>

using namespace Eigen;
using namespace std;
//-------------------------------//
//----------Classe mère----------//
//-------------------------------//

// Constructeurs
template<class T>
Solver<T>::Solver():
  _isInitialized(false)
{
}

template<class T>
Solver<T>::Solver(T& Matrix, VectorXd& Sol0, VectorXd& RHS, const int maxIt, const double tol):
  _pMatrix(&Matrix), _maxIt(maxIt), _tol(tol), _pSol0(&Sol0), _pRHS(&RHS), _Sol(Sol0), _Res(RHS - Matrix * Sol0), _beta(_Res.norm()), _beta0(1.), _isInitialized(true), _solverName()
{
}

// (Ré)Initialiseur
template<class T>
void Solver<T>::Initialize(T& Matrix, VectorXd& Sol0, VectorXd& RHS, const int maxIt, const double tol, const double dimKrylov)
{
  _isInitialized = true;
  _pMatrix = &Matrix;
  _maxIt = maxIt;
  _tol = tol;
  _Sol = Sol0;
  _pSol0 = &Sol0;
  _pRHS = &RHS;
  _Res = RHS - Matrix * Sol0;
  _beta = _Res.norm();
  _beta0 = 1.;
}

// Méthode solve générique
template<class T>
void Solver<T>::solve()
{
  if(!this->_isInitialized)
    {
      this->error("Members are not initialized !");
      return;
    }
  int k(0);
  // Itérations de la méthode
  while ((this->_beta/this->_beta0 > this->_tol) && (k < this->_maxIt))
    {
      this->oneIteration();
      ++k;
    }
  // Logger
  if ((k == this->_maxIt) && (this->_beta/this->_beta0 > this->_tol))
    {
      this->warning("The method did not converge. Normalized residual L2 Norm = " + to_string_with_precision(this->_beta/this->_beta0, IO_PRECISION));
    }
  else
    {
      this->pass("The method converged in " + std::to_string(k) + " iterations ! Normalized residual L2 Norm = " + to_string_with_precision(this->_beta/this->_beta0, IO_PRECISION));
    }
}

// Méthode solve and save générique
template<class T>
void Solver<T>::solveAndSave(const std::string &file_name)
{
  if(!this->_isInitialized)
    {
      this->error("Members are not initialized !");
      return;
    }
  // Fichier de sortie
  std::ofstream outfile(file_name, std::ostream::out);
  int k(0);
  // Sauvegarde du résidu initial
  outfile << 0 << " " << 0. << " " << this->_beta << endl;
  // Démarrage du chrono
  auto start = chrono::high_resolution_clock::now();
  // Itérations de la méthode 
  while ((this->_beta/this->_beta0 > this->_tol) && (k < this->_maxIt))
    {
      this->oneIteration();
      auto checkpoint = chrono::high_resolution_clock::now();
      ++k;
      outfile << k << " " << std::chrono::duration<double,std::milli>(checkpoint-start).count() << " " << _beta << endl;
    }
  // Logger
  if ((k == this->_maxIt) && (this->_beta/this->_beta0 > this->_tol))
    {
      this->warning("The method did not converge. Normalized residual L2 Norm = " + to_string_with_precision(this->_beta/this->_beta0, IO_PRECISION));
    }
  else
    {
      this->pass("The method converged in " + std::to_string(k) + " iterations ! Normalized residual L2 Norm = " + to_string_with_precision(this->_beta/this->_beta0, IO_PRECISION));
    }
}

// Ces 3 méthodes simplifient et rendent plus visuels les logs sur le terminal.
// Les logs sont précédés du nom du solver.
template<class T>
void Solver<T>::pass(const std::string& message) const
{
  std::cout << termcolor::green << "SOLVER::" + this->_solverName + "::PASS : " + message << std::endl;
  std::cout << termcolor::reset;
}

template<class T>
void Solver<T>::warning(const std::string& message) const
{
  std::cout << termcolor::magenta <<  "SOLVER::" + this->_solverName + "::WARNING : " + message << std::endl;
  std::cout << termcolor::reset;
}

template<class T>
void Solver<T>::error(const std::string& message) const
{
  std::cout << termcolor::red << "SOLVER::" + this->_solverName + "::ERROR : " + message << std::endl;
  std::cout << termcolor::reset;
}

//------------------------------------------//
//----------Gradient à Pas Optimal----------//
//------------------------------------------//
template<class T>
GPO<T>::GPO():
  Solver<T>()
{
  this->_solverName = "GPO";
}

template<class T>
GPO<T>::GPO(T& Matrix, VectorXd& Sol0, VectorXd& RHS, const int maxIt, const double tol):
  Solver<T>(Matrix, Sol0, RHS, maxIt, tol)
{
  this->_solverName = "GPO";
}

template<class T>
void GPO<T>::oneIteration()
{
  VectorXd z(*this->_pMatrix * this->_Res);
  double alpha(pow(this->_beta,2)/z.dot(this->_Res));
  this->_Sol += alpha * this->_Res;
  this->_Res -= alpha * z;
  this->_beta = this->_Res.norm();
}


//----------------------------------------------------------------//
//----------Gradient à Pas Optimal Preconditionné Jacobi----------//
//----------------------------------------------------------------//
template<class T>
GPOPrecJac<T>::GPOPrecJac():
  Solver<T>()
{
  this->_solverName = "GPOPrecJac";
}

template<class T>
GPOPrecJac<T>::GPOPrecJac(T& Matrix, VectorXd& Sol0, VectorXd& RHS, const int maxIt, const double tol):
  Solver<T>(Matrix, Sol0, RHS, maxIt, tol)
{
  this->_solverName = "GPOPrecJac";
  solveDiag(Matrix, _q, this->_Res);
}

template<class T>
void GPOPrecJac<T>::Initialize(T& Matrix, VectorXd& Sol0, VectorXd& RHS, const int maxIt, const double tol, const int dimKrylov)
{
  this->_isInitialized = true;
  this->_pMatrix = &Matrix;
  this->_maxIt = maxIt;
  this->_tol = tol;
  this->_Sol = Sol0;
  this->_pSol0 = &Sol0;
  this->_pRHS = &RHS;
  this->_Res = RHS - Matrix * Sol0;
  solveDiag(Matrix, _q, this->_Res);
  this->_beta = this->_Res.norm();
  this->_beta0 = 1;
}

template<class T>
void GPOPrecJac<T>::oneIteration()
{
  VectorXd z, w(*this->_pMatrix * _q);
  solveDiag(*this->_pMatrix, z, w);
  double alpha(_q.squaredNorm()/z.dot(_q));
  this->_Sol += alpha * _q;
  this->_Res -= alpha * w;
  _q -= alpha * z;
  this->_beta = this->_Res.norm();
}


//----------------------------------//
//----------Résidu Minimum----------//
//----------------------------------//
template<class T>
RM<T>::RM():
  Solver<T>()
{
  this->_solverName = "RM";
}

template<class T>
RM<T>::RM(T& Matrix, VectorXd& Sol0, VectorXd& RHS, const int maxIt, const double tol):
  Solver<T>(Matrix, Sol0, RHS, maxIt, tol)
{
  this->_solverName = "RM";
}

template<class T>
void RM<T>::oneIteration()
{
  VectorXd z(*this->_pMatrix * this->_Res);
  double alpha(this->_Res.dot(z)/z.squaredNorm());
  this->_Sol += alpha * this->_Res;
  this->_Res -= alpha * z;
  this->_beta = this->_Res.norm();
}


//----------------------------------------------------------------//
//----------Gradient à Pas Optimal Preconditionné Jacobi----------//
//----------------------------------------------------------------//
template<class T>
RMPrecJac<T>::RMPrecJac():
  Solver<T>()
{
  this->_solverName = "RMPrecJac";
}

template<class T>
RMPrecJac<T>::RMPrecJac(T& Matrix, VectorXd& Sol0, VectorXd& RHS, const int maxIt, const double tol):
  Solver<T>(Matrix, Sol0, RHS, maxIt, tol)
{
  this->_solverName = "RMPrecJac";
  solveDiag(Matrix, _q, this->_Res);
}

template<class T>
void RMPrecJac<T>::Initialize(T& Matrix, VectorXd& Sol0, VectorXd& RHS, const int maxIt, const double tol, const int dimKrylov)
{
  this->_isInitialized = true;
  this->_pMatrix = &Matrix;
  this->_maxIt = maxIt;
  this->_tol = tol;
  this->_Sol = Sol0;
  this->_pSol0 = &Sol0;
  this->_pRHS = &RHS;
  this->_Res = RHS - Matrix * Sol0;
  solveDiag(Matrix, _q, this->_Res);
  this->_beta = this->_Res.norm();
  this->_beta0 = 1;
}

template<class T>
void RMPrecJac<T>::oneIteration()
{
  VectorXd z, w(*this->_pMatrix * _q);
  solveDiag(*this->_pMatrix, z, w);
  double alpha(_q.dot(z)/z.squaredNorm());
  this->_Sol += alpha * _q;
  this->_Res -= alpha * w;
  _q -= alpha * z;
  this->_beta = this->_Res.norm();
}


//-------------------------------------//
//----------Gradient Conjugué----------//
//-------------------------------------//
template<class T>
GC<T>::GC():
  Solver<T>()
{
  this->_solverName = "GC";
}

template<class T>
GC<T>::GC(T& Matrix, VectorXd& Sol0, VectorXd& RHS, const int maxIt, const double tol):
  Solver<T>(Matrix, Sol0, RHS, maxIt, tol)
{
  this->_solverName = "GC";
  _p = this->_Res;
}

template<class T>
void GC<T>::Initialize(T& Matrix, VectorXd& Sol0, VectorXd& RHS, const int maxIt, const double tol, const int dimKrylov)
{
  this->_isInitialized = true;
  this->_pMatrix = &Matrix;
  this->_maxIt = maxIt;
  this->_tol = tol;
  this->_Sol = Sol0;
  this->_pSol0 = &Sol0;
  this->_pRHS = &RHS;
  this->_Res = RHS - Matrix * Sol0;
  _p = this->_Res;
  this->_beta = this->_Res.norm();
  this->_beta0 = 1.;
}

template<class T>
void GC<T>::oneIteration()
{
  VectorXd z(*this->_pMatrix * _p);
  double alpha(pow(this->_beta,2)/z.dot(_p));
  this->_Sol += alpha * _p;
  this->_Res -= alpha * z;
  double gamma(this->_Res.squaredNorm()/pow(this->_beta,2));
  _p = this->_Res + gamma * _p;
  this->_beta = this->_Res.norm();
}


//-----------------------------------------------------------//
//----------Gradient Conjugué Préconditionné Jacobi----------//
//-----------------------------------------------------------//
template<class T>
GCPrecJac<T>::GCPrecJac():
  Solver<T>()
{
  this->_solverName = "GCrecJac";
}

template<class T>
GCPrecJac<T>::GCPrecJac(T& Matrix, VectorXd& Sol0, VectorXd& RHS, const int maxIt, const double tol):
  Solver<T>(Matrix, Sol0, RHS, maxIt, tol)
{
  this->_solverName = "GCPrecJac";
  solveDiag(Matrix, _z, this->_Res);
  _p = _z;
}

template<class T>
void GCPrecJac<T>::Initialize(T& Matrix, VectorXd& Sol0, VectorXd& RHS, const int maxIt, const double tol, const int dimKrylov)
{
  this->_isInitialized = true;
  this->_pMatrix = &Matrix;
  this->_maxIt = maxIt;
  this->_tol = tol;
  this->_Sol = Sol0;
  this->_pSol0 = &Sol0;
  this->_pRHS = &RHS;
  this->_Res = RHS - Matrix * Sol0;
  solveDiag(Matrix, _z, this->_Res);
  _p = _z;
  this->_beta = this->_Res.norm();
  this->_beta0 = 1.;
}

template<class T>
void GCPrecJac<T>::oneIteration()
{
  VectorXd w(*this->_pMatrix * _p);
  double alpha(_z.dot(this->_Res)/_p.dot(w));
  this->_Sol += alpha * _p;
  // Stores the updated residual in a temporary vector
  // because we need to keep the previous residual to
  // compute the conjugate direction gamma
  VectorXd tempRes(this->_Res - alpha * w);
  // We do not need A*p anymore, so instead of creating a
  // new vector, we can just store the needed value in w
  solveDiag(*this->_pMatrix, w, tempRes);
  double gamma(tempRes.dot(w)/(this->_Res.dot(_z)));
  // The updated value of _z is stored in w
  _z = w;
  _p = _z + gamma * _p;
  // The updated value of the residual is in tempRes
  this->_Res = tempRes;
  this->_beta = this->_Res.norm();
}


//-----------------------------------//
//----------Krylov Methods-----------//
//-----------------------------------//
template<class T>
KrylovSolver<T>::KrylovSolver():
  Solver<T>()
{
}

template<class T>
KrylovSolver<T>::KrylovSolver(T& Matrix, VectorXd& Sol0, VectorXd& RHS, const int maxIt, const double tol, const int dimKrylov):
  Solver<T>(Matrix, Sol0, RHS, maxIt, tol)
{
  this->_dimKrylov = dimKrylov;
}

template<class T>
void KrylovSolver<T>::Initialize(T& Matrix, VectorXd& Sol0, VectorXd& RHS, const int maxIt, const double tol, const int dimKrylov)
{
  this->_isInitialized = true;
  this->_pMatrix = &Matrix;
  this->_maxIt = maxIt;
  this->_tol = tol;
  this->_Sol = Sol0;
  this->_pSol0 = &Sol0;
  this->_pRHS = &RHS;
  this->_Res = RHS - Matrix * Sol0;
  this->_beta = this->_Res.norm();
  this->_beta0 = 1.;
  this->_dimKrylov = dimKrylov;
}

template<class T>
void KrylovSolver<T>::Arnoldi(const T& Matrix, const VectorXd& v)
{
  // Dimension de l'espace de Krylov
  int m(_dimKrylov);
  // Matrice de l'espace de Krylov
  _Vm.resize(v.size(), m+1);
  _Vm.setZero();
  _Vm.col(0) = v/v.norm();

  // Matrice de Hessenberg
  _Hm.resize(m+1, m);
  _Hm.setZero();

  // Boucle
  for (int j(0) ; j < m ; ++j)
    {
      VectorXd w(Matrix * _Vm.col(j));
      VectorXd sum(w.size());
      sum.setZero();
      for (int i(0) ; i < j + 1 ; ++i)
        {
          _Hm(i,j) = w.dot(_Vm.col(i));
          sum += _Hm(i,j) * _Vm.col(i);
        }
      w -= sum;
      _Hm(j+1,j) = w.norm();
      if (_Hm(j+1,j) < 1e-10 )
        {
          this->_dimKrylov = j+1;
          _Hm.conservativeResize(this->_dimKrylov + 1, this->_dimKrylov);
          break;
        }
      _Vm.col(j+1) = w/_Hm(j+1,j);
    }
}

template<class T>
void KrylovSolver<T>::Lanczos(const T& Matrix, const VectorXd& v)
{
  // Dimension de l'espace de Krylov
  int m(_dimKrylov);
  // Matrice de l'espace de Krylov
  _Vm.resize(v.size(), m+1);
  _Vm.setZero();
  _Vm.col(0) = v/v.norm();

  // Matrice de Hessenberg
  _HmDiag.resize(m);
  _HmDiag.setZero();
  _HmSubDiag.resize(m);
  _HmSubDiag.setZero();

  // Variables temporaires nécessaires pour la première itération
  double beta(0.);
  VectorXd temp(v.size());
  temp.setZero();
  
  // Boucle
  for (int j(0) ; j < m ; ++j)
    {
      VectorXd w(Matrix * _Vm.col(j) - beta * temp);
      _HmDiag(j) = w.dot(_Vm.col(j));
      w -= _HmDiag(j) * _Vm.col(j);
      _HmSubDiag(j) = w.norm();
      if (_HmSubDiag(j) < 1e-10 )
        {
          this->_dimKrylov = j+1;
          _HmDiag.conservativeResize(this->_dimKrylov + 1);
          _HmSubDiag.conservativeResize(this->_dimKrylov + 1);
          break;
        }
      beta = _HmSubDiag(j);
      temp = _Vm.col(j);
      _Vm.col(j+1) = w/beta;
    }
}

// Décomposition QR de la matrice Matrix en utilisant l'algorithme de Givens
template<class T>
void KrylovSolver<T>::Givens(const MatrixXd& Matrix)
{
  // Initialisation
  _Rm = Matrix;
  _Qm.setIdentity(Matrix.rows(), Matrix.rows());
  double c(0.), s(0.), u1(0.), u2(0.);
  MatrixXd Rotation(Matrix.rows(), Matrix.rows());

  // Boucle
  for (int i(0) ; i < Matrix.rows() - 1 ; ++i)
    {
      // Coefficients de la rotation
      Rotation.setIdentity();
      c = _Rm(i,i)/sqrt(pow(_Rm(i+1,i),2) + pow(_Rm(i,i),2));
      s = -_Rm(i+1,i)/sqrt(pow(_Rm(i+1,i),2) + pow(_Rm(i,i),2));

      // Insertion des coefficients dans la matrice de rotation
      Rotation(i,i) = c;
      Rotation(i+1,i+1) = c;
      Rotation(i+1,i) = -s;
      Rotation(i,i+1) = s;

      // Construction de _Rm (triangulaire supérieure) colonne par colonne
      for (int j(i) ; j < Matrix.cols() ; ++j)
        {
          u1 = _Rm(i,j);
          u2 = _Rm(i+1,j);
          _Rm(i,j) = c * u1 - s * u2;
          _Rm(i+1,j) = s * u1 + c * u2;
        }
      // Construction de _Qm (orthgonale)
      _Qm = _Qm * Rotation;
    }
}


//--------------------------------------------------//
//----------Full Orthogonalization Method-----------//
//--------------------------------------------------//
template<class T>
FOM<T>::FOM():
  KrylovSolver<T>()
{
  this->_solverName = "FOM";
}

template<class T>
FOM<T>::FOM(T& Matrix, VectorXd& Sol0, VectorXd& RHS, const int maxIt, const double tol, const int dimKrylov):
  KrylovSolver<T>(Matrix, Sol0, RHS, maxIt, tol, dimKrylov)
{
  this->_solverName = "FOM";
}

template<class T>
void FOM<T>::oneIteration()
{
  // Arnoldi, puis mise à jour de la dimension de l'espace de Krylov
  // si l'algo d'Arnoldi s'est arrêté avant la fin.
  this->Arnoldi(*this->_pMatrix, this->_Res);
  int m(this->_dimKrylov);
  VectorXd y(m), gm(m);
  // Décomposition QR de la partie carrée de _Hm.
  MatrixXd HmBarre(this->_Hm.topRows(m));
  this->Givens(HmBarre);
  // Résolution de HmBarre * y = beta * e1
  gm = this->_beta * this->_Qm.row(0);
  solveTriangUp(this->_Rm, y, gm);
  // Mise à jour de la solution et du résidu
  this->_Sol += this->_Vm.leftCols(m) * y;
  this->_Res = -this->_Hm(m,m-1) * y(m-1) * this->_Vm.col(m);
  this->_beta = abs(this->_Hm(m,m-1) * y(m-1));
}


//---------------------------------//
//----------GMRes Method-----------//
//---------------------------------//
template<class T>
GMRes<T>::GMRes():
  KrylovSolver<T>()
{
  this->_solverName = "GMRES";
}

template<class T>
GMRes<T>::GMRes(T& Matrix, VectorXd& Sol0, VectorXd& RHS, const int maxIt, const double tol, const int dimKrylov):
  KrylovSolver<T>(Matrix, Sol0, RHS, maxIt, tol, dimKrylov)
{
  this->_solverName = "GMRES";
}

template<class T>
void GMRes<T>::oneIteration()
{
  // Arnoldi, puis mise à jour de la dimension de l'espace de Krylov
  // si l'algo d'Arnoldi s'est arrêté avant la fin.
  this->Arnoldi(*this->_pMatrix, this->_Res);
  int m(this->_dimKrylov);
  VectorXd y(m), gm(m+1);
  // Décomposition QR de _Hm
  this->Givens(this->_Hm);
  // Résolution de _HmBarre * y = beta * e1
  gm = this->_beta * this->_Qm.row(0);
  solveTriangUp(this->_Rm.topRows(m), y, gm.head(m));
  // Mise à jour de la solution et du résidu
  this->_Sol += this->_Vm.leftCols(m) * y;
  this->_Res = gm(m) * this->_Vm * this->_Qm.col(m);
  this->_beta = abs(gm(m));
}


//------------------------------------------------------//
//----------SPD Full Orthogonalization Method-----------//
//------------------------------------------------------//
template<class T>
SPDFOM<T>::SPDFOM():
  FOM<T>()
{
  this->_solverName += "SPD";
}

template<class T>
SPDFOM<T>::SPDFOM(T& Matrix, VectorXd& Sol0, VectorXd& RHS, const int maxIt, const double tol, const int dimKrylov):
  FOM<T>(Matrix, Sol0, RHS, maxIt, tol, dimKrylov)
{
  this->_solverName += "SPD";
}

template<class T>
void SPDFOM<T>::oneIteration()
{
  // Arnoldi SDP
  this->Lanczos(*this->_pMatrix, this->_Res);
  int m(this->_dimKrylov);
  VectorXd y(m), HmSubDiagSquare(this->_HmSubDiag.head(m-1)), betae1(m);
  betae1.setZero(); betae1(0) = this->_beta;
  // Résolution avec Cholesky
  TridiagCholeskyAndSolve(this->_HmDiag, HmSubDiagSquare, y, betae1);
  // Mise à jour de la solution et du résidu
  this->_Sol += this->_Vm.leftCols(m) * y;
  this->_Res = -this->_HmSubDiag(m-1) * y(m-1) * this->_Vm.col(m);
  this->_beta = abs(this->_HmSubDiag(m-1) * y(m-1));
}


//-------------------------------------//
//----------SPD GMRes Method-----------//
//-------------------------------------//
// GMRes adapté pour une matrice sdp. Utilise Arnoldi SDP ainsi que Cholesky tridiagonal.

//-------------------------------------------------
//---------------NE FONCTIONNE PAS-----------------
//-------------------------------------------------


template<class T>
SPDGMRes<T>::SPDGMRes():
  GMRes<T>()
{
  this->_solverName += "SPD";
}

template<class T>
SPDGMRes<T>::SPDGMRes(T& Matrix, VectorXd& Sol0, VectorXd& RHS, const int maxIt, const double tol, const int dimKrylov):
  GMRes<T>(Matrix, Sol0, RHS, maxIt, tol, dimKrylov)
{
  this->_solverName += "SPD";
}

template<class T>
void SPDGMRes<T>::oneIteration()
{
  this->Lanczos(*this->_pMatrix, this->_Res);
  int m(this->_dimKrylov);
  VectorXd y(m), HmSubDiagSquare(this->_HmSubDiag.head(m-1)), betae1(m);
  betae1.setZero(); betae1(0) = this->_beta;
  TridiagCholeskyAndSolve(this->_HmDiag, HmSubDiagSquare, y, betae1);
  this->_Sol += this->_Vm.leftCols(m) * y;
  this->_Res = *this->_pRHS - *this->_pMatrix * this->_Sol;
  this->_beta = this->_Res.norm();
}


//-----------------------------------------------------------------------------------------//
//----------List of supported template classes (Dense and Sparse Eigen matrices)-----------//
//-----------------------------------------------------------------------------------------//
// Classe mère
template class Solver<MatrixXd>;
template class Solver<SparseMatrix<double>>;

// Méthodes de gradient
template class GPO<MatrixXd>;
template class GPO<SparseMatrix<double>>;
template class GPOPrecJac<MatrixXd>;
template class GPOPrecJac<SparseMatrix<double>>;
template class RM<MatrixXd>;
template class RM<SparseMatrix<double>>;
template class RMPrecJac<MatrixXd>;
template class RMPrecJac<SparseMatrix<double>>;
template class GC<MatrixXd>;
template class GC<SparseMatrix<double>>;
template class GCPrecJac<MatrixXd>;
template class GCPrecJac<SparseMatrix<double>>;

// Solvers de Krylov
template class KrylovSolver<MatrixXd>;
template class KrylovSolver<SparseMatrix<double>>;
template class FOM<MatrixXd>;
template class FOM<SparseMatrix<double>>;
template class GMRes<MatrixXd>;
template class GMRes<SparseMatrix<double>>;
template class SPDFOM<MatrixXd>;
template class SPDFOM<SparseMatrix<double>>;
template class SPDGMRes<MatrixXd>;
template class SPDGMRes<SparseMatrix<double>>;
