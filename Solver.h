#ifndef SOLVER_H
#define SOLVER_H

#include "Eigen/Eigen/Dense"
#include "Eigen/Eigen/Sparse"
#include <fstream>


// Ce fichier définit la classe mère Solver ainsi que toutes ses classes filles,
// qui correspondent aux différents solvers que nous utilisons.


// Cette variable globale sert pour les I/O
#define IO_PRECISION 15

//----------------------------------//
//------------Classe mère-----------//
//----------------------------------//
template<class T>
class Solver
{
protected:
  // On utilise des pointeurs pour pouvoir définir la matrice et les
  // vecteurs indépendamment (en dehors) du solver afin de pouvoir garder
  // les mêmes composants pour tester plusieurs solver, en évitant les
  // copies inutiles et couteuses.

  // Pointeur vers la matrice du système
  T* _pMatrix;

  // Nombre d'itération max et tolérance
  int _maxIt;
  double _tol;
  
  // Pointeurs vers les vecteurs initiaux
  Eigen::VectorXd* _pSol0; // Initial guess
  Eigen::VectorXd* _pRHS;  // Membre de droite

  // Vecteurs solution et résidu
  Eigen::VectorXd _Sol;  // Vecteur Solution
  Eigen::VectorXd _Res;  // Vecteur des résidus

  // Norme L2 du résidu
  double _beta;
  // Facteur de normalisation du test d'arrêt (par exemple la norme de RHS).
  // Ici, dans tout le programme, il est fixé à 1 (pas de normalisation).
  double _beta0;
  
  // Flag pour vérifier que les attributs sont bien initialisés.
  bool _isInitialized;

  // Nom du solver, utile pour les logs.
  std::string _solverName;

public:
  // Constructeurs
  Solver();
  Solver(T& Matrix, Eigen::VectorXd& Sol0, Eigen::VectorXd& RHS, const int maxIt, const double tol);
  
  // Destructeur par défault
  virtual ~Solver() = default;

  // Initialiseur (virtual pour le GC et les solvers de Krylov)
  // L'argument dimKrylov n'est utile et obligatoire que pour les solvers de Krylov.
  virtual void Initialize(T& Matrix, Eigen::VectorXd& Sol0, Eigen::VectorXd& RHS, const int maxIt, const double tol, const double dimKrylov = 0);
  
  // Getters
  int getMaxIt() const {return _maxIt;};
  double getTol() const {return _tol;};
  double getResNorm() const {return _beta;};
  double getRHSNorm() const {return _beta0;};
  const T &getMatrix() const {return *_pMatrix;};
  const Eigen::VectorXd &getSolution() const {return _Sol;};
  const Eigen::VectorXd &getInitialGuess() const {return *_pSol0;};
  const Eigen::VectorXd &getRHS() const {return *_pRHS;};
  const Eigen::VectorXd &getResidual() const {return _Res;};
  std::string getSolverName() const {return _solverName;};
  
  // Setters
  void setMaxIt(const int maxIt) {_maxIt = maxIt;};
  void setTol(const double tol) {_tol = tol;};
  void setMatrix(T& Matrix) {_pMatrix = &Matrix;};
  void setInitialGuess(Eigen::VectorXd& Sol0) {_pSol0 = &Sol0;};
  void setRHS(Eigen::VectorXd& RHS) {_pRHS = &RHS;};

  // Solving and saving
  virtual void oneIteration() = 0;
  // Solve without saving anything
  void solve();
  // Solve and save the Residual norm and the cputime at each iteration.
  void solveAndSave(const std::string& file_name);

  // Logger
  void pass(const std::string& message) const;
  void warning(const std::string& message) const;
  void error(const std::string& message) const;
};

//----------------------------------//
//----------Classes filles----------//
//----------------------------------//


// Gradient à pas optimal.
template<class T>
class GPO: public Solver<T>
{
public:
  // Constructeurs
  GPO();
  GPO(T& Matrix, Eigen::VectorXd& Sol0, Eigen::VectorXd& RHS, const int maxIt, const double tol);
  // Solving and saving
  void oneIteration();
};


// Gradient à pas optimal avec préconditionneur de Jacobi à gauche.
template<class T>
class GPOPrecJac: public Solver<T>
{
private:
  Eigen::VectorXd _q;
public:
  // Constructeurs
  GPOPrecJac();
  GPOPrecJac(T& Matrix, Eigen::VectorXd& Sol0, Eigen::VectorXd& RHS, const int maxIt, const double tol);
  // Initialiseur
  void Initialize(T& Matrix, Eigen::VectorXd& Sol0, Eigen::VectorXd& RHS, const int maxIt, const double tol, const int dimKrylov = 0);
    // Solving and saving
  void oneIteration();
};

// Résidu minimum
template<class T>
class RM: public Solver<T>
{
public:
  // Constructeurs
  RM();
  RM(T& Matrix, Eigen::VectorXd& Sol0, Eigen::VectorXd& RHS, const int maxIt, const double tol);  
  // Solving and saving
  void oneIteration();
};


// Résidu Minimum avec préconditionneur de Jacobi à gauche.
template<class T>
class RMPrecJac: public Solver<T>
{
private:
  Eigen::VectorXd _q;
public:
  // Constructeurs
  RMPrecJac();
  RMPrecJac(T& Matrix, Eigen::VectorXd& Sol0, Eigen::VectorXd& RHS, const int maxIt, const double tol);
  // Initialiseur
  void Initialize(T& Matrix, Eigen::VectorXd& Sol0, Eigen::VectorXd& RHS, const int maxIt, const double tol, const int dimKrylov = 0);
    // Solving and saving
  void oneIteration();
};


// Gradient conjugué
template<class T>
class GC: public Solver<T>
{
private:
  // Le membre _p sert à sauvegarder le vecteur de conjugaison entre chaque itération.
  // C'est la manière de faire la plus propre que j'ai trouvée.
  Eigen::VectorXd _p;

public:
  // Constructeurs
  GC();
  GC(T& Matrix, Eigen::VectorXd& Sol0, Eigen::VectorXd& RHS, const int maxIt, const double tol);
  // Initialiseur requis pour le membre _p
  void Initialize(T& Matrix, Eigen::VectorXd& Sol0, Eigen::VectorXd& RHS, const int maxIt, const double tol, const int dimKrylov = 0);
  // Solving and saving
  void oneIteration();
};


// Gradient conjugué avec préconditionneur de Jacobi à gauche.
template<class T>
class GCPrecJac: public Solver<T>
{
private:
  // Le membre _p sert à sauvegarder le vecteur de conjugaison entre chaque itération.
  // C'est la manière de faire la plus propre que j'ai trouvée.
  Eigen::VectorXd _p;
  Eigen::VectorXd _z;
  
public:
  // Constructeurs
  GCPrecJac();
  GCPrecJac(T& Matrix, Eigen::VectorXd& Sol0, Eigen::VectorXd& RHS, const int maxIt, const double tol);
  // Initialiseur requis pour le membre _p
  void Initialize(T& Matrix, Eigen::VectorXd& Sol0, Eigen::VectorXd& RHS, const int maxIt, const double tol, const int dimKrylov = 0);
  // Solving and saving
  void oneIteration();
};


// Krylov solvers
template<class T>
class KrylovSolver: public Solver<T>
{
protected:
  int _dimKrylov;      // Dimension de l'espace de Krylov
  Eigen::MatrixXd _Vm; // matrice de l'espace de Krylov par Arnoldi
  Eigen::MatrixXd _Hm; // matrice d'Hessenberg dans le cas général
  Eigen::MatrixXd _Qm; // matrice Q de la décomposition QR par la méthode de Givens
  Eigen::MatrixXd _Rm; // matrice R de la decomposition QR par givens

  // Membres pour le cas SPD
  Eigen::VectorXd _HmDiag;        // Diagonale de _Hm
  Eigen::VectorXd _HmSubDiag;     // Sous diagonale de _Hm 

public:
  // Constructeurs
  KrylovSolver();
  KrylovSolver(T& Matrix, Eigen::VectorXd& Sol0, Eigen::VectorXd& RHS, const int maxIt, const double tol, const int dimKrylov);
  // Initialiseur requis pour _dimKrylov
  void Initialize(T& Matrix, Eigen::VectorXd& Sol0, Eigen::VectorXd& RHS, const int maxIt, const double tol, const int dimKrylov);

  // Getters
  int getDimKrylov() const {return _dimKrylov;};
  const Eigen::MatrixXd & GetHm() const {return _Hm;};
  const Eigen::MatrixXd & GetVm() const {return _Vm;};

  // Setters
  void setDimKrylov(int dimKrylov) {_dimKrylov = dimKrylov;};
  
  // Solving and saving
  void Givens(const Eigen::MatrixXd& Matrix); // Givens prend en argument _Hm pour GMRes, et la partie carrée de _Hm pour FOM
  void Arnoldi(const T& Matrix, const Eigen::VectorXd& v);
  void Lanczos(const T& Matrix, const Eigen::VectorXd& v); // Modified Arnoldi for the SPD case
  virtual void oneIteration() = 0;
};


// Full Orthogonalization Method
template<class T>
class FOM: public KrylovSolver<T>
{
public:
  // Constructeurs
  FOM();
  FOM(T& Matrix, Eigen::VectorXd& Sol0, Eigen::VectorXd& RHS, const int maxIt, const double tol, const int dimKrylov);
  // Solving and saving
  virtual void oneIteration();
};


// Generalized Minimum Residual Method
template<class T>
class GMRes: public KrylovSolver<T>
{
public:
  // Constucteurs
  GMRes();
  GMRes(T& Matrix, Eigen::VectorXd& Sol0, Eigen::VectorXd& RHS, const int maxIt, const double tol, const int dimKrylov);
  // Solving and saving
  virtual void oneIteration();
};


// SPD FOM
template<class T>
class SPDFOM: public FOM<T>
{
public:
  // Constucteurs
  SPDFOM();
  SPDFOM(T& Matrix, Eigen::VectorXd& Sol0, Eigen::VectorXd& RHS, const int maxIt, const double tol, const int dimKrylov);
  // Solving and saving
  void oneIteration();
};


// SPD GMRes
template<class T>
class SPDGMRes: public GMRes<T>
{
public:
  // Constucteurs
  SPDGMRes();
  SPDGMRes(T& Matrix, Eigen::VectorXd& Sol0, Eigen::VectorXd& RHS, const int maxIt, const double tol, const int dimKrylov);
  // Solving and saving
  void oneIteration();
};

#endif // SOLVER_H
