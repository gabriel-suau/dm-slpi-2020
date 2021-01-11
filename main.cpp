/*
 * Solver.h/cpp : Classes de solvers, algorithmes de résolution...
 * StringAddOn.h : Petite fonction chinée sur StackOverflow pour contrôler la précision des I/O (header only)
 * MatrixUtils.h/cpp : Opérations sur les matrices (lecture dans un fichier, résolution de systèmes diagonaus/triangulaires, cholesky tridiagonal)
 * DataFile.h/cpp : Lecture du fichier de paramètres
 * termcolor.h : Librairie header only chinée sur Github pour avoir des I/O en couleur sur le terminal (pratique pour les logs).
*/

#include "Solver.h"
#include "StringAddOn.h"
#include "MatrixUtils.h"
#include "DataFile.h"
#include "termcolor.h"
#include <iostream>
#include <algorithm>
#include <chrono>
#include <random>

using namespace Eigen;
using namespace std;

int main(int argc, char** argv)
{
  if (argc < 2)
    {
      cout << "Please, enter the name of your data file." << endl;
      exit(-1);
    }
  
  // Data File
  DataFile* DF = new DataFile(argv[1]);
  DF->ReadDataFile();
  DF->printData();
  
  // I/O precision
  std::cout.precision(IO_PRECISION);

  // Paramètres des tests
  const int maxIt(DF->getMaxIt());
  const double tolerance(DF->getTolerance());
  const int dimKrylov(DF->getDimKrylov());
  const string matrixFile(DF->getMatrixMarketFile());
  const bool isMatrixSymmetric(DF->isMatrixSymmetric());
  const bool isStationnary(DF->isStationnary());
  const double timeStep(DF->getTimeStep());
  
  // Solvers, matrices, initial guess and RHS
  Solver<MatrixXd>* DSolver;
  Solver<SparseMatrix<double>>* SSolver;
  MatrixXd Den;
  SparseMatrix<double> Spa;
  VectorXd x0, b, x;

  // Si matrice creuse, choisir de lire dans le fichier indiqué.
  // Le cas où l'utilisateur choisirait un stockage creux pour la matrice
  // random SPD est géré en amont, lors de la lecture du fichier de données.
  if (DF->isMatrixSparse())
    {
      readSparseMatrixFromFile(Spa, matrixFile, isMatrixSymmetric);
      // Modif si le problème est instationnaire
      if (!isStationnary)
        {
          SparseMatrix<double> Id(Spa.rows(), Spa.cols()); Id.setIdentity();
          Spa = Id + timeStep * Spa;
        }
      x0.setOnes(Spa.rows());
      b.setOnes(Spa.rows());
      if (DF->isGPO())
        {
          SSolver = new GPO<SparseMatrix<double>>(Spa, x0, b, maxIt, tolerance);
          SSolver->solveAndSave("norme_residu_GPO_sparse.txt");
          delete SSolver;
        }
      if (DF->isGPOPrecJac())
        {
          SSolver = new GPOPrecJac<SparseMatrix<double>>(Spa, x0, b, maxIt, tolerance);
          SSolver->solveAndSave("norme_residu_GPOPrecJac_sparse.txt");
          delete SSolver;
        }
      if (DF->isRM())
        {
          SSolver = new RM<SparseMatrix<double>>(Spa, x0, b, maxIt, tolerance);
          SSolver->solveAndSave("norme_residu_RM_sparse.txt");
          delete SSolver;
        }
      if (DF->isRMPrecJac())
        {
          SSolver = new RMPrecJac<SparseMatrix<double>>(Spa, x0, b, maxIt, tolerance);
          SSolver->solveAndSave("norme_residu_RMPrecJac_sparse.txt");
          delete SSolver;
        }
      if (DF->isGC())
        {
          SSolver = new GC<SparseMatrix<double>>(Spa, x0, b, maxIt, tolerance);
          SSolver->solveAndSave("norme_residu_GC_sparse.txt");
          delete SSolver;
        }
      if (DF->isGCPrecJac())
        {
          SSolver = new GCPrecJac<SparseMatrix<double>>(Spa, x0, b, maxIt, tolerance);
          SSolver->solveAndSave("norme_residu_GCPrecJac_sparse.txt");
          delete SSolver;
        }
      if (DF->isFOM())
        {
          SSolver = new FOM<SparseMatrix<double>>(Spa, x0, b, maxIt, tolerance, dimKrylov);
          SSolver->solveAndSave("norme_residu_FOM_sparse.txt");
          delete SSolver;
        }
      if (DF->isGMRes())
        {
          SSolver = new GMRes<SparseMatrix<double>>(Spa, x0, b, maxIt, tolerance, dimKrylov);
          SSolver->solveAndSave("norme_residu_GMRes_sparse.txt");
          delete SSolver;
        }
      if (DF->isSPDFOM())
        {
          SSolver = new SPDFOM<SparseMatrix<double>>(Spa, x0, b, maxIt, tolerance, dimKrylov);
          SSolver->solveAndSave("norme_residu_SPDFOM_sparse.txt");
          delete SSolver;
        }
      if (DF->isSPDGMRes())
        {
          SSolver = new SPDGMRes<SparseMatrix<double>>(Spa, x0, b, maxIt, tolerance, dimKrylov);
          SSolver->solveAndSave("norme_residu_SPDGMRes_sparse.txt");
          delete SSolver;
        }
    }
  // Sinon, selon le cas, lire la matrice dans un fichier ou générer une matrice SPD aléatoirement.
  else if (!DF->isMatrixSparse())
    {
      if (DF->isMatrixMarket())
        {
          readDenseMatrixFromFile(Den, matrixFile, isMatrixSymmetric); 
        }
      else
        {
          setRandomSPDDenseMatrix(Den, DF->getMatrixSize());
        }
      x0.setOnes(Den.rows());
      b.setOnes(Den.rows());
      if (DF->isGPO())
        {
          DSolver = new GPO<MatrixXd>(Den, x0, b, maxIt, tolerance);
          DSolver->solveAndSave("norme_residu_GPO.txt");
          delete DSolver;
        }
      if (DF->isGPOPrecJac())
        {
          DSolver = new GPOPrecJac<MatrixXd>(Den, x0, b, maxIt, tolerance);
          DSolver->solveAndSave("norme_residu_GPOPrecJac.txt");
          delete DSolver;
        }
      if (DF->isRM())
        {
          DSolver = new RM<MatrixXd>(Den, x0, b, maxIt, tolerance);
          DSolver->solveAndSave("norme_residu_RM.txt");
          delete DSolver;
        }
      if (DF->isRMPrecJac())
        {
          DSolver = new RMPrecJac<MatrixXd>(Den, x0, b, maxIt, tolerance);
          DSolver->solveAndSave("norme_residu_RMPrecJac.txt");
          delete DSolver;
        }
      if (DF->isGC())
        {
          DSolver = new GC<MatrixXd>(Den, x0, b, maxIt, tolerance);
          DSolver->solveAndSave("norme_residu_GC.txt");
          delete DSolver;
        }
      if (DF->isGCPrecJac())
        {
          DSolver = new GCPrecJac<MatrixXd>(Den, x0, b, maxIt, tolerance);
          DSolver->solveAndSave("norme_residu_GCPrecJac.txt");
          delete DSolver;
        }
      if (DF->isFOM())
        {
          DSolver = new FOM<MatrixXd>(Den, x0, b, maxIt, tolerance, dimKrylov);
          DSolver->solveAndSave("norme_residu_FOM.txt");
          delete DSolver;
        }
      if (DF->isGMRes())
        {
          DSolver = new GMRes<MatrixXd>(Den, x0, b, maxIt, tolerance, dimKrylov);
          DSolver->solveAndSave("norme_residu_GMRes.txt");
          delete DSolver;
        }
      if (DF->isSPDFOM())
        {
          DSolver = new SPDFOM<MatrixXd>(Den, x0, b, maxIt, tolerance, dimKrylov);
          DSolver->solveAndSave("norme_residu_SPDFOM.txt");
          delete DSolver;
        }
      if (DF->isSPDGMRes())
        {
          DSolver = new SPDGMRes<MatrixXd>(Den, x0, b, maxIt, tolerance, dimKrylov);
          DSolver->solveAndSave("norme_residu_SPDGMRes.txt");
          delete DSolver;
        }
    }
  // Normalement ce cas n'arrive jamais, mais ça nous fait un garde fou.
  else
    {
      cout << termcolor::red << "Le scénario choisi n'est pas implémenté." << endl;
      cout << termcolor::reset << "Fin du programme." << endl;
      exit(-1);
    }

  return 0;
}
