#ifndef DATA_FILE_H
#define DATA_FILE_H

#include <iostream>
#include <string>

class DataFile
{
private:
  // Nom du fichier
  std::string _fileName;

  // String contenant la liste des solvers.
  // Sert uniquement à l'affichage des infos sur le terminal.
  std::string _solvers;
  
  // Paramètres
  int _maxIt;
  int _dimKrylov;
  double _tolerance;
  std::string _matrixMarketFile;
  int _matrixSize;
  bool _isMatrixMarket;
  bool _isMatrixSparse;
  bool _isMatrixSymmetric;
  bool _isStationnary;
  double _timeStep;
  
  // Liste des solvers
  bool _isGPO;
  bool _isGPOPrecJac;
  bool _isRM;
  bool _isRMPrecJac;
  bool _isGC;
  bool _isGCPrecJac;
  bool _isFOM;
  bool _isGMRes;
  bool _isSPDFOM;
  bool _isSPDGMRes;
  
public:
  // Constructeurs
  DataFile();
  DataFile(std::string fileName);
  
  // Initialise l'objet
  void Initialize(std::string fileName);

  // Lit le fichier
  void ReadDataFile();

  // Nettoyer une ligne du fichier
  std::string cleanLine(std::string &line);

  // Getters
  int getMaxIt() const {return _maxIt;};
  int getDimKrylov() const {return _dimKrylov;};
  double getTolerance() const {return _tolerance;};
  std::string getMatrixMarketFile() const {return _matrixMarketFile;};
  int getMatrixSize() const {return _matrixSize;};
  bool isMatrixMarket() const {return _isMatrixMarket;};
  bool isMatrixSparse() const {return _isMatrixSparse;};
  bool isMatrixSymmetric() const {return _isMatrixSymmetric;};
  bool isStationnary() const {return _isStationnary;};
  double getTimeStep() const {return _timeStep;};
  bool isGPO() const {return _isGPO;};
  bool isGPOPrecJac() const {return _isGPOPrecJac;};
  bool isRM() const {return _isRM;};
  bool isRMPrecJac() const {return _isRMPrecJac;};
  bool isGC() const {return _isGC;};
  bool isGCPrecJac() const {return _isGCPrecJac;};
  bool isFOM() const {return _isFOM;};
  bool isGMRes() const {return _isGMRes;};
  bool isSPDFOM() const {return _isSPDFOM;};
  bool isSPDGMRes() const {return _isSPDGMRes;};

  // Affichage des paramètres
  void printData() const;
};

#endif // DATA_FILE_H
