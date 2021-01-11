#include "DataFile.h"
#include "termcolor.h"
#include <fstream>
#include <iostream>
#include <cmath>
#include <regex>

DataFile::DataFile()
{
}

DataFile::DataFile(std::string fileName):
  _fileName(fileName), _solvers(""), _isMatrixMarket(false), _isMatrixSparse(false), _isMatrixSymmetric(false), _isStationnary(true), _timeStep(0.), _isGPO(false), _isGPOPrecJac(false), _isRM(false), _isRMPrecJac(false), _isGC(false), _isGCPrecJac(false), _isFOM(false), _isGMRes(false),   _isSPDFOM(false), _isSPDGMRes(false)
{
}

void DataFile::Initialize(std::string fileName)
{
  _fileName = fileName;
  _solvers = "";
  _isMatrixMarket = false;
  _isMatrixSparse = false;
  _isMatrixSymmetric = false;
  _timeStep = 0.;
  _isGPO = false;
  _isGPOPrecJac = false;
  _isRM = false;
  _isRMPrecJac = false;
  _isGC = false;
  _isGCPrecJac = false;
  _isFOM = false;
  _isGMRes = false;
  _isSPDFOM = false;
  _isSPDGMRes = false;
}

std::string DataFile::cleanLine(std::string &line)
{
  std::string res = line;

  // Remove everything after a possible #
  res = regex_replace(res, std::regex("#.*$"), std::string(""));
  // Replace tabulation(s) by space(s)
  res = regex_replace(res, std::regex("\t"), std::string(" "), std::regex_constants::match_any);
  // Replace multiple spaces by 1 space
  res = regex_replace(res, std::regex("\\s+"), std::string(" "), std::regex_constants::match_any);
  // Remove any leading spaces
  res = regex_replace(res, std::regex("^ *"), std::string(""));

  return res;
}

void DataFile::ReadDataFile()
{
  // Open the data file
  std::ifstream data_file(_fileName.data());
  if (!data_file.is_open())
    {
      std::cout << termcolor::red << "Unable to open file " << _fileName << std::endl;
      std::cout << termcolor::reset;
      exit(-1);
    }
  else
    {
      std::cout << "=====================================================" << std::endl;
      std::cout << "Reading data file " << _fileName << std::endl;
    }
  // Pour stocker chaque ligne
  std::string line;
  // Run through the data_file to find the parameters
  while (getline(data_file, line))
    {
      // Clean line
      std::string proper_line(cleanLine(line));
      if (proper_line.find("MaxIteration") != std::string::npos)
        {
          data_file >> _maxIt;
        }
      if (proper_line.find("KrylovDimension") != std::string::npos)
        {
          data_file >> _dimKrylov;
        }
      if (proper_line.find("Tolerance") != std::string::npos)
        {
          data_file >> _tolerance;
        }
      if (proper_line.find("MatrixSize") != std::string::npos)
        {
          data_file >> _matrixSize;
        }
      if (proper_line.find("IsMatrixSparse") != std::string::npos)
        {
          data_file >> _isMatrixSparse;
        }
      if (proper_line.find("IsMatrixSymmetric") != std::string::npos)
        {
          data_file >> _isMatrixSymmetric;
        }
      if (proper_line.find("IsMatrixMarket") != std::string::npos)
        {
          data_file >> _isMatrixMarket;
        }
      if (proper_line.find("MatrixMarketFile") != std::string::npos)
        {
          data_file >> _matrixMarketFile;
        }
      if (proper_line.find("IsStationnary") != std::string::npos)
        {
          data_file >> _isStationnary;
        }
      if (proper_line.find("TimeStep") != std::string::npos)
        {
          data_file >> _timeStep;
        }
      if (proper_line.find("Solvers") != std::string::npos)
        {
          std::string solver_line;
          getline(data_file, solver_line);
          if (solver_line.find("GPO") != std::string::npos)
            {
              _isGPO = true;
              _solvers += "GPO ";
            }
          if (solver_line.find("GPOPrecJac") != std::string::npos)
            {
              _isGPOPrecJac = true;
              _solvers += "GPOPrecJac ";
            }
          if (solver_line.find("RM") != std::string::npos)
            {
              _isRM = true;
              _solvers += "RM ";
            }
          if (solver_line.find("RMPrecJac") != std::string::npos)
            {
              _isRMPrecJac = true;
              _solvers += "RMPRecJac ";
            }
          if (solver_line.find("GC") != std::string::npos)
            {
              _isGC = true;
              _solvers += "GC ";
            }
          if (solver_line.find("GCPrecJac") != std::string::npos)
            {
              _isGCPrecJac = true;
              _solvers += "GCPrecJac ";
            }
          if (solver_line.find("FOM") != std::string::npos)
            {
              _isFOM = true;
              _solvers += "FOM ";
            }
          if (solver_line.find("GMRes") != std::string::npos)
            {
              _isGMRes = true;
              _solvers += "GMRes ";
            }
          if (solver_line.find("SPDFOM") != std::string::npos)
            {
              _isSPDFOM = true;
              _solvers += "SPDFOM ";
            }
          if (solver_line.find("SPDGMRes") != std::string::npos)
            {
              _isSPDGMRes = true;
              _solvers += "SPDGMRes ";
            }
        }
      if (_isMatrixSparse && !_isMatrixMarket)
        {
          std::cout << termcolor::magenta << "WARNING : Storing a randomly generated matrix in a sparse matrix is not efficient. Switching to dense storage." << std::endl;
          std::cout << termcolor::reset;
          _isMatrixSparse = false;
        }
      if (!_isMatrixMarket && !_isMatrixSymmetric)
        {
          std::cout << termcolor::magenta << "WARNING : The randomly generated Matrix cannot be other than SPD." << std::endl;
          std::cout << termcolor::reset;
          _isMatrixSymmetric = true;
        }
    }
  std::cout << termcolor::green << "File read sucessfully" << std::endl;
  std::cout << termcolor::reset << "=====================================================" << std::endl;
}

void DataFile::printData() const
{
  std::cout << "=====================================================" << std::endl;
  std::cout << "Printing parameters of " << _fileName << std::endl;
  std::cout << "Max number of iterations = " << _maxIt << std::endl;
  std::cout << "Tolerance                = " << _tolerance << std::endl;
  if (_isMatrixMarket)
    {
      std::cout << "Read Matrix From File    = True" << std::endl;
      std::cout << "Matrix Market file       = " << _matrixMarketFile << std::endl;
      if (_isMatrixSymmetric)
        {
          std::cout << "Matrix is Symmetric      = True" << std::endl;
        }
      else
        {
          std::cout << "Matrix is Symmetric      = False" << std::endl;
        }
      if (_isStationnary)
        {
          std::cout << "Stationnary problem      = True" << std::endl; 
        }
      else
        {
          std::cout << "Stationnary problem      = False" << std::endl;
          std::cout << "Time step                = " << _timeStep << std::endl;
        }
    }
  else
    {
      std::cout << "Read Matrix From File    = False" << std::endl;
      std::cout << "Random Matrix Size       = " << _matrixSize << std::endl;
    }
  if (_isMatrixSparse)
    {
      std::cout << "Matrix storage           = Sparse" << std::endl;
    }
  else
    {
      std::cout << "Matrix storage           = Dense" << std::endl;
    }

  // Affichage de la liste des solvers testés.
  std::cout << "Solvers                  = " << _solvers << std::endl;

  // Affichage de la dimension de l'espace de Krylov
  // si un solver de Krylov est utilisé
  if (_isFOM || _isGMRes || _isSPDFOM || _isSPDGMRes)
    {
      std::cout << "Krylov dimension         = " << _dimKrylov << std::endl; 
    }
  std::cout << "=====================================================" << std::endl;
}
