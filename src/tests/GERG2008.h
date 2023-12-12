#ifndef AGA8GERG_H_
#define AGA8GERG_H_

#include <vector>
#include <string>

void MolarMassGERG(const std::vector<double> &x, double &Mm);
void PressureGERG(const double T, const double D, const std::vector<double> &x, double &P, double &Z);
void DensityGERG(const int iflag, const double T, const double P, const std::vector<double> &x, double &D, int &ierr, std::string &herr);
void PropertiesGERG(const double T, const double D, const std::vector<double> &x, double &P, double &Z, double &dPdD, double &d2PdD2, double &d2PdTD, double &dPdT, double &U, double &H, double &S, double &Cv, double &Cp, double &W, double &G, double &JT, double &Kappa, double &A);
void SetupGERG();

#endif
