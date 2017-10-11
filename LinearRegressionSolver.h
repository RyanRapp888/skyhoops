#pragma once
#include <vector>
#include <memory>
#include "BaseSolver.h"
#include "Eigen/Dense"

using namespace Eigen;

class LinearRegressionSolver : public BaseSolver
{
public:
	void SolveUsingClosedFormNormalEquation(std::vector<double> &out_thetas);
	void SolveUsingGradientDescent(int n_iterations, double alpha, std::vector<double> &out_thetas);
	
private:
	double ComputeCost(const MatrixXd &X, const MatrixXd &y, const std::vector<double> &in_thetas) const;
		
};