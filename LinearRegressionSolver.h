#pragma once
#include <vector>
#include <memory>
#include "Eigen/Dense"

using namespace Eigen;

class LinearRegressionSolver
{
public:
	bool SetData(
		std::vector<std::shared_ptr<std::vector<double> > > x,
		std::shared_ptr< std::vector<double> > y);

	void SolveUsingClosedFormNormalEquation(std::vector<double> &out_thetas) const;
	void SolveUsingGradientDescent(int n_iterations, double alpha, std::vector<double> &out_thetas) const;
	void PrintSourceData() const;
	

private:
	bool m_valid = { false };
	double ComputeCost(const MatrixXd &X, const MatrixXd &y, const std::vector<double> &in_thetas) const;
	void NormalizeFeatures();
	bool ValidateSourceData(
		std::vector<std::shared_ptr<std::vector<double> > > x,
		std::shared_ptr< std::vector<double> > y) const;
	
	MatrixXd m_x;
	MatrixXd m_x_normalized;
	VectorXd m_y;
};