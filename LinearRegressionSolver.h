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
	void SolveUsingGradientDescent(std::vector<double> &out_thetas) const;
	void PrintSourceData() const;
	double ComputeCost(std::vector<double> &in_thetas) const;

private:
	bool m_valid = { false };

	bool ValidateSourceData(
		std::vector<std::shared_ptr<std::vector<double> > > x,
		std::shared_ptr< std::vector<double> > y) const;
	
	MatrixXd m_x;
	VectorXd m_y;
};