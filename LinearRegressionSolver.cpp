#pragma once
#include "LinearRegressionSolver.h"
#include <iostream>

bool LinearRegressionSolver::SetData(std::vector<std::shared_ptr<std::vector<double> > > x,
	std::shared_ptr< std::vector<double> > y)
{
	if (!ValidateSourceData(x, y))
	{
		return false;
	}

	size_t n_datasets(x[0]->size());
	size_t n_features(x.size());

	MatrixXd tmp_x(x[0]->size(), x.size());
	for (int aa = 0; aa < n_features; aa++)
	{
		double *tmp = &((*(x[aa]))[0]);
		tmp_x.col(aa) = Map<VectorXd>(tmp, x[aa]->size());
	}
	m_x = tmp_x;
	double *tmp = &((*y)[0]);
	VectorXd tmp_y = Map<VectorXd>(tmp, n_datasets);
	m_y = tmp_y;
	return true;
}

void LinearRegressionSolver::SolveUsingClosedFormNormalEquation(std::vector<double> &out_thetas) const
{
	MatrixXd tmp_out_thetas;
	tmp_out_thetas = (m_x.transpose() * m_x).inverse() * m_x.transpose() * m_y;

	int n_rows = tmp_out_thetas.rows();
	int n_cols = tmp_out_thetas.cols();

	for (int aa = 0; aa < n_rows; aa++)
	{
		out_thetas.push_back(tmp_out_thetas(aa, 0));
	}
}

void LinearRegressionSolver::PrintSourceData() const
{
	std::cout << "X data = " << m_x;
	std::cout << "Y data = " << m_y;
}

bool LinearRegressionSolver::ValidateSourceData(
	std::vector<std::shared_ptr<std::vector<double> > > x,
	std::shared_ptr< std::vector<double> > y) const
{
	int n_datasets(0);
	int n_features(x.size());

	if (n_features > 0)
	{
		n_datasets = x[0]->size();
	}
	else
	{
		std::cout << "Error: x is empty" << std::endl;
		return false;
	}

	for (int aa = 1; aa < x.size(); aa++)
	{
		if (x[aa]->size() != n_datasets)
		{
			std::cout << "Error: x requires each column to have the same number of rows" << std::endl;
			return false;
		}

	}

	if (y->size() != n_datasets)
	{
		std::cout << "Error: y data must have the same number of rows as x data" << std::endl;
		return false;
	}

	return true;
}

// This will use m_x as input and m_y as target goal
double LinearRegressionSolver::ComputeCost(const MatrixXd &X, const MatrixXd &y, const std::vector<double> &in_thetas) const
{
	double cost(0);

	// initial version.
	//	for i = 1:n_training_samples
	//	dif = (theta'*X(i, :)' - y(i));
	//   J = J + dif*dif;
	//  endfor
	//	J = J / (2 * m);

	/*
	double *rawptr = &(in_thetas[0]);
	Map<VectorXd> thetas(rawptr, 1, in_thetas.size());

	MatrixXd predictions = m_x * thetas;
	predictions = predictions - m_y;

	for (int aa = 0; aa < predictions.rows(); aa++)
	{
		for (int bb = 0; bb < predictions.cols(); bb++)
		{
			predictions(aa, bb) = predictions(aa, bb) * predictions(aa, bb);
		}
	}
	cost = predictions.sum() / (2 * m_y.rows());
	*/

	// eg, m_x is a column vector, datasets (97,1)
	// eg, m_y is a column vector, datasets (97,1)
	// eg, theta is a column vector (2,1)
	double *rawptr = const_cast<double *>(&(in_thetas[0]));
	Map<MatrixXd> thetas(rawptr, in_thetas.size(), 1);

	ArrayXd dif = (X * thetas) - y;
	cost = ((dif * dif).sum()) / (2 * m_y.rows());
	return cost;
}

void LinearRegressionSolver::NormalizeFeatures()
{
   
}


void LinearRegressionSolver::SolveUsingGradientDescent(int n_iterations, double alpha, std::vector<double> &out_thetas) const
{
	//Insert a column of ones into m_x
	MatrixXd tmp_x = MatrixXd(m_x.rows(), m_x.cols()+1);
	tmp_x.col(0).setOnes();
	tmp_x.block(0, 1, m_x.rows(), m_x.cols()) = m_x;
		
	// I hardcode the initial thetas.
	// We can make this a parameter if we want
	std::vector<double> thetas(tmp_x.cols(), 0);
	
	int m = m_y.rows();
	std::vector<double> cost_history_by_iter(n_iterations, 99999);  

	for (int curiter = 0; curiter < n_iterations; curiter++)
	{
		std::vector<double> prev_theta = thetas;

		double *rawptr = &(prev_theta[0]);
		Map<MatrixXd> prevtheta_matrix(rawptr, thetas.size(), 1);
		size_t n_features = tmp_x.cols();
		
		for (int curfeat = 0; curfeat < n_features; curfeat++)
		{
			ArrayXd dif = (tmp_x * prevtheta_matrix) - m_y;
			ArrayXd tmp_deriv = dif * tmp_x.col(curfeat).array();
			double deriv = tmp_deriv.sum() / m;
			thetas[curfeat] = prevtheta_matrix(curfeat) - (alpha * deriv);
		}

		cost_history_by_iter[curiter] = ComputeCost(tmp_x, m_y, thetas);
		if (curiter > 0 && (cost_history_by_iter[curiter] - cost_history_by_iter[curiter - 1] < 0.00001)) break;
	}

	/*
	double *rawptr = const_cast<double *>(&(in_thetas[0]));
	Map<MatrixXd> thetas(rawptr, in_thetas.size(), 1);

	ArrayXd dif = (X * thetas) - y;
	cost = ((dif * dif).sum()) / (2 * m_y.rows());
	*/

	//std::cout << "(theta, cost) = (" << in_thetas[0] << "," << ComputeCost(tmp_x,m_y,in_thetas) << ")" << std::endl;
	

	// hahaha!
	out_thetas = thetas;
	
}