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

	unsigned int n_datasets(x[0]->size());
	unsigned int n_features(x.size());

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
double LinearRegressionSolver::ComputeCost(std::vector<double> &in_thetas) const
{
	double cost(0);

	// initial version.
	//	for i = 1:n_training_samples
	//	dif = (theta'*X(i, :)' - y(i));
	//   J = J + dif*dif;
	//  endfor
	//	J = J / (2 * m);

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

	return cost;
}