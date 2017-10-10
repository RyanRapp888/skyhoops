#pragma once
#include "LinearRegressionSolver.h"
#include <iostream>

/* **************************************
   SolveUsingClosedFormNormalEquation
   This function will use data in m_training_x and m_training_y
   to determine the output thetas.
   Theory is that this function is fast for small to medium datasets
   but becomes unusable with larger data
   out_thetas - The output theta values
*************************************** */
void LinearRegressionSolver::SolveUsingClosedFormNormalEquation(
	std::vector<double> &out_thetas)
{
	MatrixXd tmp_out_thetas;
	tmp_out_thetas = (m_training_x.transpose() * m_training_x).inverse() * m_training_x.transpose() * m_training_y;

	int n_rows = tmp_out_thetas.rows();
	int n_cols = tmp_out_thetas.cols();

	for (int aa = 0; aa < n_rows; aa++)
	{
		m_result_thetas.push_back(tmp_out_thetas(aa, 0));
	}
	
	if (m_result_thetas.size() > 0) m_result_thetas_calculated = true;
}

/* **************************************
   ComputeCost
   Given training data in matrix form (X)
   and the desired target values (y)
   and thetas. Determine the cost.
   This is just one possible cost function, I will add 
   the ability to pass a function pointer to a user defined
   cost function later.
*************************************** */
double LinearRegressionSolver::ComputeCost(const MatrixXd &X, const MatrixXd &y, const std::vector<double> &in_thetas) const
{
	double cost(0);

	double *rawptr = const_cast<double *>(&(in_thetas[0]));
	Map<MatrixXd> thetas(rawptr, in_thetas.size(), 1);

	ArrayXd dif = (X * thetas) - y;
	cost = ((dif * dif).sum()) / (2 * m_training_y.rows());
	return cost;
}

/* **************************************
   SolveUsingGradientDescent
   Uses presupplied m_training_x and m_training_y data.
   Also: before calling this, consider calling NormalizeFeatures().
   Once you have called this function, you can call GetPrediction()
      to get the 'learned' result for a specific set of feature values.
   n_iterations - number of iterations to spend looking for ideal thetas
   alpha - the 'step size' when determining the next iteration
   out_thetas - the output values. 
*************************************** */
void LinearRegressionSolver::SolveUsingGradientDescent(
	int n_iterations, 
	double alpha, 
	std::vector<double> &out_thetas)
{

	//Insert a column of ones into m_x
	MatrixXd tmp_x = MatrixXd(m_training_x.rows(), m_training_x.cols() + 1);
	tmp_x.col(0).setOnes();
	tmp_x.block(0, 1, m_training_x.rows(), m_training_x.cols()) = m_training_x;

	// Right now, I initialize all the thetas to zero. 
	// We can easily add the ability to pass in desired initialized values.
	std::vector<double> thetas(tmp_x.cols(), 0);

	int m = m_training_y.rows();
	std::vector<double> cost_history_by_iter(n_iterations, 99999);

	for (int curiter = 0; curiter < n_iterations; curiter++)
	{
		std::vector<double> prev_theta = thetas;

		double *rawptr = &(prev_theta[0]);
		Map<MatrixXd> prevtheta_matrix(rawptr, thetas.size(), 1);
		size_t n_features = tmp_x.cols();

		for (int curfeat = 0; curfeat < n_features; curfeat++)
		{
			ArrayXd dif = (tmp_x * prevtheta_matrix) - m_training_y;
			ArrayXd tmp_deriv = dif * tmp_x.col(curfeat).array();
			double deriv = tmp_deriv.sum() / m;
			thetas[curfeat] = prevtheta_matrix(curfeat) - (alpha * deriv);
		}

		cost_history_by_iter[curiter] = ComputeCost(tmp_x, m_training_y, thetas);
	}
	out_thetas = thetas;
	m_result_thetas = thetas;
	m_result_thetas_calculated = true;
}
