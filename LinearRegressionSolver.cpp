#pragma once
#include "LinearRegressionSolver.h"
#include <iostream>

/* **************************************
   SetTrainingData:
   Use this function to set the training data that this
   class will use to determine output thetas.
   x - list of pointers to column data, one column per feature
   y - pointer to desired y column data
*************************************** */
bool LinearRegressionSolver::SetTrainingData(
	std::vector<std::shared_ptr<std::vector<double> > > x,
	std::shared_ptr< std::vector<double> > y)
{
	if (!ValidateTrainingData(x, y))
	{
		return false;
	}

	size_t n_datasets(x[0]->size());
	size_t n_features(x.size());

	// We convert our input data into a matrix.
	MatrixXd tmp_x(x[0]->size(), x.size());
	for (int aa = 0; aa < n_features; aa++)
	{
		double *tmp = &((*(x[aa]))[0]);
		tmp_x.col(aa) = Map<VectorXd>(tmp, x[aa]->size());
	}
	m_training_x = tmp_x;

	// We convert our y data into a matrix
	double *tmp = &((*y)[0]);
	VectorXd tmp_y = Map<VectorXd>(tmp, n_datasets);
	m_training_y = tmp_y;
	return true;
}

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
PrintSourceData

*************************************** */
void LinearRegressionSolver::PrintSourceData() const
{
	std::cout << "X data = " << m_training_x;
	std::cout << "Y data = " << m_training_y;
}

/* **************************************
ValidateTrainingData
   Validate that data is valid (non_zero) and 
   that each column has the same number of rows. 
   x - list of pointers to column data, one column per feature
   y - pointer to desired y column data
*************************************** */
bool LinearRegressionSolver::ValidateTrainingData(
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
   NormalizeFeatures
   This functions operates on the presupplied m_training_x
   data. It will replace m_training_x with the new normalized 
   values. It will also store the mean and std_dev columns for
   use later in calculating predicitions.
*************************************** */
void LinearRegressionSolver::NormalizeFeatures()
{
	MatrixXd X_norm = m_training_x;
	int n_features = m_training_x.cols();
	int n_datasets = m_training_x.rows();
	std::vector<double> means_per_feature(n_features, 0);
	std::vector<double> sigmas_per_feature(n_features, 0);

	// standard deviation same as octave
	// sqrt(1 / (N - 1) SUM_i(x(i) - mean(x)) ^ 2)
	
	for (int curfeat = 0; curfeat < n_features; curfeat++)
	{
		means_per_feature[curfeat] = m_training_x.col(curfeat).mean();

		double stand_dev_accum(0);
		for (int aa = 0; aa < n_datasets; aa++)
		{
			stand_dev_accum += pow(m_training_x(aa, curfeat) - means_per_feature[curfeat], 2);
		}
		double tmp = (1.0f / (n_datasets - 1)) * stand_dev_accum;
		sigmas_per_feature[curfeat] = sqrt(tmp);
	}

	for (int curfeat2 = 0; curfeat2 < n_features; curfeat2++)
	{
		if (sigmas_per_feature[curfeat2] > 0.00001 ||
			sigmas_per_feature[curfeat2] < -0.00001)
		{
			for (int currow = 0; currow < n_datasets; currow++)
			{
				X_norm(currow, curfeat2) = (m_training_x(currow, curfeat2) - means_per_feature[curfeat2]) / sigmas_per_feature[curfeat2];
			}
		}
		else
		{
			X_norm.col(curfeat2).setZero();
		}
	}
	m_training_x = X_norm;
	m_norm_details.mean_per_feature = means_per_feature;
	m_norm_details.stddev_per_feature = sigmas_per_feature;
	m_norm_details.used_normalization = true;
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

bool LinearRegressionSolver::GetPrediction(std::vector<double> &input_features, double &prediction) const
{
	// initialize prediction to -1;
	prediction = -1;
	if (!m_result_thetas_calculated)
	{
		std::cout << "Error: you must calculate result thetas before requesting a prediction" << std::endl;
		return false;
	}

	if ((input_features.size()+1) != m_result_thetas.size())
	{
		std::cout << "Error: you did not pass in the correct number of feature values" << std::endl;
		return false;
	}

	if (m_norm_details.used_normalization &&
		(input_features.size() != m_norm_details.mean_per_feature.size() ||
			input_features.size() != m_norm_details.stddev_per_feature.size()))
	{
		std::cout << "Error: the number of features passed in for the prediction does"
			<< " not match the number size of the data stored during normalization";
		return false;
	}
	
	MatrixXd input_vals(1, m_result_thetas.size());
	input_vals(0,0) = 1;
	for (int aa = 0; aa < input_features.size(); aa++)
	{
	   input_vals(0, aa + 1) = input_features[aa];
	}
	

	if (m_norm_details.used_normalization)
	{
		for (int bb = 0; bb < m_norm_details.mean_per_feature.size(); bb++)
		{
			double curval = input_vals(0, bb + 1);
			input_vals(0, bb + 1) = (curval - m_norm_details.mean_per_feature[bb]) / m_norm_details.stddev_per_feature[bb];
		}
	}
	
	double *rawptr = const_cast<double *>(&(m_result_thetas[0]));
	Map<MatrixXd> result_thetas(rawptr, m_result_thetas.size(), 1);

	prediction = (input_vals * result_thetas).sum();
	return true;
}