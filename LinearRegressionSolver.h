#pragma once
#include <vector>
#include <memory>
#include "Eigen/Dense"

using namespace Eigen;

struct NormalizationOutput
{
	bool used_normalization = { false };
	std::vector<double> mean_per_feature;
	std::vector<double> stddev_per_feature;
};

class LinearRegressionSolver
{
public:
	bool SetTrainingData(
		std::vector<std::shared_ptr<std::vector<double> > > x,
		std::shared_ptr< std::vector<double> > y);

	void SolveUsingClosedFormNormalEquation(std::vector<double> &out_thetas);

	void NormalizeFeatures();
	void SolveUsingGradientDescent(int n_iterations, double alpha, std::vector<double> &out_thetas);
		
	bool GetPrediction(std::vector<double> &input_features, double &prediction) const;

	void PrintSourceData() const;

private:
	bool m_valid = { false };
	double ComputeCost(const MatrixXd &X, const MatrixXd &y, const std::vector<double> &in_thetas) const;

	bool ValidateTrainingData(
		std::vector<std::shared_ptr<std::vector<double> > > x,
		std::shared_ptr< std::vector<double> > y) const;
	
	
	MatrixXd m_training_x;
	VectorXd m_training_y;
	NormalizationOutput m_norm_details;
	bool m_result_thetas_calculated = { false };
	std::vector<double> m_result_thetas;
};