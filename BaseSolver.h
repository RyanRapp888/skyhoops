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

class BaseSolver
{
public:
	bool SetTrainingData(
		std::vector<std::shared_ptr<std::vector<double> > > x,
		std::shared_ptr< std::vector<double> > y);

	void NormalizeFeatures();
	
	bool GetPrediction(std::vector<double> &input_features, double &prediction) const;

	void PrintSourceData() const;

protected:
	bool m_valid = { false };
	
	bool ValidateTrainingData(
		std::vector<std::shared_ptr<std::vector<double> > > x,
		std::shared_ptr< std::vector<double> > y) const;

	MatrixXd m_training_x;
	VectorXd m_training_y;
	NormalizationOutput m_norm_details;
	bool m_result_thetas_calculated = { false };
	std::vector<double> m_result_thetas;
};