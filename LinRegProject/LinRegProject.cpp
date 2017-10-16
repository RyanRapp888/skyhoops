// LinRegProject.cpp : Defines the entry point for the console application.

#include "LinearRegressionSolver.h"
#include "JunkDrawerUtils.h"
#include "JunkMathUtils.h"
#include <iostream>

int main(int argc, char *argv[])
{
    
	if (argc != 2)
	{
		std::cout << "usage: linreg <csv file>" << std::endl;
		return EXIT_FAILURE;
	}

	std::string csv_file_name(argv[1]);
	std::vector< std::shared_ptr<std::vector<double> > > column_data;

	if (!LoadCSV(csv_file_name, column_data))
	{
		std::cout << "Unable to load csv\n";
		return EXIT_FAILURE;
	}
	
	// The last column of the csv file represents the "Y" data.
	// I pop it off the list of "X" data and make it its own variable (input_ydata)
	std::shared_ptr<std::vector<double> > input_ydata = column_data.back();
	column_data.pop_back();

	LinearRegressionSolver lrs;
	lrs.SetTrainingData(column_data, input_ydata);
	//lrs.PrintSourceData();
	std::vector<double> out_thetas;
	//lrs.SolveUsingClosedFormNormalEquation(out_thetas);
	int n_iterations = 50000;
	double alpha = .05;
	//lrs.NormalizeFeatures();
	lrs.SolveUsingGradientDescent(n_iterations, alpha, out_thetas);

	std::vector<double> input = { 30.08,8,18,1,4,2,2,1,4,5,3,5,1,4,0 };
	double prediction(0);
	if (lrs.GetPrediction(input, prediction))
	{
		std::cout << "prediction = " << prediction << std::endl;
	}
	
   int abc;
	std::cin >> abc;
	
}

