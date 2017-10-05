#pragma once

#include <fstream>
#include <vector>
#include <memory>

bool LoadCSV(const std::string &csv_filename, std::vector< std::shared_ptr<std::vector<double> > > &column_data);
