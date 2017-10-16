#pragma once
#include "JunkDrawerUtils.h"
#include <fstream>
#include <iostream>
#include <sstream>

bool LoadCSV(const std::string &csv_filename, std::vector< std::shared_ptr<std::vector<double> > > &column_data)
{
	std::ifstream input_file(csv_filename);
	if (!input_file.is_open())
	{
		std::cout << "Failed to open input file" << std::endl;
		return EXIT_FAILURE;
	}
	
	std::string curline;
	bool header_defined(false);
	int header_n_columns(1);

	while (getline(input_file, curline))
	{
		if (curline.find(',') == std::string::npos) continue;

		int aa = 0;
		unsigned int curlen = curline.size();
		unsigned int curline_n_columns(1);

		for (int aa = 0; aa < curlen; aa++)
		{
			if (curline[aa] == ',')
			{
				if (!header_defined) header_n_columns++;
				curline_n_columns++;
				curline[aa] = ' ';
			}
		}

		bool load_data_from_line(false);

		if (header_defined)
		{
			if (curline_n_columns == header_n_columns)
			{
				load_data_from_line = true;
			}
		}
		else
		{
			load_data_from_line = true;
			header_defined = true;
			column_data.resize(header_n_columns, nullptr);

			for (int aa = 0; aa < header_n_columns; aa++)
			{
				column_data[aa] = std::make_shared<std::vector<double> >();
			}
		}

		if (load_data_from_line)
		{
			std::istringstream istr(curline);

			for (int col_id = 0; col_id < header_n_columns; col_id++)
			{
				double tmp;
				istr >> tmp;
				column_data[col_id]->emplace_back(tmp);
			}
		}
	}
	return column_data.size() > 0;
}