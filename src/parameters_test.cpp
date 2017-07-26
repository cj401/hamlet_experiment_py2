/* $Id: parameters_test.cpp 18824 2015-04-15 15:50:27Z clayton $ */

/*!
 * @file parameters_test.cpp
 *
 * @author Clayton T. Morrison
 */

#include "parameters.h"

#include "probit_noise_model.h"

#include <iostream>
#include <fstream>

int main(int, char **)
{
	std::cout << ">>> parameters_test" << std::endl;

	Parameters params;
	params.output(std::cout);

	std::cout << ">>> test write to file" << std::endl;

	std::ofstream myfile;
  	myfile.open ("parameters/test_parameters_output.txt");
  	params.output(myfile);
  	myfile.close();

  	const std::string test_file = "parameters/parameters_example.config";

  	std::cout << ">>> test read from file: '" << test_file << "'" << std::endl;

  	params.read(test_file, true);

  	std::cout << ">>> test read from file that does not exist" << std::endl;

  	params.read("");

  	std::cout << ">>> test adding a parameter value" << std::endl;

  	std::string size_t_value_string = "10";
  	params.add("experiment", "iterations", size_t_value_string);

  	std::cout << ">>> test getting a parameter value" << std::endl;

  	size_t size_t_value_to_get =
            params.get_param_as<size_t>("experiment", "iterations");
  	std::cout << "size_t_value_to_get = " << size_t_value_to_get << std::endl;

  	std::cout << ">>> test output_parameter_definitions" << std::endl;

    params.output(std::cout);

    std::cout << ">>> test initializing Probit_emission_model from params" << std::endl;

    typedef Probit_noise_model EM;
    Probit_noise_parameters ep(params);

    std::cout << ">>> parameters_test DONE." << std::endl;
}

