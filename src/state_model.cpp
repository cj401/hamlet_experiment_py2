/* $Id: state_model.cpp 21443 2017-06-28 18:13:50Z chuang $ */

/*!
 * @file state_model.cpp
 *
 * @author Colin Dawson 
 */

#include "util.h"
#include "state_model.h"
#include "hdp_hmm_lt.h"
#include <l_cpp/l_int_matrix.h>
#include <boost/lexical_cast.hpp>

State_parameters::State_parameters(
    const Parameters& params,
    const std::string& name
    ) : D(params.get_param_as<size_t>(name, "D")),
        fixed_theta(params.exists(name, "state_matrix_file")),
        theta_file(
            fixed_theta
            ? params.get_param_as<std::string>(name, "state_matrix_file")
                 + ".theta"
            : "")
{}

State_model::State_model(const State_parameters* const params)
    : parent(0), theta_file(params->theta_file),
      theta_(), theta_prime_(), D_(params->D),
      D_prime_(params->D)
{}

void State_model::initialize_resources()
{
    PM(verbose_ > 0, "Allocating resources for state model...\n");
    //std::cerr << "Allocating resources for state model..." << std::endl;
    if(theta_file.empty())
    {
        PM(verbose_ > 0, "    Allocating theta...");
        //std::cerr << "    Allocating theta...";
        theta_ = State_matrix(J(), D_prime(), 0.0);
        PM(verbose_ > 0, "done.\n");
        //std::cerr << "done." << std::endl;
    } else {
        PMWP(verbose_ > 0, "    Using fixed theta matrix from file %s\n", (theta_file.c_str()));
        //std::cerr << "    Using fixed theta matrix from file " << theta_file
        //          << std::endl;
        theta_.read(theta_file.c_str());
        size_t theta_rows = theta_.get_num_rows();
        size_t theta_cols = theta_.get_num_cols();
        IFTD(theta_rows > 0, kjb::IO_error,
             "I/O ERROR: Could not find file %s for theta.",
             (theta_file.c_str())
            );
        IFTD(
            theta_rows == J() && theta_cols == D(),
            kjb::IO_error,
            "I/O ERROR: Reading in a %d by %d state matrix from file %s,"
            " but config file specifies a %d by %d matrix.",
            (theta_file.c_str())(theta_rows)(theta_cols)(J())(D())
            );
    }
}

void State_model::input_previous_results(const std::string& input_path, const std::string& name)
{
    PMWP(verbose_ > 0, "Inputting information from iteration %s for state model...\n", (name.c_str()));
    //std::cerr << "Inputting information from iteration " << name << " for state model..." << std::endl;
    PM(verbose_ > 0, "Inputting information for State matrix theta_...\n");
    //std::cerr << "Inputting information for State matrix theta_..." << std::endl;
    theta_ = State_matrix((input_path + "theta/" + name + ".txt").c_str());
    PM(verbose_ > 0, "done.\n");
    //std::cerr << "done." << std::endl;
}

void State_model::set_parent(HDP_HMM_LT* const p)
{
    parent = p;
    write_path = parent->write_path;
}

void State_model::set_up_results_log() const
{
    create_directory_if_nonexistent(write_path + "theta");
    // create_directory_if_nonexistent(write_path + "phi");
}

Emission_model_ptr State_model::emission_model() const
{
    return parent->emission_model_;
}

Similarity_model_const_ptr State_model::similarity_model() const
{
    return parent->similarity_model_;
}

const State_matrix& State_model::theta_star(const size_t& i) const
{
    return parent->theta_star(i);
}

const State_matrix_list& State_model::theta_star() const
{
    return parent->theta_star();
}

size_t State_model::T(const size_t& i) const
{
    return parent->T(i);
}

const size_t& State_model::NF() const
{
    return parent->NF();
}

/*
const State_matrix& State_model::theta_star() const
{
    return parent->theta_star();
}

const size_t& State_model::T() const {return parent->T_;}
 */

size_t State_model::J() const {return parent->J_;}

const size_t& State_model::D() const {return D_;}
