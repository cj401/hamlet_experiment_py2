/* $Id: known_weights.cpp 21443 2017-06-28 18:13:50Z chuang $ */

/*!
 * @file known_weights.cpp
 *
 * @author Colin Dawson 
 */

#include "util.h"
#include "known_weights.h"
#include "parameters.h"
#include <boost/make_shared.hpp>
#include <iostream>

Known_weights_parameters::Known_weights_parameters(
    const Parameters& params,
    const std::string& name
    ) : Base_class(params, name),
        weights_file(params.get_param_as<std::string>(name, "weights_path") + "weights.txt")
{}

Weight_prior_ptr Known_weights_parameters::make_module() const
{
    return boost::make_shared<Known_weights>(this);
}

void Known_weights::initialize_params()
{
    PMWP(verbose_ > 0, "    Using fixed weights from file %s\n", (weights_file.c_str()));
    //std::cerr << "    Using fixed weights from file " << weights_file << std::endl;
    W_.read(weights_file.c_str());
    if(include_bias_)
    {
        b_ = W_.get_row(D_prime());
        W_.resize(D_prime(), K());
    } 
    // std::cerr << "W = " << W_ << std::endl;
}

void Known_weights::generate_data(const std::string& path)
{
    std::cerr << "Writing weights to file " << path + "weights.txt" << std::endl;
    Weight_matrix W_star = W_;
    W_star.vertcat(kjb::create_row_matrix(b_)).write(
        (path + "weights.txt").c_str());
}

void Known_weights::write_state_to_file(const std::string&) const
{
    Weight_matrix W_star = W_;
    if(include_bias_) W_star.vertcat(kjb::create_row_matrix(b_));
    W_star.write((write_path + "W/W.txt").c_str());
}

