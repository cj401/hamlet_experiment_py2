/* $Id: emission_model.cpp 21443 2017-06-28 18:13:50Z chuang $ */

/*!
 * @file emission_model.cpp
 *
 * @author Colin Dawson 
 */

#include "util.h"
#include "noise_model.h"
#include "state_model.h"
#include "emission_model.h"
#include "dynamics_model.h"
#include "hdp_hmm_lt.h"
#include <iostream>

Emission_model::Emission_model(
    const Noise_param_ptr  noise_parameters
    ) : parent(0),
        nm_(noise_parameters->make_module())
    {}

void Emission_model::set_parent(HMM_base* const p )
{
    parent = p;
    write_path = parent->write_path;
}

void Emission_model::set_up_verbose_level(const size_t verbose)
{
    verbose_ = verbose;
    nm_->set_up_verbose_level(verbose);
}

void Emission_model::initialize_resources()
{
    nm_->set_parent(this);
    nm_->set_data_size(T_, K_);
    nm_->initialize_resources();
}

void Emission_model::initialize_params()
{
    nm_->initialize_params();
}

void Emission_model::input_previous_results(const std::string& input_path, const std::string& name)
{
    PM(verbose_ > 0, "Inputting information from iteration %s for emission model...\n");
    //std::cerr << "Inputting information from iteration " << name << " for emission model..." << std::endl;
    nm_->input_previous_results(input_path, name);
}

void Emission_model::add_data(const std::string& path, const std::size_t& num_files)
{
    PM(verbose_ > 0, "    Adding data to noise model...\n");
    //std::cerr << "    Adding data to noise model..." << std::endl;
    nm_->add_data(path, num_files);
    set_data_size(nm_->T_, nm_->K_);
};

void Emission_model::add_test_data(const std::string& path, const size_t& num_files)
{
    nm_->add_test_data(path, num_files);
    set_test_T(nm_->test_T_);
};

void Emission_model::generate_observations(const T_list& T, const std::string& path)
{
    nm_->set_data_size(T, K_);
    nm_->generate_data(path);
};

void Emission_model::generate_test_observations(const T_list& T, const std::string& path)
{
    nm_->set_test_data_size(T);
    test_T_ = T;
    nm_->generate_test_data(path);
}

/*
void Emission_model::generate_observations(const size_t& T, const std::string& path)
{
    nm_->set_data_size(T, K_);
    nm_->generate_data(path);
};
 
void Emission_model::generate_test_observations(const size_t& T, const std::string& path)
{
    nm_->set_test_data_size(T);
    test_T_ = T;
    nm_->generate_test_data(path);
};
 */

void Emission_model::set_up_results_log() const
{
    nm_->set_up_results_log();
}

void Emission_model::write_state_to_file(const std::string& name) const
{
    nm_->write_state_to_file(name);
};

void Emission_model::update_params()
{
    nm_->update_params();
}

Emission_model::Noise_parameters Emission_model::noise_parameters() const
{
    return nm_->parameters();
}

size_t Emission_model::J() const {return parent->J();}

size_t Emission_model::D() const {return parent->D();}

size_t Emission_model::D_prime() const {return parent->D_prime();}

size_t Emission_model::NF() const {return parent->NF();}

size_t Emission_model::test_NF() const {return parent->test_NF();}

const Time_set& Emission_model::partition_map(const size_t i, const State_indicator& j) const
{
    return parent->partition_map(i,j);
}

/*
const Time_set& Emission_model::partition_map(const State_indicator& j) const
{
    return parent->partition_map(j);
}
 */

