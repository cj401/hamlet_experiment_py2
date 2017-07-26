/* $Id: mean_emission_model.cpp 21443 2017-06-28 18:13:50Z chuang $ */

/*!
 * @file mean_emission_model.cpp
 *
 * @author Colin Dawson 
 */

#include "mean_emission_model.h"
#include "mean_prior.h"
#include <boost/make_shared.hpp>

Emission_model_ptr Mean_emission_parameters::make_module() const
{
    return boost::make_shared<Mean_emission_model>(mpp, np);
}

Mean_emission_model::Mean_emission_model(
    const Mean_prior_param_ptr mean_parameters,
    const Noise_param_ptr noise_parameters
    ) : Emission_model(noise_parameters),
        mm_(mean_parameters->make_module())
{}

void Mean_emission_model::set_up_verbose_level(const size_t verbose)
{
    Base_class::set_up_verbose_level(verbose);
    mm_->set_up_verbose_level(verbose);
}

void Mean_emission_model::initialize_resources()
{
    PM(verbose_ > 0, "Allocating resources for emission model...\n");
    //std::cerr << "Allocating resources for emission model..." << std::endl;
    Base_class::initialize_resources();
    mm_->set_parent(this);
    mm_->initialize_resources();
    PM(verbose_ > 0, "    Allocating X*...");
    //std::cerr << "    Allocating X*...";
    X_star_ = Mean_matrix_list(NF());
    for (int i = 0; i < NF(); i++)
    {
        X_star_.at(i) = Mean_matrix(T(i), K(), 0.0);
    }
}

/*
 void Mean_emission_model::initialize_resources()
{
 std::cerr << "Allocating resources for emission model..." << std::endl;
 Base_class::initialize_resources();
 mm_->set_parent(this);
 mm_-
 >initialize_resources();
 std::cerr << "    Allocating X*...";
 X_star_ = Mean_matrix(T(), K(), 0.0);
}
 */

void Mean_emission_model::initialize_params()
{
    PM(verbose_ > 0, "Initializing mean emission parameters...\n");
    //std::cerr << "Initializing mean emission parameters..." << std::endl;
    mm_->initialize_params();
    Base_class::initialize_params();
}

void Mean_emission_model::input_previous_results(const std::string& input_path, const std::string& name)
{
    PM(verbose_ > 0, "Inputting information for mean emission model...\n");
    //std::cerr << "Inputting information for mean emission model..." << std::endl;
    mm_->input_previous_results(input_path, name);
    Base_class::input_previous_results(input_path, name);
}

void Mean_emission_model::generate_data(const std::string& path)
{
    mm_->generate_data(path);
    nm_->generate_data(path);
    generate_observations(T_, path);
}

void Mean_emission_model::set_up_results_log() const
{
    mm_->set_up_results_log();
    Base_class::set_up_results_log();
}

void Mean_emission_model::write_state_to_file(const std::string & name) const
{
    PM(verbose_ > 0, "    Writing state means to file...\n");
    //std::cerr << "    Writing state means to file..." << std::endl;
    mm_->write_state_to_file(name);
    PM(verbose_ > 0, "    Writing noise parameters to file...\n");
    //std::cerr << "    Writing noise parameters to file..." << std::endl;
    Base_class::write_state_to_file(name);
    PM(verbose_ > 0, "    Done writing emission params to file.\n");
    //std::cerr << "    Done writing emission params to file." << std::endl;
}

void Mean_emission_model::update_params()
{
    PM(verbose_ > 0, "Updating emission parameters...\n");
    //std::cerr << "Updating emission parameters..." << std::endl;
    Base_class::update_params();
    mm_->update_params();
}

/*
Likelihood_matrix Mean_emission_model::get_log_likelihood_matrix(
    const State_matrix&,
    const kjb::Index_range& times
    ) const
{
    return nm_->get_log_likelihood_matrix(X(), times);
}
 */

Likelihood_matrix Mean_emission_model::get_log_likelihood_matrix(
    const size_t&           i,
    const State_matrix&,
    const kjb::Index_range& times
    ) const
{
    return nm_->get_log_likelihood_matrix(i, X(), times);
}

Likelihood_matrix Mean_emission_model::get_test_log_likelihood_matrix(
    const size_t& i, 
    const State_matrix&) const
{
    return nm_->get_test_log_likelihood_matrix(i, X());
}

Likelihood_matrix Mean_emission_model::get_conditional_log_likelihood_matrix(
    const size_t&,
    const size_t&,
    const size_t&
    ) const
{
    IFT(false, kjb::Not_implemented,
        "Conditional log likelihood not implemented for means only emission model");
}

/*
Likelihood_matrix Mean_emission_model::get_conditional_log_likelihood_matrix(
    const size_t&,
    const size_t&
    ) const
{
    IFT(false, kjb::Not_implemented,
        "Conditional log likelihood not implemented for means only emission model");
}
 */

double Mean_emission_model::log_likelihood_ratio_for_state_change(
    const size_t&, 
    const State_type&,       
    const double&,           
    const kjb::Index_range&,
    const size_t&          
    ) const
{
    IFT(false, kjb::Not_implemented,
        "Likelihood ratio not implemented for means only emission model");
}

Prob_vector Mean_emission_model::log_likelihood_ratios_for_state_range(
    const size_t&, 
    const State_type&,
    const size_t&,
    const size_t&,
    const size_t&,
    const kjb::Index_range&,
    bool
    ) const
{
    IFT(false, kjb::Not_implemented,
        "Range likelihood ratio not implemented for means only emission model");
}

Mean_matrix Mean_emission_model::build_augmented_mean_matrix_for_d(
    const size_t& i,
    const size_t& first_index,
    const size_t& range_size
    )
{
    sync_means_(i);
    return X_star(i).submatrix(0, first_index, T(i), range_size);
}

/*
Mean_matrix Mean_emission_model::build_augmented_mean_matrix_for_d(
    const size_t& first_index,
    const size_t& range_size
    )
{
    sync_means_();
    return X_star().submatrix(0, first_index, T(), range_size);
}
 */

const Mean_matrix& Mean_emission_model::X() const {return mm_->X();}

/*
const Mean_matrix& Mean_emission_model::X_star() const
{
    sync_means_();
    return X_star_;
}
 */

const Mean_matrix_list& Mean_emission_model::X_star() const
{
    sync_means_();
    return X_star_;
}

const Mean_matrix& Mean_emission_model::X_star(const size_t& i) const
{
    sync_means_(i);
    return X_star_.at(i);
}

void Mean_emission_model::sync_means_() const
{
    for (size_t i = 0; i < NF(); i++)
    {
        for (int j = 0; j < X().get_num_rows(); ++j)
        {
            Time_set times = partition_map(i,j);
            Mean_vector x = X().get_row(j);
            for(Time_set::const_iterator it = times.begin();
                it != times.end(); ++it)
            {
                X_star_.at(i).set_row((*it), x);
            }
        }
    }
}

void Mean_emission_model::sync_means_(const size_t& i) const
{
    for (int j = 0; j < X().get_num_rows(); ++j)
    {
        Time_set times = partition_map(i,j);
        Mean_vector x = X().get_row(j);
        for(Time_set::const_iterator it = times.begin();
            it != times.end(); ++it)
        {
            X_star_.at(i).set_row((*it), x);
        }
    }
}

/*
void Mean_emission_model::sync_means_() const
{
    for(int j = 0; j < X().get_num_rows(); ++j)
    {
        Time_set times = partition_map(j);
        Mean_vector x = X().get_row(j);
        for(Time_set::const_iterator it = times.begin();
            it != times.end(); ++it)
        {
            X_star_.set_row((*it), x);
        }
    }
}
 */

void Mean_emission_model::insert_latent_dimension(const size_t&, const size_t&)
{
    IFT(false, kjb::Not_implemented,
         "Attempting to insert a dimension"
         " into a means-only emission model."
         " This is not supported");
}

void Mean_emission_model::remove_latent_dimension(const size_t&)
{
    IFT(false, kjb::Not_implemented,
         "Attempting to remove a dimension"
         " into a means-only emission model."
         " This is not supported");
}

void Mean_emission_model::replace_latent_dimension(const size_t&, const size_t&)
{
    IFT(false, kjb::Not_implemented,
         "Attempting to replace a dimension"
         " into a means-only emission model."
         " This is not supported");
}
