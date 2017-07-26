/* $Id: factorial_hmm.cpp 21443 2017-06-28 18:13:50Z chuang $ */

/*!
 * @file factorial_hmm.cpp
 *
 * @author Colin Dawson 
 */

#include "util.h"
#include "factorial_hmm.h"
#include "isotropic_exponential_similarity.h"
#include "emission_model.h"
#include <boost/make_shared.hpp>
#include <boost/lexical_cast.hpp>
#include <string>

Factorial_HMM::Factorial_HMM(
    const size_t&                    num_chains,
    const Transition_prior_param_ptr transition_prior_parameters,
    const Dynamics_param_ptr         dynamics_parameters,
    const State_param_ptr            state_parameters,
    const Emission_param_ptr         emission_parameters,
    const std::string&               write_path,
    const size_t                     random_seed
    ) : HMM_base(write_path, emission_parameters->make_module()),
        J_prime_(), cum_J_prime_()
{
    D_ = num_chains;
    kjb::seed_sampling_rand(random_seed);
    initialize_gnu_rng(random_seed);
    Similarity_param_ptr similarity_params =
        boost::make_shared<Isotropic_exponential_similarity_parameters>();
    for (size_t d = 0; d < D_; ++d)
    {
        std::cerr << "    Instantiating chain " << d << " of " << D_ << "...";
        J_prime_.push_back(1);
        cum_J_prime_.push_back(d);
        chains_.push_back(
            HDP_HMM_LT(
                transition_prior_parameters->make_module(),
                dynamics_parameters->make_module(),
                state_parameters->make_module(),
                emission_model_,
                similarity_params->make_module(),
                write_path + boost::lexical_cast<std::string>(d) + "/"
                )
            );
        chains_[d].initialize_parent_links();
        std::cerr << "done." << std::endl;
        std::cerr << "    Chain " << d << "has J = " << chains_[d].J()
                  << " and D = " << chains_[d].D_prime()
                  << std::endl;
    }
}

void Factorial_HMM::generate_test_sequence(
    const size_t&      num_sequences,
    const size_t&      T,
    const std::string& name)
{
    test_NF_ = num_sequences;
    test_T_ = T_list((int) num_sequences, (int) T);
    test_T_.at(0)=T;
    emission_model_->set_test_T(test_T_);
    for(size_t d = 0; d < D(); ++d)
    {
        chains_[d].dynamics_model_->sample_labels_from_prior(test_T_);
    }
    emission_model_->generate_test_observations(test_T_, name + "test_");
    add_test_data(name);
}

void Factorial_HMM::set_up_verbose_level(const size_t verbose)
{
    verbose_ = verbose;
    for (size_t d = 0; d < D(); ++d)
    {
        chains_[d].set_up_verbose_level(verbose);
    }
}

/*
void Factorial_HMM::generate_test_sequence(const size_t & T, const std :: string & name)
{
    T_[0] = T; test_T_ = T;
    emission_model_->set_test_T(T);
    for(size_t d = 0; d < D(); ++d)
    {
        chains_[d].dynamics_model_->sample_labels_from_prior();
    }
    emission_model_->generate_test_observations(T, name + "test_");
    add_test_data(name);
}
 */

void Factorial_HMM::input_previous_results(const std::string& input_path, const std::string& name)
{
    PMWP(verbose_ > 0, "Inputting information from iteration %s...", (name.c_str()));
    //std::cerr << "Inputting information from iteration " << name << " ..." << std::endl;
    for (size_t d = 0; d < D(); d++)
    {
        chains_[d].state_model_->input_previous_results(
            input_path, name + "/" + boost::lexical_cast<std::string>(d));
        chains_[d].similarity_model_->input_previous_results(
            input_path, name + "/" + boost::lexical_cast<std::string>(d));
    }
    // do we need sync theta star differently for different chains...?
    sync_theta_star_();
    emission_model_->input_previous_results(input_path, name);
    for (size_t d = 0; d < D(); d++)
    {
        chains_[d].transition_prior_->input_previous_results(input_path, name);
        chains_[d].sync_transition_matrix_();
        chains_[d].dynamics_model_->input_previous_results(input_path, name);
        chains_[d].sync_partition_();
        chains_[d].transition_prior_->sync_transition_counts();
        chains_[d].transition_prior_->update_auxiliary_data();
    }
}

void Factorial_HMM::resample()
{
    for (size_t d = 0; d < D(); d++)
    {
        PMWP(verbose_ > 0, "Resampling state and similarity for chain %d\n", (d));
        //std::cerr << "Resampling state and similarity for chain " << d
        //          << std::endl;
        chains_[d].resample_state_and_similarity_models();
    }
    sync_theta_star_();
    PM(verbose_ > 0, "Updating emission model parameters...\n");
    //std::cerr << "Updating emission model parameters..." << std::endl;
    emission_model_->update_params();
    PM(verbose_ > 0, "Emission model updated.\n");
    //std::cerr << "Emission model updated." << std::endl;
    // std::cerr << "Size of chains_ is " << chains_.size() << std::endl;
    // std::cerr << "Size of theta_star_ is " << theta_star_.size() << std::endl;
    for(size_t d = 0; d < D(); ++d)
    {
        PMWP(verbose_ > 0, "Updating transition model for chain %d\n", (d));
        //std::cerr << "Updating transition model for chain " << d << std::endl;
        chains_[d].resample_transition_model_stage_one();
        // std::cerr << "   Stage one complete." << std::endl;
        for (size_t i = 0; i < NF(); i++)
        {
            chains_[d].dynamics_model_->update_params(i, conditional_log_likelihood_matrix(i, d));
        }
        // std::cerr << "   Dynamics model complete." << std::endl;
        chains_[d].resample_transition_model_stage_two();
        // std::cerr << "   Stage two complete." << std::endl;
        for (size_t i = 0; i < NF(); i++)
        {
            theta_star_[i].replace(0, cum_J_prime_[d] + J_prime_[d] - 1, chains_[d].theta_star(i));
        }
    }
}

/*
void Factorial_HMM::resample()
{
    for (size_t d = 0; d < D(); d++)
    {
        chains_[d].resample_state_and_similarity_models();
    }
    sync_theta_star_();
    emission_model_->update_params();
    // std::cerr << "Size of chains_ is " << chains_.size() << std::endl;
    // std::cerr << "Size of theta_star_ is " << theta_star_.size() << std::endl;
    for(size_t d = 0; d < D(); ++d)
    {
        std::cerr << "    Updating transition model for chain " << d << std::endl;
        chains_[d].resample_transition_model(conditional_log_likelihood_matrix(d));
        theta_star_.replace(0, cum_J_prime_[d] + J_prime_[d] - 1, chains_[d].theta_star());
    }
}
 */

void Factorial_HMM::set_up_results_log() const
{
    create_directory_if_nonexistent(write_path);
    PM(verbose_ > 0, "Setting up results log file...\n");
    //std::cerr << "Setting up results log files..." << std::endl;
    for(size_t d = 0; d < D(); ++d)
    {
        PMWP(verbose_ > 0, "Setting up results log files for chain %d.\n", (d));
        //std::cerr << "Setting up results log files for chain "
        //          << d << "." << std::endl;
        chains_[d].set_up_results_log();
        PM(verbose_ > 0, "done.\n");
        //std::cerr << "done." << std::endl;
    }
    create_directory_if_nonexistent(write_path + "thetastar");
    PM(verbose_ > 0, "done.\n");
    //std::cerr << "done." << std::endl;
    // std::ofstream ofs;
    // ofs.open(write_path + "train_log_likelihood.txt", std::ofstream::out);
    // ofs << "iteration value" << std::endl;
    // ofs.close();
    // ofs.open(write_path + "test_log_likelihood.txt", std::ofstream::out);
    // ofs << "iteration value" << std::endl;
    // ofs.close();
}

/*
void Factorial_HMM::write_state_to_file(const std :: string & name) const
{
    for(size_t d = 0; d < D(); ++d)
    {
        chains_[d].write_state_and_similarity_states_to_file(
            name + "/" + std::to_string(d));
        chains_[d].write_transition_model_state_to_file(
            name + "/" + std::to_string(d));
    }
    emission_model_->write_state_to_file(name);
    theta_star().floor().write(
        (write_path + "thetastar/" + name + ".txt").c_str());
    // Base_class::write_state_to_file(name);
}
 */

void Factorial_HMM::write_state_to_file(const std :: string & name) const
{
    for(size_t d = 0; d < D(); ++d)
    {
        /*
        chains_[d].write_state_and_similarity_states_to_file(
            name + "/" + boost::lexical_cast<std::string>(d));
        chains_[d].write_transition_model_state_to_file(
            name + "/" + boost::lexical_cast<std::string>(d));
         */
        chains_[d].write_state_and_similarity_states_to_file(name);
        chains_[d].write_transition_model_state_to_file(name);
    }
    emission_model_->write_state_to_file(name);
    if (NF() == 1)
    {
        theta_star(0).floor().write(
            (write_path + "thetastar/" + name + ".txt").c_str());
    }
    else
    {
        create_directory_if_nonexistent((write_path + "thetastar/" + name).c_str());
        boost::format fmt("%03i");
        for (size_t i = 0; i < NF(); i++)
        {
            std::string file_name = (fmt % (i+1)).str() + ".txt";
            theta_star(i).floor().write(
                (write_path + "thetastar/" + name + "/" + file_name).c_str());
        }
    }
    // Base_class::write_state_to_file(name);
}

size_t Factorial_HMM::D_prime() const
{
    return cum_J_prime_[D()-1] + J_prime_[D() - 1];
}

void Factorial_HMM::initialize_parent_links()
{
    PM(verbose_ > 0, "    Linking emission model...");
    //std::cerr << "    Linking emission model...";
    emission_model_->set_parent(this);
    PM(verbose_ > 0, "done.\n");
    //std::cerr << "done." << std::endl;
}

void Factorial_HMM::initialize_resources_()
{
    PM(verbose_ > 0, "Initializing resrouces fro top-lvel model...\n");
    //std::cerr << "Initializing resources for top-level model..." << std::endl;
    for(size_t d = 0; d < D(); ++d)
    {
        PMWP(verbose_ > 0, "    Allocating resources for chain %d...", (d));
        //std::cerr << "    Allocating resources for chain " << d
        // << " of " << D() << " with J = " << chains_[d].J()
        // << " states."
        //<< std::endl;
        // for now, assume the number of values in each chain
        // don't change
        chains_[d].T_ = T_;
        chains_[d].K_ = K_;
        chains_[d].NF_ = NF_;
        chains_[d].initialize_resources_();
        PM(verbose_ > 0, "done.\n");
    }
    Base_class::initialize_resources_();
}

/*
void Factorial_HMM::initialize_resources_()
{
    std::cerr << "Initializing resources for top-level model..." << std::endl;
    for(size_t d = 0; d < D(); ++d)
    {
        std::cerr << "    Allocating resources for chain " << d
                  // << " of " << D() << " with J = " << chains_[d].J()
                  // << " states."
                  << std::endl;
        // for now, assume the number of values in each chain
        // don't change
        chains_[d].T_ = T_;
        chains_[d].K_ = T_;
        chains_[d].initialize_resources_();
    }
    Base_class::initialize_resources_();
}
 */

void Factorial_HMM::initialize_params_()
{
    for(size_t d = 0; d < D(); ++d)
    {
        PMWP(verbose_ > 0, "Initializing params for chain %d of %d\n", (d)(D()));
        //std::cerr << "Initializing params for chain " << d
        //          << "of " << D() << std::endl;
        chains_[d].initialize_state_and_similarity_models();
        chains_[d].initialize_transition_model();
    }
    sync_theta_star_();
    emission_model_->initialize_params();
}

void Factorial_HMM::sync_theta_star_()
{
    PM(verbose_ > 0, "Synchronizing theta*...");
    //std::cerr << "Synchronizing theta*...";
    for(size_t d = 0; d < D(); ++d)
    {
        for (size_t i = 0; i < NF(); ++i)
        {
            theta_star_[i].replace(
                0, cum_J_prime_[d] + J_prime_[d] - 1,
                chains_[d].theta_star(i));
        }
    }
    // std::cerr << "theta_star = " << std::endl;
    // std::cerr << theta_star_ << std::endl;
    PM(verbose_ > 0, "done.\n");
    //std::cerr << "done." << std::endl;
}

/*
void Factorial_HMM::sync_theta_star_()
{
    std::cerr << "Synchronizing theta*...";
    for(size_t d = 0; d < D(); ++d)
    {
        theta_star_.replace(
            0, cum_J_prime_[d] + J_prime_[d] - 1,
            chains_[d].theta_star());
    }
    // std::cerr << "theta_star = " << std::endl;
    // std::cerr << theta_star_ << std::endl;
    std::cerr << "done." << std::endl;
}
 */
