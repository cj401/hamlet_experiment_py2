/* $Id: factorial_hmm.h 21443 2017-06-28 18:13:50Z chuang $ */

#ifndef FACTORIAL_HMM_H_
#define FACTORIAL_HMM_H_

/*!
 * @file factorial_hmm.h
 *
 * @author Colin Dawson 
 */

#include "hmm_base.h"
#include "hdp_hmm_lt.h"
#include "transition_prior.h"
#include "dynamics_model.h"
#include "emission_model.h"
#include "state_model.h"
#include "util.h"
#include <boost/make_shared.hpp>
#include <vector>

class Factorial_HMM : public HMM_base
{
public:
    typedef HMM_base Base_class;
public:
    friend class HDP_HMM_LT;
    friend class Transition_prior;
    friend class Dynamics_model;
    friend class State_model;
    friend class Emission_model;
public:
    /*------------------------------------------------------------
     * CONSTRUCTOR
     *------------------------------------------------------------*/
    Factorial_HMM(
        const size_t&                      num_chains,
        const Transition_prior_param_ptr   transition_prior_parameters,
        const Dynamics_param_ptr           dynamics_parameters,
        const State_param_ptr              state_parameters,
        const Emission_param_ptr           emission_parameters,
        const std::string&                 write_path,
        const size_t                       random_seed = kjb::DEFAULT_SEED
        );

    /*------------------------------------------------------------
     * DESTRUCTOR
     *------------------------------------------------------------*/
    ~Factorial_HMM() {}

    /*------------------------------------------------------------
     * PUBLIC INTERFACE
     *------------------------------------------------------------*/
    void generate_test_sequence(
        const size_t&      num_sequences,
        const size_t&      T,
        const std::string& name
        );

    /**
     * @brief Perform one complete Gibbs sampling iteration
     */
    void resample();

    void set_up_results_log() const;

    /**
     * @brief record the current state to a set of files
     */ 
    void write_state_to_file(const std::string& name) const;

    void add_ground_truth_eval_header() const
    {
        //TODO: generalize this (need not imply binary)
        add_binary_gt_eval_header(write_path);
    }

    void compare_state_sequence_to_ground_truth(
        const std::string& ground_truth_path,
        const std::string& name
        ) const
    {
        //TODO: generalize this (need not imply binary)
        score_binary_states(write_path, ground_truth_path, name, theta_star());
    }
    /*
    Likelihood_matrix conditional_log_likelihood_matrix(const size_t& d) const
    {
        // std::cerr << "Size of cum_J_prime and J_prime_ are "
        //           << cum_J_prime_.size() << " and " << J_prime_.size()
        //           << std::endl;
        // std::cerr << "cum_J_prime = " << cum_J_prime_ << std::endl;
        const size_t d_first = cum_J_prime_[d];
        const size_t range_size = J_prime_[d];
        // std::cerr << "range_size is " << range_size << std::endl;
        return emission_model_->get_conditional_log_likelihood_matrix(
            d_first, range_size);
    }
     */
    Likelihood_matrix conditional_log_likelihood_matrix(const size_t& i, const size_t& d) const
    {
        // std::cerr << "Size of cum_J_prime and J_prime_ are "
        //           << cum_J_prime_.size() << " and " << J_prime_.size()
        //           << std::endl;
        // std::cerr << "cum_J_prime = " << cum_J_prime_ << std::endl;
        const size_t d_first = cum_J_prime_[d];
        const size_t range_size = J_prime_[d];
        // std::cerr << "range_size is " << range_size << std::endl;
        return emission_model_->get_conditional_log_likelihood_matrix(
            i, d_first, range_size);
    }

    /*------------------------------------------------------------
     * DIMENSION ACCESSORS
     *------------------------------------------------------------*/
    size_t D_prime() const;
    
    /*------------------------------------------------------------
     * INTERNAL INFERENCE FUNCTIONS
     *------------------------------------------------------------*/
    virtual void initialize_parent_links();
    virtual void initialize_resources_();
    void initialize_params_();
    // void initialize_partition_();
    void sync_theta_star_();
    void input_previous_results(const std::string& input_path, const std::string& name);
    
    /*------------------------------------------------------------
     * VERBOSE SET UP FUNCTIONS
     *------------------------------------------------------------*/
    virtual void set_up_verbose_level(const size_t verbose);
    
private:
    /*------------------------------------------------------------
     * DIMENSION VARIABLES
     *------------------------------------------------------------*/
    std::vector<size_t> J_prime_;
    std::vector<size_t> cum_J_prime_;
    
    /*------------------------------------------------------------
     * LINKS TO COMPONENT MODULES
     *------------------------------------------------------------*/
    std::vector<HDP_HMM_LT> chains_;
    Dynamics_model_ptr dynamics_model_;
    
    /*------------------------------------------------------------
     * VERBOSE VARIABLES
     *------------------------------------------------------------*/
    size_t verbose_;
};

#endif
