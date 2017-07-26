/* $Id: hdp_hmm_lt.h 21443 2017-06-28 18:13:50Z chuang $ */

#ifndef HDP_HMM_LT_H_
#define HDP_HMM_LT_H_

/*!
 * @file hdp-hmm-lt.h
 *
 * @author Colin Dawson 
 */
#include "hmm_base.h"
#include "transition_prior.h"
#include "dynamics_model.h"
#include "emission_model.h"
#include "state_model.h"
#include "similarity_model.h"
#include "util.h"
#include <boost/make_shared.hpp>

class HDP_HMM_LT : public HMM_base
{
public:
    typedef HMM_base Base_class;
public:
    friend class Transition_prior;
    friend class Dynamics_model;
    friend class State_model;
    friend class Emission_model;
    friend class Similarity_model;
    friend class Factorial_HMM;
public:
    /*------------------------------------------------------------
     * CONSTRUCTOR
     *------------------------------------------------------------*/
    HDP_HMM_LT(
        const Transition_prior_param_ptr   transition_prior_parameters,
        const Dynamics_param_ptr           dynamics_parameters,
        const State_param_ptr              state_parameters,
        const Emission_param_ptr           emission_parameters,
        const Similarity_param_ptr         similarity_parameters,
        const std::string&                 write_path,
        const size_t                       random_seed = kjb::DEFAULT_SEED
        );

    /**
     * @brief Use this ctor if you have pre-constructed modules
     */
    HDP_HMM_LT(
        const Transition_prior_ptr transition_prior,
        const Dynamics_model_ptr dynamics_model,
        const State_model_ptr state_model,
        const Emission_model_ptr emission_model,
        const Similarity_model_ptr similarity_model,
        const std::string& write_path
        );

    /*------------------------------------------------------------
     * DESTRUCTOR
     *------------------------------------------------------------*/
    ~HDP_HMM_LT();

    /*------------------------------------------------------------
     * PUBLIC INTERFACE
     *------------------------------------------------------------*/
    virtual void generate_test_sequence(
        const size_t&      num_sequences,
        const size_t&      T,
        const std::string& name
        );
    
    /**
     * @brief Perform one complete Gibbs sampling iteration
     */
    void resample();

    void set_up_results_log() const;
    
    void input_previous_results(const std::string& input_path, const std::string& name);

    /**
     * @brief record the current state to a set of files
     */ 
    void write_state_to_file(const std::string& path) const;

    void add_ground_truth_eval_header() const
    {
        state_model_->add_ground_truth_eval_header();
    }

    virtual void compare_state_sequence_to_ground_truth(
        const std::string& ground_truth_path,
        const std::string& name
        ) const
    {
        state_model_->compare_state_sequence_to_ground_truth(
            ground_truth_path, name);
    }

    /*------------------------------------------------------------
     * DIMENSION ACCESSORS
     *------------------------------------------------------------*/
    size_t D_prime() const;
    const size_t& J() const {return J_;}
    //const size_t& test_T() const {return test_T_;}
    const T_list& test_T() const {return test_T_;}
    size_t test_T(const size_t& i) const {return test_T_.at(i);}

    /*------------------------------------------------------------
     * ACCESSORS TO  CACHE VARIABLES
     *------------------------------------------------------------*/
    virtual const Time_set& partition_map(const size_t& i, const State_indicator& j) const
    {
        return partition_map_list_.at(i).at(j);
    }
    /*
    virtual const Time_set& partition_map(const State_indicator& j) const
    {
        return partition_map_.at(j);
    }
    */

    virtual size_t dimension_map(const size_t& dprime) const
    {
        return state_model_->dimension_map(dprime);
    }
    
    /*------------------------------------------------------------
     * ACCESSORS TO SHARED VARIABLES
     *------------------------------------------------------------*/

    const Prob_matrix& A() const {return A_;}
    const Prob& A(const size_t& j, const size_t& jp) const {return A_.at(j,jp);}

    /*------------------------------------------------------------
     * ACCESSORS TO MODULE VARIABLES
     *------------------------------------------------------------*/
    const Prob_matrix& pi() const {return transition_prior_->pi();}
    const Prob_vector& pi0() const {return transition_prior_->pi0();}
    const Prob& pi0(const size_t& j) const {return transition_prior_->pi0(j);}
    const Count_matrix& N() const {return transition_prior_->N();}
    const Count_matrix& Q() const {return transition_prior_->Q();}
    Count N(const size_t& j, const size_t& jp) const {return transition_prior_->N(j,jp);}
    Count Q(const size_t& j, const size_t& jp) const {return transition_prior_->Q(j,jp);}
    const Count& n_dot(const size_t& j) const {return transition_prior_->n_dot(j);}
    const State_sequence_list& z() const {return dynamics_model_->z();}
    const State_sequence& z(const size_t& i) const {return dynamics_model_->z(i);}
    const State_indicator& z(const size_t& i, const size_t& t) const {return dynamics_model_->z(i,t);}
    /*
    const State_sequence& z() const {return dynamics_model_->z();}
    const State_indicator& z(const size_t& t) const {return dynamics_model_->z(t);}
     */
    const State_matrix& theta() const {return state_model_->theta();}
    const State_matrix& theta_prime() const {return state_model_->theta_prime();}
    State_type theta_prime(const size_t& j) const {return state_model_->theta_prime(j);}
    Coordinate theta(const size_t& j, const size_t& d) const {return state_model_->theta(j,d);}
    const Prob_matrix& Phi() const {return similarity_model_->Phi();}
    const Prob& Phi(const size_t& j, const size_t& jp) const
    {
        return similarity_model_->Phi(j,jp);
    }
    Likelihood_matrix log_likelihood_matrix(
        const size_t& i,
        const kjb::Index_range& states = kjb::Index_range::ALL,
        const kjb::Index_range& times = kjb::Index_range::ALL
        ) const
    {
        return emission_model_->get_log_likelihood_matrix(
                i, theta_prime()(states, kjb::Index_range::ALL), times);
    }
    Likelihood_matrix conditional_log_likelihood_matrix(const size_t& i, const size_t& d) const
    {
        const size_t d_first = state_model_->first_theta_prime_col_for_theta_d(d);
        const size_t range_size = state_model_->num_theta_prime_cols_for_theta_d(d);
        return emission_model_->get_conditional_log_likelihood_matrix(
            i, d_first, range_size);
    }
    
    Likelihood_matrix test_log_likelihood_matrix(const size_t& i) const
    {
        return emission_model_->get_test_log_likelihood_matrix(i, theta_prime());
    }
    
    /*
    Likelihood_matrix log_likelihood_matrix(
        const kjb::Index_range& states = kjb::Index_range::ALL,
        const kjb::Index_range& times = kjb::Index_range::ALL
        ) const
    {
        return emission_model_->get_log_likelihood_matrix(
            theta_prime()(states, kjb::Index_range::ALL), times);
    }
     
    Likelihood_matrix conditional_log_likelihood_matrix(const size_t& d) const
    {
        const size_t d_first = state_model_->first_theta_prime_col_for_theta_d(d);
        const size_t range_size = state_model_->num_theta_prime_cols_for_theta_d(d);
        return emission_model_->get_conditional_log_likelihood_matrix(
            d_first, range_size);
    }
    Likelihood_matrix test_log_likelihood_matrix() const
    {
        return emission_model_->get_test_log_likelihood_matrix(theta_prime());
    }
     */

    double log_likelihood_ratio_for_state_change(
        const size_t& j,
        const double& delta,
        const size_t& d
        ) const;

    Prob_vector log_likelihood_ratios_for_state_range(
        const size_t& j,
        const size_t& d,
        const size_t& first_index,
        const size_t& range_size,
        bool include_new = false
        ) const;
    
    /*------------------------------------------------------------
     * VERBOSE SET UP FUNCTIONS
     *------------------------------------------------------------*/
    virtual void set_up_verbose_level(const size_t verbose);
    
private:
    /*------------------------------------------------------------
     * LVALUE ACCESSORS
     *------------------------------------------------------------*/
    Prob& A(const size_t& j, const size_t& jp) {return A_.at(j,jp);}

    /*------------------------------------------------------------
     * INTERNAL INFERENCE FUNCTIONS
     *------------------------------------------------------------*/
public:
    virtual void initialize_parent_links();
    void write_state_and_similarity_states_to_file(
        const std::string& name
        ) const;

    void write_transition_model_state_to_file(
        const std::string& name
        ) const;

    void compute_and_record_marginal_likelihoods(
        const std::string& name
        ) const;
    
private:
    virtual void initialize_resources_();
    virtual void initialize_params_();
    virtual void initialize_state_and_similarity_models();
    virtual void initialize_transition_model();
    virtual void initialize_partition_();
    virtual void sync_partition_();
    virtual void sync_theta_star_();
    virtual void sync_transition_matrix_();
    virtual void resample_state_and_similarity_models();
    virtual void resample_transition_model_stage_one();
    virtual void resample_transition_model_stage_two();
    /*
    virtual void resample_transition_model(const Prob_matrix& log_likelihood_matrix);
     */
    
private:
    /*------------------------------------------------------------
     * DIMENSION VARIABLES
     *------------------------------------------------------------*/
    size_t J_;
    
    /*------------------------------------------------------------
     * LINKS TO COMPONENT MODULES
     *------------------------------------------------------------*/
    Transition_prior_ptr transition_prior_;
    Dynamics_model_ptr dynamics_model_;
    State_model_ptr state_model_;
    Similarity_model_ptr similarity_model_;

    /*------------------------------------------------------------
     * SHARED VARIABLES
     *------------------------------------------------------------*/
    Prob_matrix A_;
    Partition_map_list partition_map_list_;
    /*
    Partition_map partition_map_;
     */
};

#endif
