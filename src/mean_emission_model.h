/* $Id: mean_emission_model.h 21443 2017-06-28 18:13:50Z chuang $ */

#ifndef MEAN_EMISSION_MODEL_H_
#define MEAN_EMISSION_MODEL_H_

/*!
 * @file mean_emission_model.h
 *
 * @author Colin Dawson 
 */

#include "emission_model.h"

class Mean_emission_model;
struct Mean_emission_parameters;

class Mean_prior;
struct Mean_prior_parameters;

typedef boost::shared_ptr<Mean_prior_parameters> Mean_prior_param_ptr;
typedef boost::shared_ptr<Mean_prior> Mean_prior_ptr;

struct Mean_emission_parameters : public Emission_parameters
{
    const Mean_prior_param_ptr mpp;
    
    Mean_emission_parameters(
        const Mean_prior_param_ptr  mpp,
        const Noise_param_ptr np
        ) : Emission_parameters(np), mpp(mpp)
    {}

    virtual ~Mean_emission_parameters() {}

    virtual Emission_model_ptr make_module() const;
};

class Mean_emission_model : public Emission_model
{
public:
    /*------------------------------------------------------------
     * TYPEDEFS
     *------------------------------------------------------------*/
    
    typedef Mean_emission_model Self;
    typedef Emission_model Base_class;

    /*------------------------------------------------------------
     * CONSTRUCTORS
     *------------------------------------------------------------*/
    Mean_emission_model(
        const Mean_prior_param_ptr      mean_parameters,
        const Noise_param_ptr     noise_parameters
        );
    
    /*------------------------------------------------------------
     * DESTRUCTOR
     *------------------------------------------------------------*/
    virtual ~Mean_emission_model()
    {}

    /*------------------------------------------------------------
     * INITIALIZERS
     *------------------------------------------------------------*/
    virtual void initialize_resources();
    
    virtual void initialize_params();
    
    virtual void input_previous_results(const std::string& input_path, const std::string& name);

    virtual void generate_data(const std::string& path);
    
    /*------------------------------------------------------------
     * DISPLAY FUNCTIONS
     *------------------------------------------------------------*/
    virtual void set_up_results_log() const;

    virtual void write_state_to_file(const std::string& name) const;
    
    /*------------------------------------------------------------
     * INFERENCE INTERFACE
     *------------------------------------------------------------*/
    virtual void update_params();
    
    /*------------------------------------------------------------
     * ACCESSORS VISIBLE TO OTHER MODULES
     *------------------------------------------------------------*/

    /**
     * @brief return the TxJ log likelihood matrix
     */
    /*
    virtual Likelihood_matrix get_log_likelihood_matrix(
        const State_matrix&     theta,
        const kjb::Index_range& times = kjb::Index_range::ALL
        ) const;
     */
    
    virtual Likelihood_matrix get_log_likelihood_matrix(
        const size_t&           i,
        const State_matrix&     theta,
        const kjb::Index_range& times = kjb::Index_range::ALL
        ) const;

    /**
     * @brief return the TxJ log likelihood matrix for the test set
     */
    virtual Likelihood_matrix get_test_log_likelihood_matrix(
        const size_t& i,
        const State_matrix& theta) const;
    
    /**
     * @brief return the T x `range_size` log likelihood matrix for each of
     *        `range_size` possible changes to the current means.
     *
     * @param d_first the index of the first row of the weight matrix to consider
     * @param range_size the number of columns to consider
     * @returns a T x `range_size` matrix where each row represents a time step
     *          and each column represents a log likelihood for corresponding state change
     */
    /*
    virtual Likelihood_matrix get_conditional_log_likelihood_matrix(
        const size_t&      d_first,
        const size_t&      range_size
        ) const;
     */
    virtual Likelihood_matrix get_conditional_log_likelihood_matrix(
        const size_t&      i,
        const size_t&      d_first,
        const size_t&      range_size
        ) const;

    virtual Mean_matrix build_augmented_mean_matrix_for_d(
        const size_t& i,
        const size_t& first_index,
        const size_t& range_size
        );
    
    /*
     virtual Mean_matrix build_augmented_mean_matrix_for_d(
        const size_t& first_index,
        const size_t& range_size);
     */
    
    /*------------------------------------------------------------
     * CALCULATIONS NEEDED BY OTHER MODULES
     *------------------------------------------------------------*/
    virtual double log_likelihood_ratio_for_state_change(
        const size_t& i,
        const State_type& theta_current,
        const double& delta,
        const kjb::Index_range& indices,
        const size_t& d
        ) const;

    virtual Prob_vector log_likelihood_ratios_for_state_range(
        const size_t& i,
        const State_type& theta_current,
        const size_t& d,
        const size_t& first_index,
        const size_t& range_size,
        const kjb::Index_range& indices,
        bool include_new = false
        ) const;
    
    /**
     * @brief Get X matrix with one row per state
     */
    const Mean_matrix& X() const;
    
    /**
     * @brief Get augmented X matrix with one row per time step
     */

    const Mean_matrix_list& X_star() const;
    const Mean_matrix& X_star(const size_t& i) const;
    
    /*
     const Mean_matrix& X_star() const;
     */
    
    virtual void insert_latent_dimension(const size_t& d, const size_t& new_pos);
    virtual void remove_latent_dimension(const size_t& old_pos);
    virtual void replace_latent_dimension(const size_t& d, const size_t& pos);
    
    /*------------------------------------------------------------
     * VERBOSE SET UP FUNCTIONS
     *------------------------------------------------------------*/
    virtual void set_up_verbose_level(const size_t verbose);

protected:

    /*------------------------------------------------------------
     * INTERNAL UPDATE FUNCTIONS
     *------------------------------------------------------------*/
    void sync_means_(const size_t& i) const;
    
    void sync_means_() const;
    
    /*------------------------------------------------------------
     * LINK VARIABLES
     *------------------------------------------------------------*/
    const Mean_prior_ptr mm_;

    /*------------------------------------------------------------
     * PARAMETER VALUES
     *------------------------------------------------------------*/

    mutable Mean_matrix_list X_star_;
    /*
    mutable Mean_matrix X_star_;
     */
};

#endif

