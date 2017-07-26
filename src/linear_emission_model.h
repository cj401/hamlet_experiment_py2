/* $Id: linear_emission_model.h 21468 2017-07-11 19:35:05Z cdawson $ */

#ifndef LINEAR_EMISSION_MODEL_H_
#define LINEAR_EMISSION_MODEL_H_

/*!
 * @file linear_emission_model.h
 *
 * @author Colin Dawson 
 */

#include "emission_model.h"
#include "weight_prior.h"
#include <m_cpp/m_vector.h>

class Linear_emission_parameters : public Emission_parameters
{
public:
    const Weight_param_ptr wp;

    Linear_emission_parameters(
        const Weight_param_ptr wp,
        const Noise_param_ptr np
        ) : Emission_parameters(np), wp(wp)
    {}

    virtual ~Linear_emission_parameters() {}

    virtual Emission_model_ptr make_module() const;
};

class Linear_emission_model : public Emission_model
{
public:
    /*------------------------------------------------------------
     * TYPEDEFS
     *------------------------------------------------------------*/
    typedef Linear_emission_model Self;
    typedef Emission_model Base_class;
    typedef Weight_prior::Weight_matrix Weight_matrix;
    typedef double Coordinate;
    friend class Weight_prior;
public:
    /*------------------------------------------------------------
     * CONSTRUCTORS
     *------------------------------------------------------------*/
    Linear_emission_model(
        const Weight_param_ptr    weight_parameters,
        const Noise_param_ptr     noise_parameters
        );
    
    /*------------------------------------------------------------
     * DESTRUCTOR
     *------------------------------------------------------------*/
    virtual ~Linear_emission_model()
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
    
    /**
     * @brief access the augmented state matrix for data sequence i
     */
    const State_matrix& theta_star(const size_t& i) const;

    /**
     * @brief access the list of augmented state matrices
     */
    const State_matrix_list& theta_star() const;

    /**
     * @brief access the augmented state matrix for sequence i as lvalue
     */
    State_matrix& theta_star(const size_t& i);
    
    /**
     * @brief access the list of augmented state matrices as lvalue
     */
    State_matrix_list& theta_star();
    
    bool includes_reference_state() const {return wp_->includes_bias();}
    
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
     * @brief Get augmented X matrix with one row per time step
     */
    
    /*
    const Mean_matrix_& X_star() const
    {
        // X_star_ = wp_->get_mean_matrix(theta_star());
        return X_star_;
    }
     */
    
    const Mean_matrix_list& X_star() const
    {
        // X_star_ = wp_->get_mean_matrix(theta_star());
        return X_star_;
    }
    
    const Mean_matrix& X_star(const size_t& i) const
    {
        return X_star_.at(i);
    }
    
    /*------------------------------------------------------------
     * VERBOSE SET UP FUNCTIONS
     *------------------------------------------------------------*/
    virtual void set_up_verbose_level(const size_t verbose);
    
    /*------------------------------------------------------------
     * MANIPULATORS
     *------------------------------------------------------------*/
    // virtual void shift_mean_for_state_j(
    //     const size_t& j,
    //     const double& delta,
    //     const size_t& d
    //     )
    // {
    //     Mean_vector old_mean = X_.get_row(j);
    //     Mean_vector delta_mean = W().get_row(d) * delta;
    //     X_.set_row(j, old_mean + delta_mean);
    // }

    /*
    virtual Mean_matrix build_augmented_mean_matrix_for_d(
        const size_t& first_index,
        const size_t& range_size
        );
     */
    virtual Mean_matrix build_augmented_mean_matrix_for_d(
        const size_t& i,
        const size_t& first_index,
        const size_t& range_size
        );

    virtual size_t dimension_map(const size_t& dprime) const;

    virtual void insert_latent_dimension(const size_t& d, const size_t& new_pos);
    virtual void remove_latent_dimension(const size_t& old_pos);
    virtual void replace_latent_dimension(const size_t& d, const size_t& pos);

protected:
    /*------------------------------------------------------------
     * INTERNAL ACCESSORS
     *------------------------------------------------------------*/

    // /**
    //  * @brief access the current mean matrix
    //  */
    // const Mean_matrix& X() const {return X_;}

    // /**
    //  * @brief access a particular element of the current mean matrix
    //  */
    // const double& X(const size_t& j, const size_t& k) const
    // {
    //     return X_.at(j,k);
    // }

    /**
     * @brief access the current weight matrix
     */
    const Weight_matrix& W() const {return wp_->W_;}

    /*------------------------------------------------------------
     * LVALUE ACCESSORS
     *------------------------------------------------------------*/

    // /**
    //  * @brief access an lvalue reference to the current mean matrix
    //  */
    // Mean_matrix& X() {return X_;}
    
    // /**
    //  * @brief access an lvalue for a particular element of the current mean matrix
    //  */
    // double& X(const size_t& j, const size_t& k)
    // {
    //     return X_.at(j,k);
    // }

    /*------------------------------------------------------------
     * ACCESS TO OTHER MODULES
     *------------------------------------------------------------*/
    /**
     * @brief access the state matrix
     */
    // const State_matrix& theta_prime() const;
    
    /*------------------------------------------------------------
     * INTERNAL INFERENCE FUNCTIONS
     *------------------------------------------------------------*/

    /**
     * @brief update full mean matrix
     */
    void sync_means_() const
    {
        PM(verbose_ > 0, "    Synchronizing mean matrices...");
        for (size_t i = 0; i < NF(); i++)
        {
            X_star_.at(i) = wp_->get_mean_matrix(theta_star(i));
        }
        PM(verbose_ > 0, "done.");
    }
    /*
    void sync_means_() const
    {
        std::cerr << "    Synchronizing mean matrix...";
        X_star_ = wp_->get_mean_matrix(theta_star());
        // std::cerr << "X* = " << std::endl;
        // std::cerr << X_star_ << std::endl;
        std::cerr << "done." << std::endl;
    }
     */

    /*------------------------------------------------------------
     * STATE VARIABLES
     *------------------------------------------------------------*/
    // mutable Mean_matrix X_;
    mutable Mean_matrix_list X_star_;
    /*
    mutable Mean_matrix X_star_;
     */

    /*------------------------------------------------------------
     * LINK VARIABLES
     *------------------------------------------------------------*/
    const Weight_prior_ptr wp_;
};

#endif

