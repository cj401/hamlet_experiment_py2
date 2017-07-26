/* $Id: emission_model.h 21443 2017-06-28 18:13:50Z chuang $ */

#ifndef EMISSION_MODEL_H_
#define EMISSION_MODEL_H_

/*!
 * @file Emission_model.h
 *
 * @author Colin Dawson 
 */

#include "util.h"
#include "noise_model.h"
#include "dynamics_model.h"
#include "state_model.h"
#include <m_cpp/m_vector.h>
#include <m_cpp/m_matrix.h>
#include <boost/shared_ptr.hpp>
#include <string>

struct Emission_parameters;
typedef boost::shared_ptr<Emission_parameters> Emission_param_ptr;

struct Emission_parameters
{
    const Noise_param_ptr np;

    Emission_parameters(const Noise_param_ptr np)
        : np(np)
    {}
        
    virtual ~Emission_parameters() {}
    virtual Emission_model_ptr make_module() const = 0;
};

class Emission_model
{
public:
    /*------------------------------------------------------------
     * TYPEDEFS
     *------------------------------------------------------------*/
    typedef Noise_model::Noisy_data_matrix Noisy_data_matrix;
    typedef Noise_model::Noisy_data_matrix_list Noisy_data_matrix_list;
    typedef Noise_model::Noise_parameters Noise_parameters;
protected:
    /*------------------------------------------------------------
     * CONSTRUCTORS
     *------------------------------------------------------------*/
    Emission_model(const Noise_param_ptr noise_parameters);
      
public:

    /*------------------------------------------------------------
     * DESTRUCTOR
     *------------------------------------------------------------*/
    virtual ~Emission_model() {}

    /*------------------------------------------------------------
     * INITIALIZERS
     *------------------------------------------------------------*/

    void set_parent(HMM_base* const p );
    
    virtual void initialize_resources();

    virtual void initialize_params();
    
    virtual void input_previous_results(const std::string& input_path, const std::string& name);

    // void set_data_size(const size_t& T, const size_t& K) {T_ = T; K_ = K;}
    void set_data_size(const T_list& T, const size_t& K) {T_ = T; K_ = K;}

    //void set_test_T(const size_t& T) {test_T_ = T;}
    void set_test_T(const T_list& T) {test_T_ = T;}
    
    virtual void add_data(const std::string& path, const size_t& num_files);
    
    virtual void add_test_data(const std::string& path, const size_t& num_files);
    
    virtual void generate_data(const std::string& path) = 0;
    
    virtual void generate_observations(const T_list& T, const std::string& path);
    
    virtual void generate_test_observations(const T_list& T, const std::string& path);
    
    /*
    virtual void generate_observations(const size_t& T, const std::string& path);
    virtual void generate_test_observations(const size_t& T, const std::string& path);
     */

    /*------------------------------------------------------------
     * DISPLAY FUNCTIONS
     *------------------------------------------------------------*/
    virtual void set_up_results_log() const = 0;
    virtual void write_state_to_file(const std::string& name) const = 0;
    
    /*------------------------------------------------------------
     * INFERENCE INTERFACE
     *------------------------------------------------------------*/    
    virtual void update_params();

    /*------------------------------------------------------------
     * DIMENSION ACCESSORS
     *------------------------------------------------------------*/
    
    size_t NF() const;
    
    size_t test_NF() const;
    
    const T_list& T() const {return T_;}
    
    size_t T(const size_t& i) const {return T_.at(i);}
    
    /*
    const size_t& T() const {return T_;}
     */
    
    size_t K() const {return K_;}
    
    /*
    size_t test_T() const {return test_T_;}
     */
    const T_list& test_T() const {return test_T_;}
    
    size_t test_T(const size_t& i) const {return test_T_.at(i);}
    
    size_t J() const;

    size_t D() const;
    
    size_t D_prime() const;
    
    /*------------------------------------------------------------
     * ACCESSORS VISIBLE TO OTHER MODULES
     *------------------------------------------------------------*/
    /*
    virtual const Mean_matrix& X_star() const = 0;
     */
    virtual const Mean_matrix_list& X_star() const = 0;
    virtual const Mean_matrix& X_star(const size_t& i) const = 0;
    
    /*
    virtual Mean_matrix build_augmented_mean_matrix_for_d(
        const size_t& first_index,
        const size_t& range_size) = 0;
     */
    
    virtual Mean_matrix build_augmented_mean_matrix_for_d(
        const size_t& i,
        const size_t& first_index,
        const size_t& range_size) = 0;
    
    /*
    virtual Likelihood_matrix get_log_likelihood_matrix(
        const State_matrix&      theta,
        const kjb::Index_range&  times = kjb::Index_range::ALL
        ) const = 0;
     */
    
    virtual Likelihood_matrix get_log_likelihood_matrix(
        const size_t&            i,
        const State_matrix&      theta,
        const kjb::Index_range&  times = kjb::Index_range::ALL
        ) const = 0;
    
    /*
    virtual Likelihood_matrix get_conditional_log_likelihood_matrix(
        const size_t&      d_first,
        const size_t&      range_size
        ) const = 0;
     */
    virtual Likelihood_matrix get_conditional_log_likelihood_matrix(
        const size_t&      i,
        const size_t&      d_first,
        const size_t&      range_size
        ) const = 0;

    virtual Likelihood_matrix get_test_log_likelihood_matrix(
        const size_t& i,
        const State_matrix& theta) const = 0;

    /**
     * @brief access the "noised" data (may not be the actual data)
     */
    
    Noisy_data_matrix_list noisy_data() const {return nm_->noisy_data();}
    
    Noisy_data_matrix noisy_data(const size_t& i) const {return nm_->noisy_data(i);}
    
    /*
    Noisy_data_matrix noisy_data() const {return nm_->noisy_data();}
     */

    /**
     * @brief access a generic noise parameter object
     */
    Noise_parameters noise_parameters() const;

    /*------------------------------------------------------------
     * COMPUTATION NEEDED BY OTHER MODULES
     *------------------------------------------------------------*/
    virtual double log_likelihood_ratio_for_state_change(
        const size_t& i,
        const State_type& theta_current,
        const double& delta,
        const kjb::Index_range& indices,
        const size_t& d = 0
        ) const = 0;

    virtual Prob_vector log_likelihood_ratios_for_state_range(
        const size_t& i,
        const State_type& theta_current,
        const size_t& d,
        const size_t& first_index,
        const size_t& range_size,
        const kjb::Index_range& indices,
        bool  include_new = false
        ) const = 0;
    
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
    //     const size_t& d = 0
    //     ) = 0;

    virtual void insert_latent_dimension(const size_t& d, const size_t& new_pos) = 0;
    virtual void remove_latent_dimension(const size_t& old_pos) = 0;
    virtual void replace_latent_dimension(const size_t& d, const size_t& pos) = 0;

    /*------------------------------------------------------------
     * ACCESS TO OTHER MODULES
     *------------------------------------------------------------*/
    const Time_set& partition_map(const size_t i, const State_indicator& j) const;
    /*
    const Time_set& partition_map(const State_indicator& j) const;
     */

public:
    
    std::string write_path;
    
    /*------------------------------------------------------------
     * VERBOSE VARIABLE
     *------------------------------------------------------------*/
    size_t verbose_;

protected:

    /*------------------------------------------------------------
     * INTERNAL ACCESSORS
     *------------------------------------------------------------*/

    /*
    size_t T_;
     */
    T_list T_;
    size_t K_;
    /*
    size_t test_T_;
     */
    T_list test_T_;
    
    /*------------------------------------------------------------
     * LINKS
     *------------------------------------------------------------*/
    HMM_base* parent;

    const Noise_model_ptr nm_;
};

#endif

