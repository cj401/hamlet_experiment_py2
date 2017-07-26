/* $Id: state_model.h 21443 2017-06-28 18:13:50Z chuang $ */

#ifndef STATE_MODEL_H_
#define STATE_MODEL_H_

/*!
 * @file State_model.h
 *
 * @author Colin Dawson 
 */

#include "util.h"
#include "parameters.h"
#include "dynamics_model.h"
#include <l_cpp/l_int_vector.h>
#include <l_cpp/l_int_matrix.h>
#include <l_cpp/l_index.h>
#include <m_cpp/m_matrix.h>
#include <m_cpp/m_mat_view.h>
#include <m_cpp/m_vec_view.h>
#include <boost/enable_shared_from_this.hpp>
#include <iosfwd>

class HDP_HMM_LT;
struct State_parameters;
typedef boost::shared_ptr<State_parameters> State_param_ptr;

struct State_parameters 
{
    const size_t D;
    const bool fixed_theta;
    const std::string theta_file;
    
    State_parameters(const Parameters& params, const std::string& name);
    
    virtual ~State_parameters() {}
    virtual State_model_ptr make_module() const = 0;
};

class State_model
{
public:
    /*------------------------------------------------------------
     * TYPEDEFS
     *------------------------------------------------------------*/
    typedef State_matrix::Value_type Coordinate;
public:
    /*------------------------------------------------------------
     * CONSTRUCTORS
     *------------------------------------------------------------*/
    State_model(const State_parameters* const params);

    /*------------------------------------------------------------
     * DESTRUCTOR
     *------------------------------------------------------------*/
    virtual ~State_model(){};
    
    /*------------------------------------------------------------
     * INITIALIZERS
     *------------------------------------------------------------*/

    /**
     * @brief link this module up with a larger model
     */
    void set_parent(HDP_HMM_LT* const p );
    
    /**
     * @brief get the emission module
     */
    Emission_model_ptr emission_model() const;

    /**
     * @brief get the similarity module
     */
    Similarity_model_const_ptr similarity_model() const;
    
    /**
     * @brief allocate memory for the parameters of this sub-model
     */
    virtual void initialize_resources();

    /**
     * @brief set initial values for the parameters of this sub-model
     */
    virtual void initialize_params() = 0;
    
    virtual void input_previous_results(const std::string& input_path, const std::string& name);

    /*------------------------------------------------------------
     * INFERENCE INTERFACE
     *------------------------------------------------------------*/
    
    /**
     * @brief do a Gibbs update for the parameters of this sub-model
     */
    virtual void update_params() = 0;

    /*------------------------------------------------------------
     * LOGGING INTERFACE
     *------------------------------------------------------------*/
    virtual void set_up_results_log() const;
    virtual void add_ground_truth_eval_header() const = 0;
    
    virtual void compare_state_sequence_to_ground_truth(
        const std::string& ground_truth_path,
        const std::string& name
        ) const = 0;

    /*------------------------------------------------------------
     * DISPLAY FUNCTIONS
     *------------------------------------------------------------*/
    virtual void write_state_to_file(const std::string& name) const = 0;
    
    /*------------------------------------------------------------
     * ACCESSORS VISIBLE TO OTHER MODULES
     *------------------------------------------------------------*/

    size_t J() const;

    const size_t& D() const;

    size_t D_prime() const {return D_prime_;}
    
    const State_matrix& theta() const {return theta_;}
    State_matrix& theta() {return theta_;}

    const State_matrix& theta_prime() const {return *theta_prime_;}

    State_type theta_prime(const size_t& j) const {return theta_prime_->get_row(j);}

    Coordinate theta(const size_t& j, const size_t& d) const
    {
        return theta_.at(j,d);
    }

    virtual size_t dimension_map(const size_t& dprime) const {return dprime;}

    virtual size_t first_theta_prime_col_for_theta_d(const size_t& d) {return d;}
    virtual size_t num_theta_prime_cols_for_theta_d(const size_t&) {return 1;}
    
    /*------------------------------------------------------------
     * VERBOSE SET UP FUNCTIONS
     *------------------------------------------------------------*/
    void set_up_verbose_level(const size_t verbose){verbose_ = verbose;};

protected:
    /*------------------------------------------------------------
     * LVALUE ACCESSORS
     *------------------------------------------------------------*/

    Coordinate& theta(const size_t& j, const size_t& d)
    {
        return theta_.at(j,d);
    }
    
    /*------------------------------------------------------------
     * ACCESS TO OTHER MODULES
     *------------------------------------------------------------*/
    
    size_t T(const size_t& i) const;
    
    const State_matrix& theta_star(const size_t& i) const;
    
    const State_matrix_list& theta_star() const;
    
    const size_t& NF() const;
    
    /*
    const size_t& T() const;

    const State_matrix& theta_star() const;
     */
    
public:
    std::string write_path;

protected:
    /*------------------------------------------------------------
     * LINK VARIABLES
     *------------------------------------------------------------*/
    const HDP_HMM_LT* parent;
    const std::string theta_file;

    /*------------------------------------------------------------
     * STATE VARIABLES
     *------------------------------------------------------------*/
    State_matrix theta_;
    State_matrix* theta_prime_;
    
    /*------------------------------------------------------------
     * VERBOSE VARIABLE
     *------------------------------------------------------------*/
    size_t verbose_;

    /*------------------------------------------------------------
     * CONSTANTS
     *------------------------------------------------------------*/

    const size_t D_;
    size_t D_prime_;
    
    friend class HDP_HMM_LT;
};

#endif

