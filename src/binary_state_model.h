/* $Id: binary_state_model.h 21103 2017-01-22 16:27:51Z chuang $ */

#ifndef BINARY_STATE_MODEL_H_
#define BINARY_STATE_MODEL_H_

/*!
 * @file binary_state_model.h
 *
 * @author Colin Dawson 
 */

#include "parameters.h"
#include "util.h"
#include "state_model.h"
#include "linear_emission_model.h"
#include <l_cpp/l_int_matrix.h>
#include <boost/shared_ptr.hpp>

class Emission_model;
class Binary_state_model;
struct Binary_state_parameters;

typedef boost::shared_ptr<Binary_state_parameters> Binary_state_param_ptr;
typedef boost::shared_ptr<Binary_state_model> Binary_state_model_ptr;

struct Binary_state_parameters : public State_parameters
{
    const bool combinatorial_theta;
    const double a_mu;
    const double b_mu;

    Binary_state_parameters(
        const Parameters & params,
        const std::string& name = "Binary_state_model"
        );

    static const std::string& bad_theta_prior()
    {
        static const std::string msg =
            "ERROR: For a Binary state model, config file must either specify "
            "state_matrix_file, or specify positive a_mu and b_mu.";
        return msg;
    }
        
    virtual ~Binary_state_parameters() {}

    virtual State_model_ptr make_module() const;
};

class Binary_state_model : public State_model
{
public:
    /*------------------------------------------------------------
     * TYPEDEFS
     *------------------------------------------------------------*/
    typedef Binary_state_model        Self;
    typedef State_model               Base_class;
public:
    /*------------------------------------------------------------
     * CONSTRUCTORS
     *------------------------------------------------------------*/
    Binary_state_model(
        const Binary_state_parameters* const  hyperparams
        );

    /*------------------------------------------------------------
     * DESTRUCTOR
     *------------------------------------------------------------*/
    virtual ~Binary_state_model() {}
    
    /*------------------------------------------------------------
     * INITIALIZERS
     *------------------------------------------------------------*/
    virtual void initialize_params();
    virtual void initialize_resources();

    virtual void input_previous_results(const std::string& input_path, const std::string& name);

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
     * EVALUATION INTERFACE
     *------------------------------------------------------------*/
    virtual void add_ground_truth_eval_header() const;
    
    virtual void compare_state_sequence_to_ground_truth(
        const std::string& ground_truth_path,
        const std::string& name
        ) const;

protected:
    /*------------------------------------------------------------
     * ACCESS TO OTHER MODULES
     *------------------------------------------------------------*/
    int get_theta(const size_t& j, const size_t& d) const {return Base_class::theta(j,d) + 0.5;}
    /*------------------------------------------------------------
     * INTERNAL ACCESSORS
     *------------------------------------------------------------*/
    Prob mu(const size_t& d) const {return mu_.at(d);}
    Count entity_counts(const size_t& d) const {return entity_counts_.at(d);}

    /*------------------------------------------------------------
     * LVALUE ACCESSORS
     *------------------------------------------------------------*/
    Prob& mu(const size_t& d) {return mu_.at(d);}
    Count& entity_counts(const size_t& d) {return entity_counts_.at(d);}
    
    /*------------------------------------------------------------
     * INTERNAL INFERENCE FUNCTIONS
     *------------------------------------------------------------*/
    void update_mu_();
    void update_theta_(const size_t& j, const size_t& d);
    void update_theta_();
    double compute_zeta_jd_(const size_t& j, const size_t& d);

    /*------------------------------------------------------------
     * HYPERPARAMETERS
     *------------------------------------------------------------*/
    const double a_mu_;
    const double b_mu_;
    const bool fixed_theta;
    const bool combinatorial_theta;

    /*------------------------------------------------------------
     * STATE VARIABLES
     *------------------------------------------------------------*/
    Prob_vector mu_;
    
    /*------------------------------------------------------------
     * CACHE VARIABLES
     *------------------------------------------------------------*/
    Count_vector entity_counts_;
};

#endif

