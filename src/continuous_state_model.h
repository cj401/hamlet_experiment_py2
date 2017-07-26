/* $Id: continuous_state_model.h 21103 2017-01-22 16:27:51Z chuang $ */

#ifndef CONTINUOUS_STATE_MODEL_H_
#define CONTINUOUS_STATE_MODEL_H_

/*!
 * @file continuous_state_model.h
 *
 * @author Colin Dawson 
 */

#include "state_model.h"
#include "parameters.h"
#include "similarity_model.h"
#include <boost/shared_ptr.hpp>

class Emission_model;
class Continuous_state_model;
struct Continuous_state_parameters;

typedef boost::shared_ptr<Continuous_state_model> Cts_state_model_ptr;
typedef boost::shared_ptr<Continuous_state_parameters> Cts_state_params_ptr;

struct Continuous_state_parameters : public State_parameters
{
    const double prior_precision;
    const size_t L;
    const double epsilon;

    Continuous_state_parameters(
        const Parameters&   params,
        const std::string&  name = "Continuous_state_model"
        );

    static const std::string& bad_prior_precision()
    {
        static const std::string msg =
            "ERROR: For a Continuous state model, config file must specify "
            "a positive prior_precision parameter.";
        return msg;
    }

    virtual ~Continuous_state_parameters() {}

    virtual State_model_ptr make_module() const;
};

class  Continuous_state_model : public State_model
{
    /*------------------------------------------------------------
     * TYPEDEFS
     *------------------------------------------------------------*/
    typedef Continuous_state_model Self;
    typedef State_model            Base_class;

public:
    /*------------------------------------------------------------
     * CONSTRUCTORS
     *------------------------------------------------------------*/

    Continuous_state_model(
        const Continuous_state_parameters* const hyperparams
        );

    /*------------------------------------------------------------
     * DESTRUCTOR
     *------------------------------------------------------------*/
    virtual ~Continuous_state_model() {}

    /*------------------------------------------------------------
     * INITIALIZERS
     *------------------------------------------------------------*/

    virtual void initialize_params();
    virtual void initialize_resources();
    
    virtual void input_previous_results(const std::string& input_path, const std::string& name);

    /*------------------------------------------------------------
     * CALCULATION FUNCTIONS
     *------------------------------------------------------------*/

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
        const std::string&,
        const std::string&
        ) const;

protected:

    void update_theta_();

    Prob theta_log_posterior(const Similarity_model::HMC_interface& model) const;
    std::vector<Prob> theta_log_posterior_gradient(
        const Similarity_model::HMC_interface& model
        ) const;

    /*------------------------------------------------------------
     * HYPERPARAMETERS
     *------------------------------------------------------------*/

    const double prior_precision_;
    const size_t L_; //<! number of HMC leapfrog steps per Gibbs iteration
    const double epsilon_; //<! size of one HMC leapfrog step (on each dimension)

    /*------------------------------------------------------------
     * LOG VARIABLES
     *------------------------------------------------------------*/

    size_t last_accepted;
    double acceptance_rate;
    size_t starting_iterations;
    size_t starting_num_accepted;
};

#endif

