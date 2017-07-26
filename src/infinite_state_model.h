
#ifndef INFINITE_STATE_MODEL_H_
#define INFINITE_STATE_MODEL_H_

#include "parameters.h"
#include "util.h"
#include "state_model.h"
#include "linear_emission_model.h"
#include <boost/shared_ptr.hpp>

class Emission_model;
class Infinite_state_model;
struct Infinite_state_parameters;

typedef boost::shared_ptr<Infinite_state_parameters> Infinite_state_param_ptr;
typedef boost::shared_ptr<Infinite_state_model> Infinite_state_model_ptr;

struct Infinite_state_parameters : public State_parameters
{
    const bool alpha;
    
    Infinite_state_parameters(
        const Parameters & params,
        const std::string& name = "Infinite_state_model"
        );
    
    static const std::string& bad_theta_prior()
    {
        static const std::string msg =
            "ERROR: For a Infinite state model, config file must specify positive alpha.";
        return msg;
    }
    
    virtual ~Infinite_state_parameters() {}
    
    virtual State_model_ptr make_module() const;
};

class Infinite_state_model : public State_model
{
public:
    /*------------------------------------------------------------
     * TYPEDEFS
     *------------------------------------------------------------*/
    typedef Infinite_state_model        Self;
    typedef State_model                 Base_class;
    
public:
    /*------------------------------------------------------------
     * CONSTRUCTORS
     *------------------------------------------------------------*/
    Infinite_state_model(
        const Infinite_state_parameters* const  hyperparams
        );
    
    /*------------------------------------------------------------
     * DESTRUCTOR
     *------------------------------------------------------------*/
    virtual ~Infinite_state_model() {}
    
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
     * ACCESS BY OTHER MODULES
     *------------------------------------------------------------*/
    double alpha() const {return alpha_;};
    
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
    
    void sample_inactive_states();
    void update_mu_();
    void update_theta_(const size_t& j, const size_t& d);
    void update_theta_();
    double compute_zeta_jd_(const size_t& j, const size_t& d);
    friend double mu_log_density(double mu, Self* model);
    // double get_minimums(Prob_vector v, double* min);
    
    /*------------------------------------------------------------
     * HYPERPARAMETERS
     *------------------------------------------------------------*/
    const double alpha_;
    
    /*------------------------------------------------------------
     * STATE VARIABLES
     *------------------------------------------------------------*/
    Prob_vector mu_;
    
    /*------------------------------------------------------------
     * CACHE VARIABLES
     *------------------------------------------------------------*/
    double mu_star_;
    double mu_star2_;
    double s_;
    Count_vector entity_counts_;
    size_t num_active_states_;
    size_t num_inactive_states_;
};

#endif
