/* $Id: dynamics_model.h 21443 2017-06-28 18:13:50Z chuang $ */

#ifndef DYNAMICS_MODEL_H_
#define DYNAMICS_MODEL_H_

/*!
 * @file dynamics_model.h
 *
 * @author Colin Dawson 
 */

#include "util.h"
#include "parameters.h"
#include <boost/shared_ptr.hpp>

// class Transition_model;
class HDP_HMM_LT;
class Dynamics_model;
struct Dynamics_parameters;

typedef boost::shared_ptr<Dynamics_parameters> Dynamics_param_ptr;
typedef boost::shared_ptr<Dynamics_model> Dynamics_model_ptr;

struct Dynamics_parameters
{
    const std::string sampling_method;

    Dynamics_parameters(
        const Parameters& params,
        const std::string& name
        ) : sampling_method(
                params.exists(name, "sampling_method") ?
                params.get_param_as<std::string>(
                    name, "sampling_method") :
                "weak_limit")
    {}
    
    virtual Dynamics_model_ptr make_module() const = 0;
    virtual ~Dynamics_parameters() {}
};

class Dynamics_model
{
    typedef Dynamics_model Self;
    typedef Dynamics_parameters Params;
protected:
    /*------------------------------------------------------------
     * CONSTRUCTORS
     *------------------------------------------------------------*/
    Dynamics_model(const Params* const hyperparams)
        : parent(), sampling_method(hyperparams->sampling_method)
 {}
public:
    /*------------------------------------------------------------
     * DESTRUCTOR
     *------------------------------------------------------------*/
    virtual ~Dynamics_model()
    {}

    /*------------------------------------------------------------
     * INITIALIZATION FUNCTIONS
     *------------------------------------------------------------*/
    virtual void initialize_resources();
    virtual void initialize_params();
    void set_parent(HDP_HMM_LT* const p);
    //virtual void sample_labels_from_prior() = 0;
    virtual void sample_labels_from_prior(const T_list& T) = 0;
    
    /*------------------------------------------------------------
     * CONTINUE PREVIOUS INFERENCE
     *------------------------------------------------------------*/
    
    void input_previous_results(const std::string& input_path, const std::string& name);

    /*------------------------------------------------------------
     * INFERENCE INTERFACE
     *------------------------------------------------------------*/
    /*
    virtual void update_params(const Likelihood_matrix& llm);
     */
    virtual void update_params(const size_t& i, const Likelihood_matrix& llm);
    virtual double get_log_marginal_likelihood(const size_t& i, const Likelihood_matrix& llm) = 0;
    virtual double get_test_log_marginal_likelihood(const size_t& i, const Likelihood_matrix& llm) = 0;
    /*
    virtual double get_log_marginal_likelihood(const Likelihood_matrix& llm) = 0;
     */
    
    /*------------------------------------------------------------
     * DISPLAY FUNCTIONS
     *------------------------------------------------------------*/
    virtual void set_up_results_log() const;
    virtual void write_state_to_file(const std::string& name) const;

    /*------------------------------------------------------------
     * ACCESSORS VISIBLE TO OTHER MODULES
     *------------------------------------------------------------*/
    const State_sequence& z(const size_t& i) const {return z_.at(i);}
    State_indicator z(const size_t& i, const size_t& t) const {return z_.at(i).at(t);};
    const State_sequence_list& z() const {return z_;}
    //const State_sequence& z() const {return z_;}
    //State_indicator z(const size_t& t) const {return z_.at(t);}

    /**
     * @brief return a map from state indices to sets of time steps
     */
    const Time_set& partition_map(const State_indicator& j) const;
    
    /*------------------------------------------------------------
     * VERBOSE SET UP FUNCTIONS
     *------------------------------------------------------------*/
    void set_up_verbose_level(const size_t verbose){verbose_ = verbose;};
    
protected:
    /*------------------------------------------------------------
     * ACCESS TO OTHER MODULES
     *------------------------------------------------------------*/
    const size_t& NF() const;
    size_t T(const size_t& i) const;
    T_list T() const;
    //size_t T() const;
    size_t J() const;
    //size_t test_T() const;
    size_t test_T(const size_t& i) const;
    T_list test_T() const;
    const Prob_vector& pi0() const;
    const Prob& pi0(const size_t& j) const;
    const Prob_matrix& A() const;
    const Prob& A(const size_t& j, const size_t& jp) const;
    const Count& n_dot(const size_t& j) const;
    void update_augmented_state_matrix();
    
    Likelihood_matrix log_likelihood_matrix(
        const size_t&           i,
        const kjb::Index_range& states = kjb::Index_range::ALL,
        const kjb::Index_range& times =  kjb::Index_range::ALL
        ) const;
    
    /*
    Likelihood_matrix log_likelihood_matrix(
        const kjb::Index_range& states = kjb::Index_range::ALL,
        const kjb::Index_range& times =  kjb::Index_range::ALL
        ) const;
     */

    /*------------------------------------------------------------
     * LVALUE ACCESSORS
     *------------------------------------------------------------*/
    Prob& A(const size_t& j, const size_t& jp);
    /*
    Prob& log_marginal_likelihood(){return log_marginal_likelihood_;}
     */
    Prob_vector& log_marginal_likelihood(){return log_marginal_likelihood_;}
    Prob& log_marginal_likelihood(const size_t& i){return log_marginal_likelihood_.at(i);}
    
    State_sequence& z(const size_t i) {return z_.at(i);}
    State_indicator& z(const size_t i, const size_t t) {return z_.at(i).at(t);}
    //State_indicator& z(const size_t& t) {return z_.at(t);}
    
    /*------------------------------------------------------------
     * INTERNAL INFERENCE FUNCTIONS
     *------------------------------------------------------------*/
    virtual void mask_A() {}
    void initialize_partition_();
    void update_partition_();

    /*------------------------------------------------------------
     * STATE VARIABLES
     *------------------------------------------------------------*/
    //State_sequence     z_; //!< Indicator variables for state sequence
    State_sequence_list  z_; //A vector of sequences
    
    /*------------------------------------------------------------
     * LINKS
     *------------------------------------------------------------*/
    HDP_HMM_LT* parent;
    std::string write_path;
    
    /*------------------------------------------------------------
     * VERBOSE VARIABLE
     *------------------------------------------------------------*/
    size_t verbose_;

    /*------------------------------------------------------------
     * MODE VARIABLES
     *------------------------------------------------------------*/
    const std::string sampling_method;
    
    /*------------------------------------------------------------
     * EVALUATION VARIABLES
     *------------------------------------------------------------*/
    /*
    mutable Prob log_marginal_likelihood_;
     */
    mutable Prob_vector log_marginal_likelihood_;

    friend class HDP_HMM_LT;
};

#endif
