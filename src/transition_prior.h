/* $Id: transition_prior.h 21443 2017-06-28 18:13:50Z chuang $ */

#ifndef TRANSITION_PRIOR_H_
#define TRANSITION_PRIOR_H_

/*!
 * @file transition_prior.h
 *
 * @author Colin Dawson 
 */

#include "util.h"
#include "parameters.h"
#include <boost/shared_ptr.hpp>

class HDP_HMM_LT;
typedef boost::shared_ptr<Transition_prior_parameters> Transition_prior_param_ptr;

struct Transition_prior_parameters
{
    const size_t J;
    
    Transition_prior_parameters(
        const Parameters& params,
        const std::string& name
        ) : J(params.get_param_as<size_t>(name, "J"))
    {}

    Transition_prior_parameters(const size_t& J)
        : J(J)
    {}
    
    virtual ~Transition_prior_parameters() {}

    virtual Transition_prior_ptr make_module() const = 0;
};

class Transition_prior
{
public:
    friend class Transition_model;
    typedef Transition_prior Self;
    typedef Transition_prior_parameters Params;
public:
    Transition_prior(const Params* const hyperparams);
    virtual ~Transition_prior() {}
    
    virtual void initialize_resources();
    virtual void initialize_params();
    virtual void update_params();
    void set_parent(HDP_HMM_LT* const p);
    
    virtual void update_auxiliary_data();

    virtual void set_up_results_log() const;
    virtual void write_state_to_file(const std::string& name) const;
    
    /*------------------------------------------------------------
     * CONTINUE PREVIOUS INFERENCE
     *------------------------------------------------------------*/
    
     virtual void input_previous_results(const std::string& input_path, const std::string& name);

    /*------------------------------------------------------------
     * ACCESSORS VISIBLE TO OTHER MODULES
     *------------------------------------------------------------*/
    
    /**
     * @brief get the unscaled transition matrix
     */
    const Prob_matrix& pi() const {return pi_;}

    /**
     * @brief get element (j,jp) of the unscaled transition matrix
     */
    Prob pi(const size_t& j, const size_t& jp) const {return pi_.at(j,jp);}
    
    /**
     * @brief get the number of states
     */
    const size_t& J() const {return J_;}
    
    /**
     * @brief get the initial distribution
     */
    const Prob_vector& pi0() const {return pi0_;}
    
    /**
     * @brief get a particular element of the initial distribution
     */
    const Prob& pi0(const size_t& j) const {return pi0_.at(j);}
    
    /**
     * @brief get the matrix of successful transition counts
     */
    const Count_matrix& N() const {return N_;}

    /**
     * @brief return the number of successful transitions from j to jp
     */
    Count N(const size_t& j, const size_t& jp) const {return N_.at(j,jp);}

    /**
     * @brief return the total number of successful transitions to jp
     */
    Count n_dot(const size_t& j) const {return n_dot_.at(j);}
    
    /**
     * @brief return the matrix of failed jump attempts
     */
    const Count_matrix& Q() const {return Q_;}

    /**
     * @brief return the number of failed jump attempts from j to jp
     */
    Count Q(const size_t& j, const size_t& jp) const {return Q_.at(j,jp);}

    /*------------------------------------------------------------
     * ACCESS TO OTHER MODULES
     *------------------------------------------------------------*/
    const Prob_matrix& A() const;
    const Prob_matrix& Phi() const;
    Prob Phi(const size_t& j, const size_t& jp) const;
    const State_sequence_list& z() const;
    const State_sequence& z(const size_t& i) const;
    const State_indicator& z(const size_t& i, const size_t& t) const;
    /*
    const State_sequence& z() const;
    const State_indicator& z(const size_t& t) const;
     */
    
    /*------------------------------------------------------------
     * VERBOSE SET UP FUNCTIONS
     *------------------------------------------------------------*/
    void set_up_verbose_level(const size_t verbose){verbose_ = verbose;};

    /*------------------------------------------------------------
     * LVALUE ACCESSORS
     *------------------------------------------------------------*/
    Count_matrix& N() {return N_;}
    Count& n_dot(const size_t& j) {return n_dot_.at(j);}
    Prob& pi(const size_t& j, const size_t& jp) {return pi_.at(j,jp);}
    Prob_vector& pi0() {return pi0_;}
    Prob& pi0(const size_t& j) {return pi0_.at(j);}
    Count& N(const size_t& j, const size_t& jp) {return N_.at(j,jp);}
    Count& Q(const size_t& j, const size_t& jp) {return Q_.at(j,jp);}

    /*------------------------------------------------------------
     * SYNC FUNCTIONS
     *------------------------------------------------------------*/
    void sync_transition_counts() {sync_transition_counts(z());}
    void sync_transition_counts(const State_sequence_list& labels);
    /*
    void sync_transition_counts() {sync_transition_counts(z());}
    void sync_transition_counts(const State_sequence& labels);
     */

    /*------------------------------------------------------------
     * DIMENSION VARIABLES
     *------------------------------------------------------------*/
    const size_t J_;
    
    /*------------------------------------------------------------
     * STATE VARIABLES
     *------------------------------------------------------------*/
    Prob_vector        pi0_; //!< Log initial distribution
    Prob_matrix        pi_; //!< Log unscaled transition matrix
    Count_matrix       N_; //!< Matrix of state-to-state transition counts
    Count_vector       n_dot_; //!< Marginal occurrences of each state

    /*------------------------------------------------------------
     * AUXILIARY DATA VARIABLES
     *------------------------------------------------------------*/
    Count_matrix       Q_; //!< Auxiliary failed state-to-state jump attempts
    
    HDP_HMM_LT* parent;
    std::string write_path;
    
    /*------------------------------------------------------------
     * VERBOSE VARIABLE
     *------------------------------------------------------------*/
    size_t verbose_;
};

#endif

