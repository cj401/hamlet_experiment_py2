/* $Id: dirichlet_transition_prior.h 21103 2017-01-22 16:27:51Z chuang $ */

#ifndef DIRICHLET_TRANSITION_PRIOR_H_
#define DIRICHLET_TRANSITION_PRIOR_H_

/*!
 * @file symmetric_dirichlet_transition_prior.h
 *
 * @author Colin Dawson 
 */

#include "transition_prior.h"
#include "parameters.h"

struct Dirichlet_transition_prior_parameters :
    public Transition_prior_parameters
{
    const bool fixed_alpha;
    const bool use_sticky;
    const bool fixed_kappa;
    const bool informative_beta;
    const double alpha;
    const std::string beta_file;
    const Prob_vector beta;
    const double a_alpha;
    const double b_alpha;
    const double kappa;
    const double c_kappa;
    const double d_kappa;
    
    Dirichlet_transition_prior_parameters(
        const Parameters& params,
        const std::string& name = "Dirichlet_hyperprior"
        );

    Dirichlet_transition_prior_parameters(
        const Parameters&  params,
        const size_t&      J,
        const Prob_vector& beta,
        const std::string& name = "Dirichlet_hyperprior"
        );

public:
    static const std::string& bad_alpha_prior()
    {
        static const std::string msg = 
            "ERROR: For a Dirichlet transition prior, "
            " config file must either specify alpha > 0, or "
            " both positive a_alpha (shape) and b_alpha (rate) "
            " hyperparameters.";
        return msg;
    }
    
    static const std::string& bad_kappa_prior()
    {
        static const std::string msg =
            "ERROR: Sticky HDP-HMM requires sticky_kappa >= 0 for "
            "a fixed kappa, or sticky_c_kappa and sticky_d_kappa > 0";
        return msg;
    }
    
    virtual ~Dirichlet_transition_prior_parameters() {}

    virtual Transition_prior_ptr make_module() const;
};

class Dirichlet_transition_prior : public Transition_prior
{
public:
    typedef Dirichlet_transition_prior Self;
    typedef Dirichlet_transition_prior_parameters Params;
    typedef Transition_prior Base_class;
public:
    Dirichlet_transition_prior(
        const Params* const hyperparams
        );

    void initialize_resources();
    
    void initialize_params();
    
    virtual void input_previous_results(const std::string& input_path, const std::string& name);

    virtual void update_params();
    
    virtual void update_auxiliary_data()
    {
        Base_class::update_auxiliary_data();
        update_u_and_Q_();
        update_M_();
    }

    virtual void set_up_results_log() const;

    virtual void write_state_to_file(const std::string& name) const;

    virtual ~Dirichlet_transition_prior() {}
protected:
    Count M(const size_t& j, const size_t& jp) const {return M_.at(j,jp);}
    Count& M(const size_t& j, const size_t& jp) {return M_.at(j,jp);}
    Conc alpha() const {return alpha_;}
    Prob beta(const size_t& j) const {return beta_.at(j);}
    Cts_duration u(const size_t& j) const {return u_.at(j);}
    Count C(const size_t& j, const size_t& jp) const {return C_.at(j,jp);}

    Prob& beta(const size_t& j) {return beta_.at(j);}
    Cts_duration& u(const size_t& j) {return u_.at(j);}

    /*------------------------------------------------------------
     * INTERNAL INFERENCE FUNCTIONS
     *------------------------------------------------------------*/
    virtual void initialize_beta_();
    void update_alpha_();
    void update_pi_();
    void update_u_and_Q_();
    void update_M_();
public:
    const bool fixed_alpha;
    const bool informative_beta;
    const bool use_sticky;
    const bool fixed_kappa;
protected:
    /*------------------------------------------------------------
     * HYPERPARAMETERS
     *------------------------------------------------------------*/
    const double a_alpha_;
    const double b_alpha_;
    const double c_kappa_;
    const double d_kappa_;

    /*------------------------------------------------------------
     * PARAMETERS
     *------------------------------------------------------------*/
    Conc alpha_;
    Prob_vector beta_;
    Conc kappa_;
    Prob rho_;
    
    /*------------------------------------------------------------
     * AUXILIARY DATA
     *------------------------------------------------------------*/
    Time_array         u_; //!< Auxiliary vector of time spent in each state
    double             sum_log_one_plus_u_;
    Count_matrix       M_; //!< Auxiliary "sticks-per-transition-type"
    Count_vector       m_dot_; //!< Marginal sticks-per-destination-state across all sources
    Count              m_dot_dot_; //!< Double marginal of M_
    Count_vector       w_; //!< 'Override' counts for Sticky HDP-HMM
    Count              w_dot_; //!< Sum of w_j

    /*------------------------------------------------------------
     * CACHE VARIABLES
     *------------------------------------------------------------*/
    Count_matrix       C_; //!< Matrix of total transitions (N + Q)
};

size_t sample_table_count(const double& base, const size_t& max_val);

#endif
