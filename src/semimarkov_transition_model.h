/* $Id: semimarkov_transition_model.h 21265 2017-02-21 15:54:09Z chuang $ */

#ifndef SEMIMARKOV_TRANSITION_MODEL_H_
#define SEMIMARKOV_TRANSITION_MODEL_H_

/*!
 * @file semimarkov_transition_model.h
 *
 * @author Colin Dawson 
 */

#include "parameters.h"
#include "transition_prior.h"
#include "dynamics_model.h"

class Semimarkov_transition_model;
struct Semimarkov_transition_parameters;

typedef boost::shared_ptr<Semimarkov_transition_parameters> Semimarkov_param_ptr;
typedef boost::shared_ptr<Semimarkov_transition_model> Semimarkov_model_ptr;

struct Semimarkov_transition_parameters : public Dynamics_parameters
{
    const double a_omega; //!< shape parameter for omega_ prior
    const double b_omega; //!< rate parameter for omega_ prior

    Semimarkov_transition_parameters(
        const Parameters & params,
        const std::string& name = "Semimarkov_transition_model"
        ) : Dynamics_parameters(params, name),
            a_omega(
                params.get_param_as<double>(
                    name, "a_omega", bad_omega_prior(), valid_shape_param)),
            b_omega(
                params.get_param_as<double>(
                    name, "b_omega", bad_omega_prior(), valid_rate_param))
    {}

    static const std::string& bad_omega_prior()
    {
        static const std::string msg = 
            "ERROR: For an HSMM (Poisson) dynamics, "
            " config file must specify "
            " both positive a_omega (shape) and b_omega (rate) "
            " hyperparameters.";
        return msg;
    }

    virtual ~Semimarkov_transition_parameters() {}

    virtual Dynamics_model_ptr make_module() const;
};

class Semimarkov_transition_model : public Dynamics_model
{
public:
    typedef Semimarkov_transition_model Self;
    typedef Dynamics_model Base_class;
    typedef Poisson_dist Dur_dist;
    typedef Gamma_dist Dur_param_dist;
    typedef std::vector<Dur_dist> Dur_dist_array;
    typedef std::vector<Prob_matrix> Prob_matrix_list;
    typedef std::vector<Prob_matrix_list> Prob_matrix_list_list;
    using Dynamics_model::A;
public:
    /*------------------------------------------------------------
     * CONSTRUCTORS
     *------------------------------------------------------------*/
    
    Semimarkov_transition_model(
        const Semimarkov_transition_parameters* const hyperparams
        ) : Base_class(hyperparams),
            a_omega_(hyperparams->a_omega),
            b_omega_(hyperparams->b_omega),
            omega_(), duration_priors_(), d_totals_(),
            B_(), Bstar_(), interval_likelihoods_()
    {};

    virtual ~Semimarkov_transition_model() {}
protected:

    /*------------------------------------------------------------
     * ACCESSORS TO OTHER MODULES
     *------------------------------------------------------------*/
    /*
    Likelihood_matrix log_likelihood_matrix(
        const kjb::Index_range& states = kjb::Index_range::ALL,
        const kjb::Index_range& times = kjb::Index_range::ALL
        );
     */
    
    virtual void mask_A_();
    /*
    virtual void update_z_(const Likelihood_matrix& llm);
     */
    void update_duration_priors_();
    virtual void update_z_(const size_t& i, const Likelihood_matrix& llm);

    virtual void initialize_resources();
    virtual void initialize_params();
    virtual void sample_labels_from_prior(const T_list& T);
    /*
    virtual void sample_labels_from_prior();
    virtual void update_params(const Likelihood_matrix& llm);
    virtual double get_log_marginal_likelihood(const Likelihood_matrix& llm);
     */
    virtual void update_params(const size_t& i, const Likelihood_matrix& llm);
    virtual double get_log_marginal_likelihood(const size_t& i, const Likelihood_matrix& llm);
    virtual double get_test_log_marginal_likelihood(const size_t& i, const Likelihood_matrix& llm);
    virtual void set_up_results_log() const;

    //virtual void input_previous_results(const std::string& name);
    virtual void write_state_to_file(const std::string& name) const;
    
    /*
    virtual void pass_messages_backwards_(
        Prob&                     log_marginal_likelihood,
        const Likelihood_matrix&  llm
        );
     virtual void sample_z_forwards_();
     virtual void pass_test_messages_backwards_(
        Prob&                     log_marginal_likelihood,
        const Likelihood_matrix&  llm
     );
     */
    virtual void pass_messages_backwards_(
        Prob&                     log_marginal_likelihood,
        const size_t&             i,
        const Likelihood_matrix&  llm
        );
    virtual void pass_test_messages_backwards_(
        Prob&                     log_marginal_likelihood,
        const size_t&             i,
        const Likelihood_matrix&  llm
        );
    virtual void sample_z_forwards_(const size_t& i);

    Count d_totals(const size_t& j) const {return d_totals_.at(j);}
    Count& d_totals(const size_t& j) {return d_totals_.at(j);}
    const Prob_matrix& interval_likelihoods(const size_t& i, const size_t& t) const {return interval_likelihoods_.at(i).at(t);}
    Prob_matrix& interval_likelihoods(const size_t& i, const size_t& t) {return interval_likelihoods_.at(i).at(t);}
    /*
    const Prob_matrix& interval_likelihoods(const size_t& t) const {return interval_likelihoods_.at(t);}
    Prob_matrix& interval_likelihoods(const size_t& t) {return interval_likelihoods_.at(t);}
     */

    /// Hyperparameters
    const double a_omega_;
    const double b_omega_;

    /// Duration parameters
    Time_array omega_;

    /// For message passing
    /*
    Prob_matrix duration_priors_;
     */
    Prob_matrix_list duration_priors_;
    Count_vector d_totals_;
    // The (t,j) entry in B_ is the likelihood of all
    // data from time t to T given that the chain is in state j at time t-1
    
    Prob_matrix_list B_;
    Prob_matrix_list Bstar_;
    Prob_matrix_list_list interval_likelihoods_;
    
    //Prob_matrix B_; //!< (log of) message matrix B described in Johnson and Wilsky
    // The other message matrix.  Analogous to the other, but "off by one":
    // the (t,j) entry is the likelihood of all data from time t to T, given
    // that we are in state j at time t
    //Prob_matrix Bstar_; //!< (log of) message matrix B* described in Johnson and Wilsky
    // a list containing one matrix per time step, where the (d,j) entry
    // of the matrix at position t represents the joint log likelihood fn
    // for the d+1 observations starting at time t, assuming
    // a state segment in state j starting at time t.  
    //std::vector<Prob_matrix> interval_likelihoods_;
};

#endif
