/* $Id: hdp_transition_prior.h 21468 2017-07-11 19:35:05Z cdawson $ */

#ifndef HDP_TRANSITION_PRIOR_H_
#define HDP_TRANSITION_PRIOR_H_

/*!
 * @file hdp_transition_prior.h
 *
 * @author Colin Dawson 
 */

#include "util.h"
#include "parameters.h"
#include "dirichlet_transition_prior.h"

class HDP_transition_prior;

struct HDP_transition_prior_parameters : public Dirichlet_transition_prior_parameters
{
    typedef Dirichlet_transition_prior_parameters Base;
    const bool   fixed_gamma;
    const double gamma;
    const double a_gamma;
    const double b_gamma;

    HDP_transition_prior_parameters(
        const Parameters& params,
        const std::string& name = "HDP_hyperprior"
        ) :
        Base(params, name),
        fixed_gamma(params.exists(name, "gamma")),
        gamma(
            !fixed_gamma ? NULL_CONC :
            params.get_param_as<double>(
                name, "gamma", bad_gamma_prior(), valid_conc_param)),
        a_gamma(
            fixed_gamma ? NULL_SHAPE :
            params.get_param_as<double>(
                name, "a_gamma", bad_gamma_prior(), valid_shape_param)),
        b_gamma(
            fixed_gamma ? NULL_RATE :
            params.get_param_as<double>(
                name, "b_gamma", bad_gamma_prior(), valid_rate_param))
    {
        IFT((a_gamma > 0 && b_gamma > 0) || gamma > 0, kjb::IO_error,
            bad_gamma_prior());
    }

    HDP_transition_prior_parameters(
        const Parameters&  params,
        const size_t&      J,
        const std::string& name = "HDP_hyperprior"
        ) : Base(params, J, Prob_vector(), name),
            fixed_gamma(params.exists(name, "gamma")),
            gamma(
                !fixed_gamma ? NULL_CONC :
                params.get_param_as<double>(
                    name, "gamma", bad_gamma_prior(), valid_conc_param)),
            a_gamma(
                fixed_gamma ? NULL_SHAPE :
                params.get_param_as<double>(
                    name, "a_gamma", bad_gamma_prior(), valid_shape_param)),
            b_gamma(
                fixed_gamma ? NULL_RATE :
                params.get_param_as<double>(
                    name, "b_gamma", bad_gamma_prior(), valid_rate_param))
    {
        IFT(a_gamma > 0 && b_gamma > 0, kjb::IO_error,
            bad_gamma_prior());
    }

    static const std::string& bad_gamma_prior()
    {
        static const std::string msg = 
            "ERROR: For an HDP transition prior, "
            " config file must either specify gamma > 0, or "
            " both positive a_gamma (shape) and b_gamma (rate) "
            " hyperparameters.";
        return msg;
    }

    virtual ~HDP_transition_prior_parameters() {}

    virtual Transition_prior_ptr make_module() const;
};

class HDP_transition_prior : public Dirichlet_transition_prior
{
    typedef Dirichlet_transition_prior Base_class;
public:
    HDP_transition_prior(
        const HDP_transition_prior_parameters* const hyperparams
        );

    virtual ~HDP_transition_prior() {}

    virtual void initialize_resources();

    virtual void initialize_params();
    
    virtual void input_previous_results(const std::string& input_path, const std::string& name);

    virtual void update_params();

    virtual void update_auxiliary_data()
    {
        Base_class::update_auxiliary_data();
        PM(verbose_ > 0, "    Updating r and t...");
        update_r_();
        update_t_();
        PM(verbose_ > 0, "done.");
    }

    virtual void set_up_results_log() const;

    virtual void write_state_to_file(const std::string& name) const;

    Conc alpha() const {return alpha_;}

protected:
    void update_gamma_();
    void update_beta_();
    void update_r_();
    void update_t_();

public:
    const bool fixed_gamma;

protected:
    const double  a_gamma_;
    const double  b_gamma_;
    Conc          gamma_; //!< log concentration parameter
    Count_vector  r_; //!< Sticks per state at top level
    Count         r_dot_; //!< Total sticks at top level
    double        t_; //!< Auxiliary variable to sample gammma
};

#endif
