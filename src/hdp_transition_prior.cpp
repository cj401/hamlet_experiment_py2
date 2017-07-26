/* $Id: hdp_transition_prior.cpp 21468 2017-07-11 19:35:05Z cdawson $ */

/*!
 * @file hdp_transition_prior.cpp
 *
 * @author Colin Dawson 
 */

#include "hdp_transition_prior.h"
#include <boost/make_shared.hpp>

HDP_transition_prior::HDP_transition_prior(
    const HDP_transition_prior_parameters* const   hyperparams
    ) : Dirichlet_transition_prior(hyperparams),
        fixed_gamma(hyperparams->fixed_gamma),
        a_gamma_(hyperparams->a_gamma),
        b_gamma_(hyperparams->b_gamma),
        gamma_(log(hyperparams->gamma)),
        r_(), r_dot_(0), t_(0.0)
{}

Transition_prior_ptr HDP_transition_prior_parameters::make_module() const
{
    return boost::make_shared<HDP_transition_prior>(this);
}

void HDP_transition_prior::initialize_resources()
{
    /*
     allocate memory for auxiliary variable r which is used for sampling
     gamma in the GEM(gamma) sticky-breaking process
     refer to equation (19) in section 3.1 in the paper
     */
    Base_class::initialize_resources();
    PM(verbose_ > 0, "    Allocating r...");
    //std::cerr << "    Allocating r...";
    r_ = Count_vector(J(), 0);
    r_dot_ = 0;
    t_ = 0;
    PM(verbose_ > 0, "done.\n");
    //std::cerr << "done." << std::endl;
}

void HDP_transition_prior::initialize_params()
{
    /*
     gamma_ is in log, which gamma is initialized to 1,
     or informative fixed gamma
     Then do the sticky-breaking process to get beta
     */
    Base_class::initialize_params();
    if(fixed_gamma)
    {
        PMWP(verbose_ > 0, "    Using fixed gamma = %.6f\n", (exp(gamma_)));
        //std::cerr << "    Using fixed gamma = " << exp(gamma_) << std::endl;
    } else {
        PM(verbose_ > 0, "Initializing gamma = ");
        //std::cerr << "    Initializing gamma = ";
        gamma_ = 0.0;
        PMWP(verbose_ > 0, "%.6f", (exp(gamma_)));
        //std::cerr << exp(gamma_) << std::endl;
    }
    PM(verbose_ > 0, "    Initializing beta from prior...\n");
    //std::cerr << "    Initializing beta from prior..." << std::endl;
    update_beta_();
    update_pi_();
}

void HDP_transition_prior::input_previous_results(const std::string& input_path, const std::string& name)
{
    PM(verbose_ > 0, "Inputting information for HDP transition prior...\n");
    //std::cerr << "Inputting information for HDP transition prior..." << std::endl;
    Base_class::input_previous_results(input_path, name);
    if (!fixed_gamma)
    {
        PM(verbose_ > 0, "Inputting information for gamma_...\n");
        //std::cerr << "Inputting information for gamma_..." << std::endl;
        gamma_ = log(input_to_value<Conc>(input_path, "gamma.txt", name));
        PMWP(verbose_ > 0, "%.6f", (exp(gamma_)));
        //std::cerr << "gamma_ = " << gamma_ << std::endl;
    }
    PM(verbose_ > 0, "done.");
    //std::cerr << "done." << std::endl;
}

void HDP_transition_prior::update_params()
{
    Transition_prior::update_params();
    if(!fixed_alpha || !fixed_kappa) update_alpha_();
    if(!fixed_gamma) update_gamma_();
    update_beta_();
    update_pi_();
}

void HDP_transition_prior::set_up_results_log() const
{
    Base_class::set_up_results_log();
    std::ofstream ofs;
    ofs.open(write_path + "gamma.txt", std::ofstream::out);
    ofs << "iteration value" << std::endl;
    ofs.close();
}

void HDP_transition_prior::write_state_to_file(const std::string& name) const
{
    const std::string filestem = write_path + name;
    Base_class::write_state_to_file(name);
    std::ofstream ofs;
    ofs.open(write_path + "gamma.txt", std::ofstream::out | std::ofstream::app);
    ofs << name << " " << exp(gamma_) << std::endl;
    ofs.close();
}

void HDP_transition_prior::update_gamma_()
{
    /*
     sample gamma, refer to equation (20) in section 3.1 in the paper
     */
    PM(verbose_ > 0, "    Updating gamma = ");
    //std::cerr << "    Updating gamma = ";
    double shape = a_gamma_ + r_dot_ - 1;
    // threshold_a_double(shape, write_path, "gamma");
    double rate = b_gamma_ - t_;
    kjb::Categorical_distribution<> r_indicator(0,1,shape, rate);
    Conc_dist r_gamma(shape + kjb::sample(r_indicator), 1.0 / rate);
    gamma_ = log(kjb::sample(r_gamma));
    PMWP(verbose_> 0, "%d",(exp(gamma_)));
}

void HDP_transition_prior::update_beta_()
{
    /*
     sample beta, refer to equation (14) in section 3.1 in the paper
     the posterior is a dirichlet
     */
    PM(verbose_ > 0, "    Updating beta...");
    //std::cerr << "    Updating beta...";
    sample_log_from_dirichlet_posterior_with_symmetric_prior(
        exp(gamma_ - log((double) J())), m_dot_.begin(), m_dot_.end(),
        beta_.begin(), beta_.end(), write_path);
    PM(verbose_ > 0, "done.\n");
    //std::cerr << "done." << std::endl;
    //std::cerr << "beta = " << kjb::ew_exponentiate(beta_) << std::endl;
}

void HDP_transition_prior::update_r_()
{
    /*
     sample auxiliary variable r, refer to equation (18) in section 3.1
     */
    r_dot_ = 0;
    Count_vector::const_iterator m_dot_it = m_dot_.begin();
    double base = exp(gamma_ - log(J()));
    for(Count_vector::iterator r_it = r_.begin(); r_it != r_.end(); ++r_it, ++m_dot_it)
    {
        (*r_it) = sample_table_count(base, (*m_dot_it));
        r_dot_ += (*r_it);
    }
}

void HDP_transition_prior::update_t_()
{
    /*
     sample auxiliary variable t, refer to equation (19) in section 3.1
     */
    Beta_dist r_t(exp(gamma_) + 1, m_dot_dot_);
    double exp_t = kjb::sample(r_t);
    threshold_a_double(exp_t, write_path, "t");
    t_ = log(exp_t);
}
