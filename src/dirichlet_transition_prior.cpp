/* $Id: dirichlet_transition_prior.cpp 21468 2017-07-11 19:35:05Z cdawson $ */

/*!
 * @file dirichlet_transition_prior.cpp
 *
 * @author Colin Dawson 
 */

#include "util.h"
#include "dirichlet_transition_prior.h"
#include "dynamics_model.h"
#include <boost/make_shared.hpp>

size_t sample_table_count(const double& base, const size_t& max_val)
{
    /*
     use Chinese Restaurant Process to get the axuiliary variable m_{j,j'} where
     m_{j,j'} is number of tables for n_{j,j'} people
     s(n_{j,j'}, m_{j,j'}) is an unsigned Stirling number of the first kind
     refer to Samping beta in section 3 in the paper
     */
    if(max_val == 0)
    {
        // std::cerr << "            Trivial case: m = 0" << std::endl;
        return 0;
    } else {
        return kjb::sample_occupied_tables(kjb::Chinese_restaurant_process(base, max_val));
    }
}

Dirichlet_transition_prior_parameters::Dirichlet_transition_prior_parameters(
    const Parameters& params,
    const std::string& name
    ) : Transition_prior_parameters(params, name),
        fixed_alpha(params.exists(name, "alpha")),
        use_sticky(params.exists(name, "sticky_kappa") ||
                   params.exists(name, "sticky_c_kappa")),
        fixed_kappa(params.exists(name, "sticky_kappa")),
        informative_beta(params.exists(name, "beta_file")),
        alpha(
            !fixed_alpha ? NULL_CONC :
            params.get_param_as<double>(
                name, "alpha", bad_alpha_prior(), valid_conc_param)),
        beta_file(
            !informative_beta ? NULL_STRING :
            params.get_param_as<std::string>(
                name, "beta_file") + ".beta"),
        beta(informative_beta ? Prob_vector(beta_file.c_str()) : Prob_vector()),
        a_alpha(
            fixed_alpha ? NULL_SHAPE :
            params.get_param_as<double>(
                name, "a_alpha", bad_alpha_prior(), valid_shape_param)),
        b_alpha(
            fixed_alpha ? NULL_RATE :
            params.get_param_as<double>(
                name, "b_alpha", bad_alpha_prior(), valid_rate_param)),
        kappa(
            !use_sticky || !fixed_kappa ? 0 :
            params.get_param_as<double>(
                name, "sticky_kappa", bad_kappa_prior(), valid_mass)),
        c_kappa(
            !use_sticky || fixed_kappa ? NULL_SHAPE :
            params.get_param_as<double>(
                name, "sticky_c_kappa", bad_kappa_prior(), valid_shape_param)),
        d_kappa(
            !use_sticky || fixed_kappa ? NULL_SHAPE :
            params.get_param_as<double>(
                name, "sticky_d_kappa", bad_kappa_prior(), valid_shape_param))
{
    /*
     Read or sample parameters 
     alpha, concentration parameter for DP, fixed alpha, or alpha~Gamma(a_alpha, b_alpha)
     beta, base measure for DP
     a_alpha, shape parameter for alpha~Gamma(a_alpha, b_alpha)
     b_alpha, rate parameter for alpha~Gamma(a_alpha, b_alpha)
     kappa, extra weight on self-transiition in the Sticky-HMM paper (Fox et al. 2008)
     c_kappa, d_kappa, parameters of beta distribution for rho = kappa/(alpha + kappa)
     */
    IFT(valid_conc_param(alpha) ||
        (valid_shape_param(a_alpha) && valid_rate_param(b_alpha)),
        kjb::Illegal_argument,
        bad_alpha_prior());
    IFT(!informative_beta || beta.size() == (int) J,
        kjb::Illegal_argument,
        "ERROR: When specifying a beta vector for a Dirichlet transition prior, "
        "the size of the vector must match the J parameter.");
}

Dirichlet_transition_prior_parameters::Dirichlet_transition_prior_parameters(
    const Parameters&  params,
    const size_t&      J,
    const Prob_vector& beta,
    const std::string& name
    ) : Transition_prior_parameters(J),
        fixed_alpha(params.exists(name, "alpha")),
        use_sticky(false), fixed_kappa(true),
        informative_beta(true),
        alpha(
            !fixed_alpha ? NULL_CONC :
            params.get_param_as<double>(
                name, "alpha", bad_alpha_prior(), valid_conc_param)),
        beta(beta),
        a_alpha(
            fixed_alpha ? NULL_SHAPE :
            params.get_param_as<double>(
                name, "a_alpha", bad_alpha_prior(), valid_shape_param)),
        b_alpha(
            fixed_alpha ? NULL_RATE :
            params.get_param_as<double>(
                name, "b_alpha", bad_alpha_prior(), valid_rate_param)),
        kappa(0),
        c_kappa(NULL_SHAPE), d_kappa(NULL_SHAPE)
{
    IFT(valid_conc_param(alpha) ||
        (valid_shape_param(a_alpha) && valid_rate_param(b_alpha)),
        kjb::Illegal_argument,
        bad_alpha_prior());
    IFT(beta.size() == (int) J,
        kjb::Illegal_argument,
        "ERROR: When specifying a beta vector for a Dirichlet transition prior, "
        "the size of the vector must match the J parameter.");
}

Dirichlet_transition_prior::Dirichlet_transition_prior(
    const Params* const hyperparams
    ) : Base_class(hyperparams),
        fixed_alpha(hyperparams->fixed_alpha),
        fixed_kappa(hyperparams->fixed_kappa),
        informative_beta(hyperparams->informative_beta),
        use_sticky(hyperparams->use_sticky),
        a_alpha_(hyperparams->a_alpha),
        b_alpha_(hyperparams->b_alpha),
        c_kappa_(hyperparams->c_kappa),
        d_kappa_(hyperparams->d_kappa),
        alpha_(log(hyperparams->alpha)),
        beta_(hyperparams->beta),
        kappa_(hyperparams->kappa),
        rho_(0),
        u_(), sum_log_one_plus_u_(0.0),
        M_(), m_dot_(), m_dot_dot_(0), w_(), C_()
{}

Transition_prior_ptr Dirichlet_transition_prior_parameters::make_module() const
{
    return boost::make_shared<Dirichlet_transition_prior>(this);
}

void Dirichlet_transition_prior::initialize_resources()
{
    /*
     allocate memory resource for transition module variable and correponding auxiliary variable
     u: u_(j) ~ Gamma(n_{j.}, T_{j}), refer to equation (10) in the paper, 
     u_{j} is the total time spent in state j given that it is visited n_{j} times
     M: m_{j,j'} is the auxiliary variable for sampling beta (section 3 in the paper), and 
        we will sample it by using Chinese Restaurant Process
     w: auxiliary vairables for the update of parameters in Sticky-HMM
     C_: c_{j,j'} total transition from state j to j' (n_{j,j'} + q_{j,j'})
     */
    Base_class::initialize_resources();
    if(informative_beta)
    {
        PM(verbose_ > 0, "     Using fixed beta from file.\n");
        //std::cerr << "    Using fixed beta from file."
        //          << std::endl;
    } else {
        PM(verbose_ > 0, "     Allocating beta...");
        //std::cerr << "    Allocating beta...";
        beta_ = Prob_vector((int) J(), 0.0);
        PM(verbose_ > 0, "done.\n");
        //std::cerr << "done." << std::endl;
    }
    PM(verbose_ > 0, "     Allocating u...");
    //std::cerr << "    Allocating u...";
    u_ = Time_array(J() + 1, 0);
    sum_log_one_plus_u_ = 0;
    PM(verbose_ > 0, "done.\n");
    //std::cerr << "done." << std::endl;
    PM(verbose_ > 0, "     Allocating M...");
    //std::cerr << "    Allocating M...";
    M_ = Count_matrix(J() + 1, J(), 0);
    m_dot_ = Count_vector(J(), 0);
    m_dot_dot_ = 0;
    w_ = Count_vector(J(), 0);
    PM(verbose_ > 0, "done.\n");
    //std::cerr << "done." << std::endl;
    //std::cerr << "    Allocating C...";
    PM(verbose_ > 0, "    Allocating C...");
    C_ = Count_matrix(J() + 1, J(), 0);
    PM(verbose_ > 0, "done.\n");
    //std::cerr << "done." << std::endl;
}

void Dirichlet_transition_prior::initialize_params()
{
    /*
     Initialize the parameters for 
     alpha, concentration parameter for DP,
     kappa, extra mass for self-transition in Sticky-HMM
     beta, base measure for DP
     distribution parameters inputted in constructor
     */
    Base_class::initialize_params();
    if(fixed_alpha)
    {
        //std::cerr << "    Using fixed alpha = " << exp(alpha_) << std::endl;
        PMWP(verbose_ > 0, "    Using fixed alpha = %.6f\n", (exp(alpha_)));
    } else {
        Conc_dist r_alpha(a_alpha_, b_alpha_);
        PM(verbose_ > 0, "    Initializing alpha = ");
        // std::cerr << "    Initializing alpha = ";
        alpha_ = log(1 + kjb::sample(r_alpha));
        PMWP(verbose_ > 0, "%.6f", (exp(alpha_)));
        //std::cerr << exp(alpha_) << std::endl;
    }
    if(fixed_kappa)
    {
        //std::cerr << "    Using fixed kappa = " << kappa_ << std::endl;
        PMWP(verbose_ > 0, "     Using fixed kappa = %.6f", (kappa_));
    } else if(use_sticky)
    {
        kjb::Beta_distribution r_rho(c_kappa_, d_kappa_);
        rho_ = kjb::sample(r_rho);
        PMWP(verbose_ > 0, "     Initializing kappa = %.6f\n", (exp(alpha_) * rho_ / (1 - rho_)));
        //std::cerr << "    Initializing kappa = " << exp(alpha_) * rho_ / (1 - rho_) << std::endl;
        PMWP(verbose_ > 0, "     Rho = %.6f\n", (rho_));
        //std::cerr << "    Rho = " << rho_ << std::endl;
    }
    this->initialize_beta_();
    update_pi_();
}

void Dirichlet_transition_prior::input_previous_results(const std::string& input_path, const std::string& name)
{
    Base_class::input_previous_results(input_path, name);
    PM(verbose_ > 0, "Inputting information for dirichlet transition prior...\n");
    //std::cerr << "Inputting information for dirichlet transition prior..." << std::endl;
    if (!fixed_alpha)
    {
        PM(verbose_ > 0, "Inputting information for alpha_...\n");
        //std::cerr << "Inputting information for alpha_..." << std::endl;
        alpha_ = log(input_to_value<double>(input_path, "alpha.txt", name));
        //std::cerr << "alpha_ = " << alpha_ << std::endl;
        PMWP(verbose_ > 0, "alpha = %.6f\n", (alpha_));
    }
    std::cerr << "Inputting information for beta_..." << std::endl;
    PM(verbose_ > 0, "Inpuuting information for beta...\n");
    beta_ = kjb::ew_log(input_to_vector<double>(input_path, "beta.txt", name));
    if (verbose_ > 0) {std::cerr << "beta_ = " << beta_ << std::endl;}
    //std::cerr << "Inputting information for u_..." << std::endl;
    //u_ = input_to_vector<Cts_duration>(write_path, "u.txt", name);
    //std::cerr << "u_ = " << u_ << std::endl;
    //std::cerr << "Inputting information for Q_..." << std::endl;
    //Q_ = Count_matrix((write_path + "Q/" + name + ".txt").c_str());
    if (!fixed_kappa && use_sticky && (input_path == write_path))
    {
        //std::cerr << "Inputting information for kappa_..." << std::endl;
        PM(verbose_ > 0, "Inputting information for kappa_...\n");
        kappa_ = input_to_value<double>(input_path, "kappa.txt", name);
        PMWP(verbose_ > 0, "kappa_ = %.6f\n", (kappa_));
        //std::cerr << "kappa_ = " << kappa_ << std::endl;
    }
    std::cerr << "done." << std::endl;
}

void Dirichlet_transition_prior::update_params()
{
    Base_class::update_params();
    if(!fixed_alpha) update_alpha_();
    update_pi_();
}

void Dirichlet_transition_prior::set_up_results_log() const
{
    Base_class::set_up_results_log();
    create_directory_if_nonexistent(write_path + "pi");
    std::ofstream ofs;
    ofs.open(write_path + "alpha.txt", std::ofstream::out);
    ofs << "iteration value" << std::endl;
    ofs.close();
    ofs.open(write_path + "beta.txt", std::ofstream::out);
    ofs << "iteration value" << std::endl;
    ofs.close();
    ofs.open(write_path + "u.txt", std::ofstream::out);
    ofs << "iteration value" << std::endl;
    ofs.close();
    if (use_sticky)
    {
        ofs.open(write_path + "kappa.txt", std::ofstream::out);
        ofs << "iteration value" << std::endl;
        ofs.close();
    }
}

void Dirichlet_transition_prior::write_state_to_file(const std::string& name) const
{
    Base_class::write_state_to_file(name);
    std::ofstream ofs;
    ofs.open(write_path + "alpha.txt", std::ofstream::out | std::ofstream::app);
    ofs << name << " " << exp(alpha()) << std::endl;
    ofs.close();
    ofs.open(write_path + "beta.txt", std::ofstream::out | std::ofstream::app);
    ofs << name << kjb::ew_exponentiate(beta_) << std::endl;
    ofs.close();
    ofs.open(write_path + "u.txt", std::ofstream::out | std::ofstream::app);
    ofs << name << " " << u_ << std::endl;
    ofs.close();
    if (use_sticky)
    {
        ofs.open(write_path + "kappa.txt", std::ofstream::out | std::ofstream::app);
        ofs << name << " " << kappa_ << std::endl;
        ofs.close();
    }
}

void Dirichlet_transition_prior::initialize_beta_()
{
    /*
     initialize beta, the base measure for DP
     */
    if(informative_beta)
    {
        IFTD(beta_.size() == (int) J(),
             kjb::IO_error,
             "I/O ERROR: Reading in a length %d beta vector, but model has J = %d.",
             (beta_.size())(J())
            );
        beta_ = kjb::log_normalize(kjb::ew_log(beta_));
    } else {
        PM(verbose_ > 0, "    Intializing beta to uniform...");
        //std::cerr << "    Initializing beta to uniform...";
        beta_ = Prob_vector((int) J(), -log((double) J()));
        PM(verbose_ > 0, "done.\n");
        //std::cerr << "done." << std::endl;
    }
}

void Dirichlet_transition_prior::update_alpha_()
{
    /*
     sampling alpha, concentration parameters
     refer to section 3.1 in the paper, sampling concentration parameters
     */
    double shape = a_alpha_ + m_dot_dot_ + w_dot_;
    threshold_a_double(shape, write_path, "alpha");
    double scale = exp(-log(b_alpha_ + sum_log_one_plus_u_));
    // std::cerr << "        Sampling from G(" << shape << "," << scale << ") distribution" << std::endl;
    Conc_dist r_alpha(shape, scale);
    double alpha_plus_kappa = kjb::sample(r_alpha);
    // TODO: replace 1s with hyperparams
    if(use_sticky && !fixed_kappa)
    {
        kjb::Beta_distribution r_rho(c_kappa_ + w_dot_, d_kappa_ + m_dot_dot_);
        rho_ = use_sticky ? kjb::sample(r_rho) : 0.0;
        kappa_ = rho_ * alpha_plus_kappa;
        PMWP(verbose_ > 0, "   Updating rho = %.6f\n", (rho_));
        //std::cerr << "    Updating rho = " << rho_ << std::endl;
        PMWP(verbose_ > 0, "   Updating kappa = %.6f\n", (kappa_));
        //std::cerr << "    Updating kappa = " << kappa_ << std::endl;
    }
    alpha_ = log((1 - rho_) * alpha_plus_kappa);
    PMWP(verbose_ > 0, "    Updating alpha = %.6f\n", (exp(alpha_)));
    //std::cerr << "    Updating alpha = ";
    //std::cerr << exp(alpha_) << std::endl;
}

void Dirichlet_transition_prior::update_pi_()
{
    /*
     sampling pi, the transition matrix 
     refer to section 3.1 in the paper, sampling pi
     */
    PM(verbose_ > 0, "    Updating pi...");
    //std::cerr << "    Updating pi...";
    for(size_t j = 0; j < J(); ++j)
    {
        double shape0 = exp(alpha() + beta(j)) + N(J(),j);
        // double scale0 = exp(-log(1.0 + u(J())));
        if(log(shape0) < -100)
        {
            pi0(j) = log(0.0);
        } else {
            // Rate_dist r_pi0(shape0, scale0);
            Rate_dist r_pi0(shape0, 1.0);
            pi0(j) = log(kjb::sample(r_pi0));
        }
        for(size_t jp = 0; jp < J(); ++jp)
        {
            int self_transition(j == jp);
            double shape = exp(alpha() + beta(jp)) + self_transition * kappa_ +
                N(j,jp) + Q(j,jp);
            double scale = exp(-log(1.0 + u(j)));
            if(log(shape) < -100)
            {
                pi(j,jp) = log(0.0);
            } else {
                Rate_dist r_pi(shape, scale);
                pi(j,jp) = log(kjb::sample(r_pi));
            }
        }
    }
    PM(verbose_ > 0, "done.\n");
    //std::cerr << "done." << std::endl;
    //std::cerr << "pi_ = " << std::endl;
    //std::cerr << pi_ << std::endl;
}

void Dirichlet_transition_prior::update_u_and_Q_()
{
    /*
     update auxilary variable u, 
        u_{j} is the total time spent in state j given that it is visited n_{j} times,
     and Q, q_{j,j'} number of unsuccessful attempts to jump from state j to j', 
        initialized in base class (transition_prior)
     refer to section 3.2 in the paper
     based on information from hidden state squence z in dynamic model
     */
    PM(verbose_ > 0, "     Updating u and Q...");
    //std::cerr << "    Updating u and Q...";
    sum_log_one_plus_u_ = 0;

    // const double log_u0_rate = kjb::log_sum(pi0().begin(), pi0().end());
    // if(log_u0_rate < -100)
    // {
    //     std::cerr << "Warning: Underflow in log sum of pi0. "
    //               << "pi0 = " << pi0_
    //               << std::endl;
    // }
    // Dur_dist r_u0(1.0, exp(-log_u0_rate));
    // u(J()) = kjb::sample(r_u0);
    for(size_t j = 0; j < J(); ++j)
    {
        //std::cerr << "n_dot(" << j << ") = " << n_dot(j) << std::endl;
        if(n_dot(j) == 0) u(j) = 0;
        else {
            kjb::Vector v = A().get_row(j);
            //std::cerr << "v_ = " << kjb::ew_exponentiate(v) << std::endl;
            const double log_u_rate = kjb::log_sum(v.begin(), v.end());
            if(log_u_rate < -100)
            {
                std::cerr << "Warning: Underflow in log sum of row "
                          << j << " of pi."
                          << std::endl;
            }
            Dur_dist r_u(n_dot(j), exp(-log_u_rate));
            u(j) = kjb::sample(r_u);
            sum_log_one_plus_u_ += log_of_one_more_than(u(j));
        }
        for (size_t jp = 0; jp < J(); ++jp)
        {
            const double q_mean = exp(pi(j,jp) + log(u(j))) * (1 - exp(Phi(j,jp)));
            if(q_mean < FLT_EPSILON)
            {
                Q(j,jp) = 0;
            } else {
                Count_dist r_q(q_mean);
                Q(j,jp) = kjb::sample(r_q);
            }
            // if(Q(j,jp) > 100000)
            // {
            //     std::cerr << "        Q(" << j << "," << jp << ") = " << Q(j,jp) << std::endl;
            // }
        }
    }
    //std::cerr << "u_ = " << u_ << std::endl;
    //std::cerr << "Q_ = " << Q_ << std::endl;
    //std::cerr << "done." << std::endl;
    PM(verbose_ > 0, "done.\n");
}

void Dirichlet_transition_prior::update_M_()
{
    /*
     update M, the auxiliary variable for sampling beta, refer to section 3.1 in the paper
     m_{j,j'} is ranged from 0 to n_{jj'}+q_{jj'}
     and s(n,m) is an unsigned Sirling number of the first kind
     C_: c_{j,j'} = n_{j,j'} + q_{j,j'}
     */
    //std::cerr << "    Updating M...";
    PM(verbose_ > 0, "    Updating M...");
    C_ = N() + Q();
    std::fill(m_dot_.begin(), m_dot_.end(), 0);
    std::fill(w_.begin(), w_.end(), 0);
    w_dot_ = 0;
    for(size_t j = 0; j <= J(); ++j)
    {
        //assert(N(J(), j) == 0 || N(J(), j) == 1);
        // m_dot_[j] += N(J(), j);
        for(size_t jp = 0; jp < J(); ++jp)
        {
            if(C(j,jp) > 100000000)
            {
                PMWP(verbose_ > 0,
                     "         Sampling table count with C(%d, %d) = %d\n",
                     (j)(jp)(C(j,jp)));
                //std::cerr << "        Sampling table count with " << std::endl;
                //std::cerr << "        C(" << j << "," << jp << ") = " << C(j,jp) << std::endl;
            }
            M(j,jp) = sample_table_count(
                exp(alpha() + beta(jp)) + int(j == jp) * kappa_,
                (size_t) C(j,jp));
            m_dot_[jp] += M(j,jp);
        }
        if(j != J() && use_sticky && M(j,j) > 0)
        {
            kjb::Binomial_distribution rw_j(M(j,j), rho_ / (rho_ + exp(beta(j)) * (1 - rho_)));
            w_[j] = kjb::sample(rw_j);
            M(j,j) -= w_[j];
            m_dot_[j] -= w_[j];
            w_dot_ += w_[j];
        }
    }
    m_dot_dot_ = kjb::sum_elements(M_);
    PM(verbose_ > 0, "done.\n");
    //std::cerr << "done." << std::endl;
}
