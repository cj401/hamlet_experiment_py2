/* $Id: binary_state_model.cpp 21509 2017-07-21 17:39:13Z cdawson $ */

/*!
 * @file binary_state_model.cpp
 *
 * @author Colin Dawson 
 */

#include "util.h"
#include "binary_state_model.h"
#include "emission_model.h"
#include "similarity_model.h"
#include "hdp_hmm_lt.h"
#include <third_party/underflow_utils.h>
#include <l_cpp/l_int_vector.h>
#include <boost/make_shared.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>

State_model_ptr Binary_state_parameters::make_module() const
{
    return boost::make_shared<Binary_state_model>(this);
}

inline double logistic(double x)
{
    return LogitInverse(x);
}

Binary_state_parameters::Binary_state_parameters(
    const Parameters & params,
    const std::string& name
    ) : State_parameters(params, name),
        combinatorial_theta(
            params.exists(name, "combinatorial_theta")
            ? params.get_param_as<bool>(name, "combinatorial_theta")
            : false),
        a_mu(fixed_theta || combinatorial_theta ? NULL_SHAPE :
             params.get_param_as<double>(
                 name, "a_mu", bad_theta_prior(),
                 valid_shape_param)),
        b_mu(fixed_theta || combinatorial_theta ? NULL_RATE :
             params.get_param_as<double>(
                 name, "b_mu", bad_theta_prior(),
                 valid_rate_param))
{}

Binary_state_model::Binary_state_model(
    const Binary_state_parameters* const hyperparams
    ) : State_model(hyperparams),
        a_mu_(hyperparams->a_mu), b_mu_(hyperparams->b_mu),
        fixed_theta(hyperparams->fixed_theta),
        combinatorial_theta(hyperparams->combinatorial_theta),
        mu_(), 
        entity_counts_(hyperparams->D, 0)
{}

void Binary_state_model::initialize_params()
{
    //std::cerr << "Initializing binary state model..." << std::endl;
    PM(verbose_ > 0, "Initializing binary state model...\n");
    // Initialize lambda from the prior
    if(fixed_theta)
    {
        PMWP(verbose_ > 0,
            "    Using fixed theta matrix from file '%s'\n",
             (theta_file.c_str()));
        //std::cerr << "    Using fixed theta matrix from file "
        //          << theta_file << std::endl;
    } else if(combinatorial_theta) {
        theta_ = generate_matrix_of_binary_range(D());
    } else {
        // Initialize mu from the prior
        //std::cerr << "    Initializing mu...";
        PM(verbose_ > 0, "    Initializing mu...");
        Beta_dist r_mu(a_mu_, b_mu_);
        for(Prob_vector::iterator mu_it = mu_.begin(); mu_it != mu_.end(); ++mu_it) 
        {
            (*mu_it) = kjb::sample(r_mu);
        }
        // std::cerr << mu_ << std::endl;
        PM(verbose_ > 0, "    done...\n");
        // std::cerr << "done." << std::endl;
        // Initialize theta from the prior conditioned on mu
        PM(verbose_ > 0, "    Initializing theta...");
        //std::cerr << "    Initializing theta...";
        for(size_t j = 0; j < J(); ++j)
        {
            for(size_t d = 0; d < D(); ++d)
            {
                Bernoulli_dist r_theta(mu(d));
                theta(j,d) = kjb::sample(r_theta);
                entity_counts(d) += theta(j,d);
            }
        }
        PM(verbose_ > 0, "    done...\n");
        //std::cerr << "done." << std::endl;
    }
    theta_prime_ = &theta_;
}

void Binary_state_model::initialize_resources()
{
    PMWP(verbose_ > 0,
         "Initializing resources for binary state model with combinatorial_theta %s\n",
         (combinatorial_theta));
    
    //std::cerr << "Initializing resources for binary state model "
    //          << " with combinatorial_theta = " << combinatorial_theta
    //          << std::endl;
              // << ", parent = " << parent << ","
              // << " and (J,D) = " << J() << "," << D() << ")" << std::endl;
    
    if(combinatorial_theta)
    {
        if(J() != pow(2,D()))
        {
            PMWP(verbose_ > 0,
                 "WARNING: J parameter specified as %d, but when combinatorial theta is used, \
                 J must be equal to 2^D.\n", (J()));
            PMWP(verbose_ > 0,
                 "Since D is %d, J should be %d.", (D())(pow(2,D())));
            /*
            std::cerr << "WARNING: J parameter specified as " << J()
                      << ", but when combinatorial theta is used, "
                      << "J must be equal to 2^D." << std::endl;
            std::cerr << "Since D is " << D()
                      << ", J should be " << pow(2,D())
                      << "." << std::endl;
             */
            // J() = pow(2,D());
        }
    }
    Base_class::initialize_resources();
    if(!(fixed_theta || combinatorial_theta))
    {
        PMWP(verbose_ > 0, "    (J is %d)\n", (J()));
        PM(verbose_ > 0, "    Allocating mu...");
        //std::cerr << "    (J is " << J() << ")" << std::endl;
        //std::cerr << "    Allocating mu...";
        mu_ = Prob_vector((int) D(), 0.0);
        PM(verbose_ > 0, "done.\n");
        //std::cerr << "done." << std::endl;
        PM(verbose_ > 0, "    Allocating entity counts...");
        //std::cerr << "    Allocating entity counts...";
        entity_counts_ = Count_vector(D(), 0);
        PM(verbose_ > 0, "done.\n");
        //std::cerr << "done." << std::endl;
    }
}

void Binary_state_model::update_params()
{
    PM(verbose_ > 0, "Updating state parameters...\n");
    //std::cerr << "Updating state parameters..." << std::endl;
    if(!(fixed_theta || combinatorial_theta))
    {
        update_theta_(); 
        update_mu_(); 
    }
}

void Binary_state_model::input_previous_results(const std::string& input_path, const std::string& name)
{
    PM(verbose_ > 0, "Inputting information for binary state model...\n");
    //std::cerr << "Inputting information for binary state model..." << std::endl;
    Base_class::input_previous_results(input_path, name);
    if (!(fixed_theta || combinatorial_theta))
    {
        PM(verbose_ > 0, "Inputting information for prob vector mu_...\n");
        //std::cerr << "Inputting information for prob vector mu_..." << std::endl;
        mu_ = Prob_vector(input_to_vector<double>(input_path, "mu.txt", name));
        PM(verbose_ > 0, "Recalculating the entity counts...\n");
        //std::cerr << "Recalculating the entity counts..." << std::endl;
        entity_counts_ = Count_vector(D(), 0);
        for(size_t j = 0; j < J(); ++j)
        {
            for(size_t d = 0; d < D(); ++d)
            {
                entity_counts(d) += theta(j,d);
            }
        }
        PM(verbose_ > 0, "done.\n");
        //std::cerr << "done." << std::endl;
    }
}

void Binary_state_model::add_ground_truth_eval_header() const
{
    add_binary_gt_eval_header(write_path);
}

void Binary_state_model::compare_state_sequence_to_ground_truth(
    const std::string& ground_truth_path,
    const std::string& name
    ) const
{
    // State_matrix thetastar = theta_star().submatrix(0,0,T(),D());
    //need change later
    score_binary_states(write_path, ground_truth_path, name, theta_star());
}

void Binary_state_model::set_up_results_log() const
{
    Base_class::set_up_results_log();
    if(!(fixed_theta || combinatorial_theta))
    {
        std::ofstream ofs;
        ofs.open(write_path + "mu.txt", std::ofstream::out);
        ofs << "iteration value" << std::endl;
        ofs.close();
    } else {
        theta().floor().write((write_path + "theta/theta.txt").c_str());
    }
}

/*
void Binary_state_model::write_state_to_file(const std::string& name) const
{
    if(!(fixed_theta || combinatorial_theta))
    {
        const std::string filestem = write_path + name;
        std::ofstream ofs;
        kjb::Matrix(theta()).submatrix(0,0,J(), D()).floor().write(
            (write_path + "theta/" + name + ".txt").c_str());
        ofs.open(write_path + "mu.txt", std::ofstream::out | std::ofstream::app);
        ofs << name << " " << mu_ << std::endl;
        ofs.close();
    }
    kjb::Matrix(theta_star()).submatrix(0,0,T(), D_prime()).floor().write(
        (write_path + "thetastar/" + name + ".txt").c_str());
}
 */

void Binary_state_model::write_state_to_file(const std::string& name) const
{
    if(!(fixed_theta || combinatorial_theta))
    {
        const std::string filestem = write_path + name;
        std::ofstream ofs;
        kjb::Matrix(theta()).submatrix(0,0,J(), D()).floor().write(
            (write_path + "theta/" + name + ".txt").c_str());
        ofs.open(write_path + "mu.txt", std::ofstream::out | std::ofstream::app);
        ofs << name << " " << mu_ << std::endl;
        ofs.close();
    }
    if (NF() == 1)
    {
        kjb::Matrix(theta_star(0)).submatrix(0,0,T(0), D_prime()).floor().write(
            (write_path + "thetastar/" + name + ".txt").c_str());
    }
    else
    {
        create_directory_if_nonexistent((write_path + "thetastar/" + name).c_str());
        boost::format fmt("%03i");
        for (size_t i = 0; i < NF(); i++)
        {
            std::string file_name = (fmt % (i+1)).str() + ".txt";
            kjb::Matrix(theta_star(i)).submatrix(0,0,T(i), D_prime()).floor().write(
                (write_path + "thetastar/" + name + "/" + file_name).c_str());
        }
    }
}

void Binary_state_model::update_mu_()
{
    //std::cerr << "    Updating mu...";
    PM(verbose_ > 0, "    Updating mu...");
    // kjb::Int_vector::const_iterator ec_it = entity_counts_.begin();
    // for(Prob_vector::iterator mu_it = mu_.begin(); mu_it != mu_.end(); ++mu_it, ++ec_it)
    // {
    //     Beta_dist r_mu(a_mu_ + (*ec_it), b_mu_ + J() - (*ec_it));
    //     (*mu_it) = kjb::sample(r_mu);
    // }
    for(size_t d = 0; d < D(); ++d)
    {
        size_t ones_at_d = entity_counts(d);
        Beta_dist r_mu(a_mu_ + ones_at_d, b_mu_ + J() - ones_at_d);
        mu(d) = kjb::sample(r_mu);
    }
    PM(verbose_ > 0, "done.\n");
    //std::cerr << "done." << std::endl;
}

double Binary_state_model::compute_zeta_jd_(const size_t &j, const size_t &d)
{
    /// zeta_jd represents the probability that theta_jd = 1
    /// We first compute the log odds, and then apply the logistic transform
    // the prior piece, plus the likelihood component due to observed transitions
    double prior_log_odds = log(mu(d) / (1 - mu(d)));
    double log_odds = prior_log_odds;
    // std::cerr << "    prior log odds = " << prior_log_odds << std::endl;
    log_odds += (similarity_model()->log_likelihood_for_state(j,d,1)
                 - similarity_model()->log_likelihood_for_state(j,d,0));
    // std::cerr << "    similarity component = " << log_odds - prior_log_odds << std::endl;
    // now compute the contribution of the data.  now we compute a ratio for
    // flipping vs the status quo.  If theta(j,d) = 1, we want the inverse of this.
    int emission_step_sign = (get_theta(j,d) == 1 ? -1 : 1);
    double emission_part =
        parent->log_likelihood_ratio_for_state_change(j, emission_step_sign, d);
    // std::cerr << "    Emission part[" << j << "," << d << "] = " << emission_part
    //           << std::endl;
    log_odds += emission_step_sign * emission_part;
    // std::cerr << "    total likelihood component = " << log_odds - prior_log_odds << std::endl;
    return logistic(log_odds);
}

void Binary_state_model::update_theta_(const size_t& j, const size_t& d)
{
    const int theta_old = get_theta(j,d);
    int theta_new;
    double zeta = compute_zeta_jd_(j,d);
    // std::cerr << "zeta[" << j << "," << d << "] = " << zeta << std::endl;
    Bernoulli_dist r_theta(zeta);
    theta_new = kjb::sample(r_theta);
    if(theta_old == theta_new) return;
    theta(j,d) = theta_new;
    similarity_model()->sync_after_theta_update(j, d, theta_old, theta_new);
    entity_counts(d) += 2*theta_new - 1;
    // emission_model()->shift_mean_for_state_j(j, 2*theta_new - 1, d);
}

void Binary_state_model::update_theta_()
{
    //std::cerr << "    Updating theta:" << std::endl;
    PM(verbose_ > 0, "    Updating theta:\n");
    similarity_model()->sync_before_theta_update();
    //std::cerr << "        Resampling theta values...";
    PM(verbose_ > 0, "        Resampling theta values...");
    for(size_t j = 0; j < J(); ++j)
    {
        for(size_t d = 0; d < D(); ++d)
        {
            update_theta_(j,d);
        }
    }
    PM(verbose_ > 0, "done.\n");
    //std::cerr << "done." << std::endl;
    // std::cerr << theta_.floor() << std::endl;
    //std::cerr << "    Done updating theta." << std::endl;
    PM(verbose_ > 0, "    Done updating theta.\n");
}

