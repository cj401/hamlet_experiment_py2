/*!
 * @file infinite_state_model.cpp
 *
 */

#include "util.h"
#include "infinite_state_model.h"
#include "emission_model.h"
#include "hdp_hmm_lt.h"
#include <l_cpp/l_int_vector.h>
#include <third_party/underflow_utils.h>
#include <third_party/arms.h>
#include <boost/make_shared.hpp>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>

double get_minimums(Prob_vector v, double* min);

State_model_ptr Infinite_state_parameters::make_module() const
{
    return boost::make_shared<Infinite_state_model>(this);
}

inline double logistic(double x)
{
    return LogitInverse(x);
}

Infinite_state_parameters::Infinite_state_parameters(
    const Parameters & params,
    const std::string& name
    ) : State_parameters(params, name),
        alpha(params.get_param_as<double>(name, "alpha", bad_theta_prior(), valid_rate_param))
{}

Infinite_state_model::Infinite_state_model(
    const Infinite_state_parameters* const hyperparams
    ) : State_model(hyperparams),
        alpha_(hyperparams->alpha),
        mu_(),
        entity_counts_()
{}

void Infinite_state_model::set_up_results_log() const
{
    Base_class::set_up_results_log();
    std::ofstream ofs;
    ofs.open(write_path + "mu.txt", std::ofstream::out);
    ofs << "iteration value" << std::endl;
    ofs.close();
}

void Infinite_state_model::write_state_to_file(const std::string& name) const
{
    const std::string filestem = write_path + name;
    std::ofstream ofs;
    kjb::Matrix(theta()).submatrix(0, 0, J(), D()).floor().write(
        (write_path + "theta/" + name + ".txt").c_str());
    ofs.open(write_path + "mu.txt", std::ofstream::out | std::ofstream::app);
    ofs << name << " " << mu_ << std::endl;
    ofs.close();
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

void Infinite_state_model::input_previous_results(const std::string& input_path, const std::string& name)
{
    PM(verbose_ > 0, "Inputting information for infinite state model...");
    Base_class::input_previous_results(input_path, name);
    mu_ = Prob_vector(input_to_vector<double>(input_path, "mu.txt", name));
    mu_star2_ = get_minimums(mu_, &mu_star_);
    entity_counts_ = Count_vector(mu_.size(), 0);
    for (size_t j = 0; j < J(); ++j)
    {
        for (size_t d = 0; d < entity_counts_.size(); ++d)
        {
            entity_counts(d) += theta(j,d);
        }
    }
    PM(verbose_ > 0, "done.\n");
}

void Infinite_state_model::initialize_params()
{
    PM(verbose_ > 0, "Initializing infinite state model...\n");
    PM(verbose_ > 0, "Initializing theta...");
    Count_vector d_priors_ = Count_vector(J()+1);
    for (size_t j = 1; j < J()+1; j++)
    {
        Count_dist r_dishes(alpha_/j);
        d_priors_.at(j) = kjb::sample(r_dishes);
    }
    Count_vector cum_d_priors_ = Count_vector(J()+1);
    std::partial_sum(d_priors_.begin(), d_priors_.end(), cum_d_priors_.begin());
    theta_ = State_matrix((int) J(), (int) cum_d_priors_[J()]);
    int dimension_prior = (int) cum_d_priors_[J()];
    entity_counts_ = Count_vector(dimension_prior, 0);
    mu_ = Prob_vector((int) cum_d_priors_[J()]);
    for (size_t j = 0; j < J(); j++)
    {
        for (size_t d = 0; d < cum_d_priors_[j+1]; d++)
        {
            if (d < cum_d_priors_[j])
            {
                Bernoulli_dist r_theta(entity_counts(d) / (j+1));
                theta(j,d) = kjb::sample(r_theta);
            }
            else
            {
                theta(j,d) = 1;
            }
            entity_counts(d) += theta(j,d);
        }
    }
    for (size_t d = 0; d < dimension_prior; d++)
    {
        Beta_dist r_mu(entity_counts(d), 1 + J() - entity_counts(d));
        mu(d) = kjb::sample(r_mu);
    }
    mu_star2_ = get_minimums(mu_, &mu_star_);
    num_active_states_ = mu_.size();
}

void Infinite_state_model::initialize_resources()
{
    PM(verbose_ > 0, "Initialize resources for infinite state model.\n");
    Base_class::initialize_resources();
}

double mu_log_density(double mu, Infinite_state_model* model)
{
    int J_ = model->J();
    double sum_part_ = 0;
    for (size_t j = 1; j <= J_; j++)
    {
        sum_part_ += pow(1 - mu, j) / j;
    }
    return sum_part_ + (model->alpha() - 1) * log(mu) + J_ * log(1 - mu);
}

void Infinite_state_model::sample_inactive_states()
{
    PM(verbose_ > 0, "Get Slice Sampler...");
    kjb::Uniform_distribution rs(0.0, mu_star_);
    s_ = kjb::sample(rs);
    PM(verbose_ > 0, "Sample inactive states by using ARS...");
    num_inactive_states_ = 0;
    bool continue_ = true;
    double xl = 0.0, xr = mu_star_;
    while (continue_)
    {
        double new_mu_ = xr, old_mu_ = xr;
        double (*fp) (double, void*) = (double (*)(double, void*)) &mu_log_density;
        arms_simple(4, &xl, &xr, fp, this, 0, &old_mu_, &new_mu_);
        if (new_mu_ < s_)
        {
            continue_ = false;
        }
        else
        {
            mu_.push_back(new_mu_);
            entity_counts_.push_back(0);
            theta_.insert_zero_column((int)(num_active_states_ + num_inactive_states_));
            emission_model()->insert_latent_dimension(0, (int)(num_active_states_ + num_inactive_states_));
            num_inactive_states_ += 1;
            xr = new_mu_;
        }
    }
    PM(verbose_ > 0, "done.\n");
    D_prime_ = theta_.get_num_cols();
}

void Infinite_state_model::update_params()
{
    PM(verbose_ > 0, "Updating state parameters...\n");
    sample_inactive_states();
    update_theta_();
    update_mu_();
}

void Infinite_state_model::add_ground_truth_eval_header() const
{
    //// Doesn't make a lot of sense to do this here.
}

void Infinite_state_model::compare_state_sequence_to_ground_truth(
    const std :: string &,
    const std :: string &) const
{
    //// Doesn't make a lot of sense to do this here.
}

void Infinite_state_model::update_theta_()
{
    // or ask a way to merge 2 kjb vector together? we can converge a std::vector to kjb::vector
    PM(verbose_ > 0, "Updating theta...");
    similarity_model()->sync_before_theta_update();
    PM(verbose_ > 0, "        Resampling theta values...");
    for (size_t j = 0; j < J(); ++j)
    {
        for (size_t d = 0; d < D_prime(); ++d)
        {
            update_theta_(j,d);
        }
    }
}

void Infinite_state_model::update_theta_(const size_t& j, const size_t& d)
{
    const int theta_old = get_theta(j,d);
    int theta_new;
    double zeta = compute_zeta_jd_(j,d);
    Bernoulli_dist r_theta(zeta);
    theta_new = kjb::sample(r_theta);
    if (theta_old == theta_new) return;
    theta(j,d) = theta_new;
    similarity_model()->sync_after_theta_update(j, d, theta_old, theta_new);
    entity_counts(d) += 2 * theta_new - 1;
}

double Infinite_state_model::compute_zeta_jd_(const size_t &j, const size_t &d)
{
    /// zeta_jd represents the probability that theta_jd = 1
    /// We first compute the log odds, and then apply the logistic transform
    // the prior piece, plus the likelihood component due to observed transitions
    //double prior_log_odds = log(mu(d) / (1 - mu(d)));
    double prior_log_odds = 0;
    if (entity_counts(d) == 0 && mu(d) < mu_star_)
    {
        prior_log_odds = log(mu(d)) - log(mu_star_);
    }
    if (entity_counts(d) == 1 && mu(d) == mu_star_)
    {
        prior_log_odds = log(mu_star2_) - log(mu_star_);
    }
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

void Infinite_state_model::update_mu_()
{
    Count_vector new_entity_counts_;
    Prob_vector new_mu_;
    int inactive_ = 0;
    for (size_t d = 0; d < D_prime(); d++)
    {
        if (entity_counts(d) == 0)
        {
            emission_model()->remove_latent_dimension((int)(d - inactive_));
            theta_.remove_column((int)(d - inactive_));
            inactive_ += 1;
        }
        else
        {
            new_entity_counts_.push_back(entity_counts(d));
            Beta_dist r_mu(entity_counts(d), 1 + J() - entity_counts(d));
            new_mu_.push_back(kjb::sample(r_mu));
        }
    }
    entity_counts_ = new_entity_counts_;
    mu_ = new_mu_;
    mu_star2_ = get_minimums(mu_, &mu_star_);
    num_active_states_ = mu_.size();
}

double get_minimums(Prob_vector v, double* min)
{
    int min_index;
    *min = v.min(&min_index);
    v[min_index] = 1;
    return (v.min(&min_index));
}

