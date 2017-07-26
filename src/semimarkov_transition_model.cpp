/* $Id: semimarkov_transition_model.cpp 21443 2017-06-28 18:13:50Z chuang $ */

/*!
 * @file semimarkov_transition_model.cpp
 *
 * @author Colin Dawson 
 */

#include "util.h"
#include "hdp_hmm_lt.h"
#include "semimarkov_transition_model.h"
#include <m_cpp/m_matrix.h>
#include <m_cpp/m_mat_view.h>
#include <l_cpp/l_index.h>
#include <prob_cpp/prob_pdf.h>
#include <prob_cpp/prob_util.h>
#include <boost/make_shared.hpp>
#include <algorithm>
#include <fstream>

Dynamics_model_ptr Semimarkov_transition_parameters::make_module() const
{
    return boost::make_shared<Semimarkov_transition_model>(this);
}

/*
Likelihood_matrix Semimarkov_transition_model::log_likelihood_matrix(
    const kjb::Index_range& states,
    const kjb::Index_range& times
    )
{
    return parent->log_likelihood_matrix(states, times);
}
 */

Likelihood_matrix cumulative_likelihoods(
    const Likelihood_matrix& llm
    )
{
    const int T = llm.get_num_rows() - 1;
    const int J = llm.get_num_cols();
    Likelihood_matrix result(T+1,J);
    Likelihood_vector curr_total(J, 0.0);
    for(size_t t = T; t > 0; --t)
    {
        curr_total += llm.get_row(t);
        result.set_row(t, curr_total);
    }
    result.set_row(0, curr_total);
    return result;
}

void Semimarkov_transition_model::pass_messages_backwards_(
        Prob&                    log_marginal_likelihood_i,
        const size_t&            i,
        const Likelihood_matrix& llm
        )
{
    /*
     pass messages backwards in the HSMM (same idea to the message forward passing in HMM)
     Also to update the log marginal likelihood
     */
    typedef kjb::Index_range Range;
    const Likelihood_matrix cum_llm = cumulative_likelihoods(llm);
    const size_t T = llm.get_num_rows() - 1;
    B_.at(i).set_row(T, kjb::Vector((int) J(), 0.0));
    interval_likelihoods(i,T) = cum_llm.submatrix(T,0,1,J());
    for(int t = T; t > 0; --t)
    {
        kjb::Matrix message_matrix =
        duration_priors_.at(i)(Range(0, T-t, 1), Range::ALL)
        + interval_likelihoods(i,t)
        + B_.at(i)(Range(t, T, 1), Range::ALL);
        kjb::Vector curr_bstar_message =
        kjb::log_marginalize_over_rows(message_matrix);
        Bstar_.at(i).set_row(t, curr_bstar_message);
        kjb::Vector curr_b_message =
        kjb::log_marginalize_over_cols(
            kjb::shift_rows_by(A(), curr_bstar_message));
        B_.at(i).set_row(t-1, curr_b_message);
        kjb::Vector current_cum_likelihood(cum_llm(Range(t-1), Range::ALL));
        interval_likelihoods(i, t-1) =
        (-kjb::shift_rows_by(
                cum_llm(Range(t, T, 1), Range::ALL),
                -1 * current_cum_likelihood
                )).vertcat(kjb::create_row_matrix(current_cum_likelihood));
    }
    Prob_vector log_weighted_marginal_likelihood = pi0() + Bstar_.at(i).get_row(1);
    Bstar_.at(i).set_row(0, log_weighted_marginal_likelihood);
    log_marginal_likelihood_[i] =
    kjb::log_sum(
                 log_weighted_marginal_likelihood.begin(),
                 log_weighted_marginal_likelihood.end());
    log_marginal_likelihood_i = log_marginal_likelihood_[i];
}

void Semimarkov_transition_model::pass_test_messages_backwards_(
    Prob&                    test_log_marginal_likelihood,
    const size_t&            i, 
    const Likelihood_matrix& llm
    )
{
    KJB_THROW_2(kjb::Not_implemented, "test data marginal likelihood not yet implemented!");
}

/*
void Semimarkov_transition_model::pass_messages_backwards_(
    Prob&                    log_marginal_likelihood,
    const Likelihood_matrix& llm
    )
{
    typedef kjb::Index_range Range;
    const Likelihood_matrix cum_llm = cumulative_likelihoods(llm);
    const size_t T = llm.get_num_rows() - 1;
    B_.set_row(T, kjb::Vector((int) J(), 0.0));
    interval_likelihoods(T) = cum_llm.submatrix(T,0,1,J());
    for(int t = T; t > 0; --t)
    {
        // Set B*[t]  /// TODO: Handle censoring
        // the (d,j) element of message_matrix is the (log) product of 
        // the prior that state j lasts for d time steps given that it starts at time t+1,
        // the likelihood of the data assuming we are in state j during that time,
        // and the marginal likelihood of all of the data after that interval.
        kjb::Matrix message_matrix =
            duration_priors_(Range(0, T-t, 1), Range::ALL)
            + interval_likelihoods(t)
            + B_(Range(t, T, 1), Range::ALL);
        // By marginalizing down the rows, we get the B* message for
        // time t (likelihood of data from t on, assuming we begin a segment
        // of state j at time t, marginalizing over the durations of that segment
        kjb::Vector curr_bstar_message =
            kjb::log_marginalize_over_rows(message_matrix);
        // assert(curr_bstar_message.size() == J());
        // std::cerr << "B*_" << t << " = " << curr_bstar_message << std::endl;
        Bstar_.set_row(t, curr_bstar_message);
        // now set B[t]
        // By taking into account the probabilities of transitioning
        // to each state j' from each state j, we can get the joint of: the 
        // prior probability of going to state j' at t+1, and the marginal
        // likelihood of all the data from t+1 on given that we are in j'
        // for at least one time step.  Marginalizing over columns of
        // the JxJ matrix with these probabilities yields a vector whose
        // jth element is the marginal likelihood
        // of all of the data after time t given that we are in state j at
        // time t.
        kjb::Vector curr_b_message =
            kjb::log_marginalize_over_cols(
                kjb::shift_rows_by(A(), curr_bstar_message));
        // std::cerr << "B_" << t-1 << " = " << curr_b_message << std::endl;
        B_.set_row(t-1, curr_b_message);
        // for the next iteration
        kjb::Vector current_cum_likelihood(cum_llm(Range(t-1), Range::ALL));
        interval_likelihoods(t-1) =
            (-kjb::shift_rows_by(
                cum_llm(Range(t, T, 1), Range::ALL),
                -1 * current_cum_likelihood
                )).vertcat(kjb::create_row_matrix(current_cum_likelihood));
    }
    // Not completely consistent with the other rows of Bstar, but represents the
    // marginal posterior distribution for time 1
    Prob_vector log_weighted_marginal_likelihood = pi0() + Bstar_.get_row(1);
    Bstar_.set_row(0, log_weighted_marginal_likelihood);
    log_marginal_likelihood =
        kjb::log_sum(
            log_weighted_marginal_likelihood.begin(),
            log_weighted_marginal_likelihood.end());
}
 */

void Semimarkov_transition_model::sample_z_forwards_(const size_t& i)
{
    /*
     sample z forward, similar to sample z backward in HMM
     */
    typedef kjb::Index_range Range;
    kjb::Categorical_distribution<> r_z0(
        kjb::log_normalize_and_exponentiate(Bstar_.at(i).get_row(0)), 0
        );
    z(i, 1) = kjb::sample(r_z0);
    for(size_t t = 1; t < T(i);)
    {
        Prob_matrix duration_posterior =
        duration_priors_.at(i)(Range(0, T(i)-t, 1), Range(z(i,t)))
        + interval_likelihoods(i,t)(Range::ALL, Range(z(i,t)))
        + B_.at(i)(Range(t-1, T(i)-1, 1), Range(z(i,t)));
        kjb::Categorical_distribution<> r_d(
                kjb::log_normalize_and_exponentiate(Prob_vector(duration_posterior.at(i))), 0);
        size_t D = kjb::sample(r_d);
        for(size_t d = 1; d <= D; ++d)
        {
            // std::cerr << "t = " << t << ", d = " << d << std::endl;
            z(i, t+d) = z(i, t);
            d_totals(z(i, t))++;
            // std::cerr << "z(" << t+1 << ") = " << z(t+1) << std::endl;
        }
        t += D;
        if(t < T(i))
        {
            kjb::Categorical_distribution<> r_z(
                kjb::log_normalize_and_exponentiate(A().get_row(z(i,t)) + Bstar_.at(i).get_row(t+1)), 0
                                                );
            z(i, t+1) = kjb::sample(r_z);
            ++t;
            // std::cerr << "z(" << t+1 << ") = " << z(t+1) << std::endl;
        }
    }
}

/*
void Semimarkov_transition_model::sample_z_forwards_()
{
    typedef kjb::Index_range Range;
    kjb::Categorical_distribution<> r_z0(
        kjb::log_normalize_and_exponentiate(Bstar_.get_row(0)), 0
        );
    z(1) = kjb::sample(r_z0);
    for(size_t t = 1; t < T();)
    {
        // assert(interval_likelihoods(t).get_num_rows() == T() - t + 1);
        Prob_matrix duration_posterior =
            duration_priors_(Range(0, T()-t, 1), Range(z(t)))
            + interval_likelihoods(t)(Range::ALL, Range(z(t)))
            + B_(Range(t-1, T()-1, 1), Range(z(t)));
        // assert(duration_posterior.get_num_rows() == T() - t + 1);
        // assert(duration_posterior.get_num_cols() == 1);
        kjb::Categorical_distribution<> r_d(
            kjb::log_normalize_and_exponentiate(Prob_vector(duration_posterior)), 0);
        size_t D = kjb::sample(r_d);
        for(size_t d = 1; d <= D; ++d)
        {
            // std::cerr << "t = " << t << ", d = " << d << std::endl;
            z(t+d) = z(t);
            d_totals(z(t))++;
            // std::cerr << "z(" << t+1 << ") = " << z(t+1) << std::endl;
        }
        t += D;
        if(t < T())
        {
            kjb::Categorical_distribution<> r_z(
                kjb::log_normalize_and_exponentiate(A().get_row(z(t)) + Bstar_.get_row(t+1)), 0
                );
            z(t+1) = kjb::sample(r_z);
            ++t;
            // std::cerr << "z(" << t+1 << ") = " << z(t+1) << std::endl;
        }
    }
}
 */

void Semimarkov_transition_model::update_z_(const size_t& i, const Likelihood_matrix& llm)
{
    /*
     update all the axuliary variables and sample the state
     */
    // zero out all the counts
    std::fill(d_totals_.begin(), d_totals_.end(), 0);
    PM(verbose_ > 0, "Computing HSMM messages...");
    //std::cerr << "Computing HSMM messages...";
    pass_messages_backwards_(log_marginal_likelihood(i), i, llm);
    PM(verbose_ > 0, "done.\n");
    //std::cerr << "done." << std::endl;
    PM(verbose_ > 0, "Sampling z...");
    //std::cerr << "Sampling z...";
    sample_z_forwards_(i);
    PM(verbose_ > 0, "done.\n");
    //std::cerr << "done." << std::endl;
}

/*
void Semimarkov_transition_model::update_z_(const Likelihood_matrix& llm)
{
    // zero out all the counts
    std::fill(d_totals_.begin(), d_totals_.end(), 0);
    std::cerr << "Computing HSMM messages...";
    pass_messages_backwards_(log_marginal_likelihood(), llm);
    std::cerr << "done." << std::endl;
    std::cerr << "Sampling z...";
    sample_z_forwards_();
    std::cerr << "done." << std::endl;
}
 */

double Semimarkov_transition_model::get_log_marginal_likelihood(
        const size_t&            i,
        const Likelihood_matrix& llm
        )
{
    double result;
    pass_messages_backwards_(result, i, llm);
    return result;
}

double Semimarkov_transition_model::get_test_log_marginal_likelihood(
    const size_t&            i,
    const Likelihood_matrix& llm
    )
{
    double result;
    pass_test_messages_backwards_(result, i, llm);
    return result;
}

/*
double Semimarkov_transition_model::get_log_marginal_likelihood(
    const Likelihood_matrix& llm
    )
{
    double result;
    pass_messages_backwards_(result, llm);
    return result;
}

double Semimarkov_transition_model::get_test_log_marginal_likelihood(
    const Likelihood_matrix& llm
    )
{
    double result;
    pass_test_messages_backwards_(result, llm);
    return result;
}
 */

void Semimarkov_transition_model::update_duration_priors_()
{
    for(size_t j = 0; j < J(); ++j)
    {
        double shape = a_omega_ + d_totals(j);
        double scale = 1.0 / (b_omega_ + n_dot(j));
        Dur_param_dist r_omega(shape, scale);
        omega_[j] = kjb::sample(r_omega);
        Dur_dist r_d(omega_[j]);
        for (size_t i = 0; i < NF(); ++i)
        {
            for(size_t t = 0; t < T(i); ++t)
            {
                duration_priors_.at(i)(t,j) = kjb::log_pdf(r_d, t);
            }
        }
    }
}

/*
void Semimarkov_transition_model::update_duration_priors_()
{
    for(size_t j = 0; j < J(); ++j)
    {
        double shape = a_omega_ + d_totals(j);
        double scale = 1.0 / (b_omega_ + n_dot(j));
        // std::cerr << "        Sampling from G(" << shape << "," << scale << ")" << std::endl;
        Dur_param_dist r_omega(shape, scale);
        omega_[j] = kjb::sample(r_omega);
        Dur_dist r_d(omega_[j]);
        for(size_t t = 0; t < T(); ++t)
        {
            // the (t,j) element of the resulting matrix (D) corresponds to the prob
            // that state j lasts t+1 steps (so, NB, t = 0 is a duration of 1, but
            // this results from evaluating the corresponding dist at 0, not 1)
            // We can think of the duration random variable as giving the number of
            // *extra* time steps after the first.  In the geometric and neg bin case, this
            // corresponds to the total number of *trials* required for r successes to occur,
            // rather than the number of failures
            duration_priors_(t,j) = kjb::log_pdf(r_d, t);
            // std::cerr << "            duration_priors(" << t << "," << j << ")"
            //           << " set to " << duration_priors_(t,j) << std::endl;
        }
    }
}
 */

void Semimarkov_transition_model::mask_A_()
{
    for (size_t j = 0; j < J(); ++j)
    {
        A(j,j) = log(0.0);
    }
}

void Semimarkov_transition_model::initialize_resources()
{
    Base_class::initialize_resources();
    PM(verbose_ > 0, "Allocating omega...");
    //std::cerr << "    Allocating omega...";
    omega_ = Time_array(J(), 0.0);
    PM(verbose_ > 0, "done.\n");
    //std::cerr << "done." << std::endl;
    PM(verbose_ > 0, "    Allocating duration priors and message matrices...");
    //std::cerr << "    Allocating duration priors and message matrices...";
    d_totals_ = Count_vector(J(), 0);
    duration_priors_ = Prob_matrix_list(NF());
    B_ = Prob_matrix_list(NF());
    Bstar_ = Prob_matrix_list(NF());
    interval_likelihoods_ = Prob_matrix_list_list(NF());
    for (size_t i = 0; i < NF(); i++)
    {
        duration_priors_.at(i) = Prob_matrix(T(i), J(), 0.0);
        B_.at(i) = Prob_matrix(T(i)+1, J(), 0.0);
        Bstar_.at(i) = Prob_matrix(T(i)+1, J(), 0.0);
        interval_likelihoods_.at(i) = Prob_matrix_list(T(i)+1, Prob_matrix(T(i)+1, J(), 0.0));
    }
    PM(verbose_ > 0, "done.\n");
    //std::cerr << "done." << std::endl;
}

/*
void Semimarkov_transition_model::initialize_resources()
{
    Base_class::initialize_resources();
    std::cerr << "    Allocating omega...";
    omega_ = Time_array(J(), 0.0);
    std::cerr << "done." << std::endl;
    std::cerr << "    Allocating duration priors...";
    d_totals_ = Count_vector(J(), 0);
    duration_priors_ = Prob_matrix(T(), J(), 0.0);
    std::cerr << "done." << std::endl;
    std::cerr << "    Allocating message matrices...";
    B_ = Prob_matrix(T() + 1, J(), 0.0);
    Bstar_ = Prob_matrix(T() + 1, J(), 0.0);
    interval_likelihoods_ =
        std::vector<Prob_matrix>(
            T() + 1, Prob_matrix(T() + 1, J(), 0.0));
    std::cerr << "done." << std::endl;
}
 */

/*
void Semimarkov_transition_model::initialize_params()
{
    Base_class::initialize_params();
    update_duration_priors_();
    sample_labels_from_prior();
}
 */
void Semimarkov_transition_model::initialize_params()
{
    Base_class::initialize_params();
    update_duration_priors_();
    sample_labels_from_prior(T());
}

void Semimarkov_transition_model::update_params(const size_t& i, const Likelihood_matrix& llm)
{
    Base_class::update_params(i, llm);
    update_duration_priors_();
    update_z_(i, llm);
}

/*
void Semimarkov_transition_model::update_params(const Likelihood_matrix& llm)
{
    Base_class::update_params(llm);
    update_duration_priors_();
    update_z_(llm);
}
 */

void Semimarkov_transition_model::set_up_results_log() const
{
    Base_class::set_up_results_log();
    std::ofstream ofs;
    ofs.open(write_path + "omega.txt", std::ofstream::out);
    ofs << "iteration value" << std::endl;
    ofs.close();
    ofs.open(write_path + "dtotals.txt", std::ofstream::out);
    ofs << "iteration value" << std::endl;
    ofs.close();
}

/*
void Semimarkov_transition_model::input_previous_results(const std::string& name)
{
    Base_class::input_previous_results(name);
    std::cerr << "Inputting information for omega_..." << std::endl
    omega_ = input_to_vector<Cts_duration>(write_path, "omega.txt", name) //std vector
    std::cerr << "done!" << std::endl;
    std::cerr << "Inputting information for d_totals_..." << std::endl
    d_totals_ = input_to_vector<Count>(write_path, "dtotals.txt", name) //std vector
    std::cerr << "done!" << std::endl;
}
*/

void Semimarkov_transition_model::write_state_to_file(const std::string& name) const
{
    Base_class::write_state_to_file(name);
    std::ofstream ofs;
    ofs.open(write_path + "omega.txt", std::ofstream::out | std::ofstream::app);
    ofs << name << " " << omega_ << std::endl;
    ofs.close();
    ofs.open(write_path + "dtotals.txt", std::ofstream::out | std::ofstream::app);
    ofs << name << " " << d_totals_ << std::endl;
    ofs.close();
}

void Semimarkov_transition_model::sample_labels_from_prior(const T_list& T)
{
    PM(verbose_ > 0, "    Initializing z...");
    //std::cerr << "    Initializing z...";
    std::vector<kjb::Categorical_distribution<> > row_dists;
    for(size_t j = 0; j < J(); ++j)
    {
        row_dists.push_back(
            kjb::Categorical_distribution<>(ew_exponentiate(A().get_row(j)), 0));
    }
    for (size_t i = 0; i < T.size(); ++i)
    {
        z(i,0) = J();
        z(i,1) = kjb::sample(
                    kjb::Categorical_distribution<>(
                        kjb::log_normalize_and_exponentiate(pi0()), 0));
        for (size_t t = 1; t < T[i]; ++t)
        {
            kjb::Categorical_distribution<> r_d(
                kjb::log_normalize_and_exponentiate(duration_priors_[i].get_col(z(i, t))), 0
                    );
            for (int d = kjb::sample(r_d); d > 0 && t < T[i]; --d, ++ t)
            {
                z(i, t+1) = z(i, t);
                d_totals(z(i,t))++;
            }
            if (t < T[i])
            {
                z(i, t+1) = kjb::sample(row_dists[z(i, t)]);
            }
        }
    }
    PM(verbose_ > 0, "done.\n");
    //std::cerr << "done." << std::endl;
}

/*
void Semimarkov_transition_model::sample_labels_from_prior()
{
    std::cerr << "    Initializing z...";
    std::vector<kjb::Categorical_distribution<> > row_dists;
    for(size_t j = 0; j < J(); ++j)
    {
        row_dists.push_back(
                            kjb::Categorical_distribution<>(ew_exponentiate(A().get_row(j)), 0));
    }
    for (size_t i = 0; i < NF(); ++i)
    {
        z(i,0) = J();
        z(i,1) = kjb::sample(
                             kjb::Categorical_distribution<>(
                                                             kjb::log_normalize_and_exponentiate(pi0()), 0));
        for (size_t t = 1; t < T(i); ++t)
        {
            kjb::Categorical_distribution<> r_d(
                                                kjb::log_normalize_and_exponentiate(duration_priors_[i].get_col(z(i, t))), 0
                                                );
            for (int d = kjb::sample(r_d); d > 0 && t < T(i); --d, ++ t)
            {
                z(i, t+1) = z(i, t);
                d_totals(z(i,t))++;
            }
            if (t < T(i))
            {
                z(i, t+1) = kjb::sample(row_dists[z(i, t)]);
            }
        }
    }
    std::cerr << "done." << std::endl;
}
*/

/*
void Semimarkov_transition_model::sample_labels_from_prior()
{
    std::cerr << "    Initializing z...";
    z_ = State_sequence(T() + 1);
    std::vector<kjb::Categorical_distribution<> > row_dists;
    for(size_t j = 0; j < J(); ++j)
    {
        row_dists.push_back(
            kjb::Categorical_distribution<>(ew_exponentiate(A().get_row(j)), 0));
    }
    z(0) = J();
    z(1) =
        kjb::sample(
            kjb::Categorical_distribution<>(
                kjb::log_normalize_and_exponentiate(pi0()), 0));
    for(size_t t = 1; t < T(); ++t)
    {
        kjb::Categorical_distribution<> r_d(
            kjb::log_normalize_and_exponentiate(duration_priors_.get_col(z(t))), 0
            );
            
        for(int d = kjb::sample(r_d); d > 0 && t < T(); --d, ++t)
        {
            z(t+1) = z(t);
            // z(t+1) = 0;
            d_totals(z(t))++;
        }
        if(t < T())
        {
            z(t+1) = kjb::sample(row_dists[z(t)]);
        }
    }
    // std::cerr << "        z = " << z_ << std::endl;
    std::cerr << "done." << std::endl;
}
 */

