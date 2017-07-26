/* $Id: markov_transition_model.cpp 21443 2017-06-28 18:13:50Z chuang $ */

/*!
 * @file markov_transition_model.cpp
 *
 * @author Colin Dawson 
 */

#include "markov_transition_model.h"
#include "state_model.h"
#include "util.h"
#include <boost/make_shared.hpp>

Dynamics_model_ptr Markov_transition_parameters::make_module() const
{
    return boost::make_shared<Markov_transition_model>(this);
}

void Markov_transition_model::initialize_resources()
{
    /*
     initializing auxiliary variable for general HMM state sampling
     or Beam sampling (Gael, Saatci, Teh, and Ghahramani paper)
     NF: number of train files
     log_marginal_likelihood is a vector with NF elements
     messages is a vector with NF elements each is a Ti by J matrix
     slice_variables_, slice_indicies_ are auxiliary variable for Beam Sampling
     */
    Base_class::initialize_resources();
    PM(verbose_ > 0, "     Allocating HMM message matrix...");
    //std::cerr << "    Allocating HMM message matrix...";
    log_marginal_likelihood_ = Prob_vector((int) NF(), 0.0);
    messages_ = Message_list(NF());
    slice_variables_ = Prob_list(NF());
    slice_indices_ = Index_list_list_list(NF());
    for (int i = 0; i < NF(); i++)
    {
        messages_.at(i) = Message_matrix(T(i) + 1, Message_vector(static_cast<int>(J()), 0.0));
        slice_variables_.at(i) = Prob_vector(T(i) + 1);
        slice_indices_.at(i) = Index_list_list(T(i) + 1);
    }
    PM(verbose_ > 0, "done.\n");
}

/*
void Markov_transition_model::initialize_resources()
{
    Base_class::initialize_resources();
    std::cerr << "    Allocating HMM message matrix...";
    messages_ = Message_matrix(T() + 1, Message_vector(static_cast<int>(J()), 0.0));
    std::cerr << "done." << std::endl;
    slice_variables_ = Prob_vector(T() + 1);
    slice_indices_ = Index_list_list(T() + 1);
}
 */

void Markov_transition_model::initialize_params()
{
    /*
     initialize the parameters, state label, etc.
     */
    Base_class::initialize_params();
    sample_labels_from_prior(T());
}

/*
void Markov_transition_model::initialize_params()
{
    Base_class::initialize_params();
    sample_labels_from_prior();
}
 */

void Markov_transition_model::update_params(const size_t& i, const Likelihood_matrix& llm)
{
    /*
     update parameters state parameters and other auxiliary variable
     recalculate the new train log likelihood
     */
    Base_class::update_params(i, llm);
    update_z_(i, llm);
}

/*
void Markov_transition_model::update_params(const Likelihood_matrix& llm)
{
    Base_class::update_params(llm);
    update_z_(llm);
}
 */

double Markov_transition_model::get_log_marginal_likelihood(
    const size_t& i,
    const Likelihood_matrix& llm
    )
{
    /*
     calcualte the train log likelihood by passing message forward
     */
    double result;
    pass_messages_forward_(result, i, llm);
    return result;
}

double Markov_transition_model::get_test_log_marginal_likelihood(
    const size_t& i,
    const Likelihood_matrix& llm
    )
{
    /*
     calculate the log lieklihood by passing message forward
     */
    double result;
    pass_test_messages_forward_(result, i, llm);
    return result;
}

/*
double Markov_transition_model::get_log_marginal_likelihood(const Likelihood_matrix& llm)
{
    double result;
    pass_messages_forward_(result, llm);
    return result;
}
 
double Markov_transition_model::get_test_log_marginal_likelihood(const Likelihood_matrix& llm)
{
    double result;
    pass_test_messages_forward_(result, llm);
    return result;
}
 */

void Markov_transition_model::sample_labels_from_prior(const T_list& T)
{
    /*
     sample prior state by A_ and pi0_ from transitiion model,
     S0 based on pi0_ and other states based on A_
     */
    PM(verbose_ > 0, "    Initiailzing z...");
    //std::cerr << "    Initializing z...";
    std::vector<kjb::Categorical_distribution<> > row_dists;
    for(size_t j = 0; j < J(); ++j)
    {
        row_dists.push_back(
            kjb::Categorical_distribution<>(kjb::ew_exponentiate(A().get_row(j)), 0));
    }
    for (size_t i = 0; i < T.size(); ++i)
    {
        z(i,0) = J();
        z(i,1) = kjb::sample(
                    kjb::Categorical_distribution<>(
                        kjb::log_normalize_and_exponentiate(pi0()), 0));
        for (size_t t = 2; t <= T[i]; ++t)
        {
            z(i,t) = kjb::sample(row_dists[z(i,t-1)]);
        }
    }
    PM(verbose_ > 0, "done.\n");
    //std::cerr << "done." << std::endl;
}

/*
void Markov_transition_model::sample_labels_from_prior()
{
    std::cerr << "    Initializing z...";
    std::vector<kjb::Categorical_distribution<> > row_dists;
    for(size_t j = 0; j < J(); ++j)
    {
        row_dists.push_back(
                            kjb::Categorical_distribution<>(kjb::ew_exponentiate(A().get_row(j)), 0));
    }
    for (size_t i = 0; i < NF(); ++i)
    {
        z(i,0) = J();
        z(i,1) = kjb::sample(
                             kjb::Categorical_distribution<>(
                                                             kjb::log_normalize_and_exponentiate(pi0()), 0));
        for (size_t t = 2; t <= T(i); ++t)
        {
            z(i,t) = kjb::sample(row_dists[z(i,t-1)]);
        }
    }
    std::cerr << "done." << std::endl;
}
*/

/*
void Markov_transition_model::sample_labels_from_prior()
{
    std::cerr << "    Initializing z...";
    z_ = State_sequence(T() + 1);
    std::vector<kjb::Categorical_distribution<> > row_dists;
    for(size_t j = 0; j < J(); ++j)
    {
        row_dists.push_back(
            kjb::Categorical_distribution<>(kjb::ew_exponentiate(A().get_row(j)), 0));
    }
    z(0) = J();
    z(1) =
        kjb::sample(
            kjb::Categorical_distribution<>(
                kjb::log_normalize_and_exponentiate(pi0()), 0));
    for(size_t t = 2; t <= T(); ++t)
    {
        z(t) = kjb::sample(row_dists[z(t-1)]);
    }
    // std::cerr << "z = " << z_ << std::endl;
    std::cerr << "done." << std::endl;
}
 */

void Markov_transition_model::update_z_(const size_t& i, const Likelihood_matrix& llm)
{
    /*
     decide on which method to update_z: beam or general
     */
    PMWP(verbose_ > 0, "    Updating z for sequence %d using %s sampling...\n",
         (i)(sampling_method.c_str()));
    //std::cerr << "    Updating z for sequence " << i << " using "
    //          << sampling_method << " sampling..."
    //          << std::endl;
    // std::cerr << "    Log_likelihood matrix = " << std::endl;
    // std::cerr << llm << std::endl;
    if(sampling_method == "weak_limit_with_beam")
    {
        update_z_beamily_(i);
    } else {
        pass_messages_forward_(log_marginal_likelihood(i), i, llm);
        sample_z_backwards_(i);
    }
}

/*
void Markov_transition_model::update_z_(const Likelihood_matrix& llm)
{
    std::cerr << "    Updating z using " << sampling_method << " sampling..."
              << std::endl;
    // std::cerr << "    Log_likelihood matrix = " << std::endl;
    // std::cerr << llm << std::endl;
    if(sampling_method == "weak_limit_with_beam")
    {
        update_z_beamily_();
    } else {
        pass_messages_forward_(log_marginal_likelihood(), llm);
        sample_z_backwards_();
    }
}
 */

void Markov_transition_model::update_z_beamily_(const size_t& i)
{
    /*
     Get thershold for slicing in each transition
     Check Beam Sampling paper
     */
    PM(verbose_ > 0, "     Sampling slice variables...");
    //std::cerr << "        Sampling slice variables...";
    kjb::Uniform_distribution r_slice0(0.0, exp(pi0(z(i,1))));
    slice_variables_[i][1] = log(kjb::sample(r_slice0));
    for(size_t t = 1; t < T(i); ++t)
    {
        kjb::Uniform_distribution r_slice(0.0, exp(A(z(i,t), z(i,t+1))));
        slice_variables_[i][t+1] = log(kjb::sample(r_slice));
    }
    PM(verbose_ > 0, "done.\n");
    //std::cerr << "done." << std::endl;
    pass_messages_forward_beamily_(log_marginal_likelihood(i), i);
    sample_z_backwards_beamily_(i);
}

/*
void Markov_transition_model::update_z_beamily_()
{
    // std::cerr << "    Log_likelihood matrix = " << std::endl;
    // std::cerr << llm << std::endl;
    std::cerr << "        Sampling slice variables...";
    //           << std::endl;
    // std::cerr << "            pi0 = " << pi0() << std::endl;
    // std::cerr << "            pi0[z_1] = " << pi0(z(1)) << std::endl;
    // std::cerr << "            exp(pi0[z_1]) = " << exp(pi0(z(1))) << std::endl;
    kjb::Uniform_distribution r_slice0(0.0, exp(pi0(z(1))));
    slice_variables_[1] = log(kjb::sample(r_slice0));
    for(size_t t = 1; t < T(); ++t)
    {
        kjb::Uniform_distribution r_slice(0.0, exp(A(z(t), z(t+1))));
        slice_variables_[t+1] = log(kjb::sample(r_slice));
    }
    std::cerr << "done." << std::endl;
    pass_messages_forward_beamily_(log_marginal_likelihood());
    sample_z_backwards_beamily_();
}
 */

void Markov_transition_model::pass_messages_forward_(
     Prob&                    log_marginal_likelihood_i,
     const size_t&            i,
     const Likelihood_matrix& llm
     )
{
    /*
     Pass Messages Forward to get train log likelihood and update necessary auxiliary variable
     the forward algorithm in log-form
     */
    PMWP(verbose_ > 0,
         "         Passing messages forward using weak limit truncation for sequence %d...", (i));
    //std::cerr << "        Passing messages forward using weak limit truncation for sequence " << i << " ...";
    Prob_vector pztp1_x1t(pi0());
    for (size_t t = 1; t <= T(i); ++t)
    {
        Message_vector p_zt_x1t = llm.get_row(t) + pztp1_x1t;
        messages(i, t) = p_zt_x1t; // this gets used for backward sampling
        Prob_matrix p_ztp1_zt_x1t = kjb::shift_columns_by(A(), p_zt_x1t);
        pztp1_x1t = kjb::log_marginalize_over_rows(p_ztp1_zt_x1t);
    }
    log_marginal_likelihood_[i] = kjb::log_sum(pztp1_x1t.begin(), pztp1_x1t.end());
    log_marginal_likelihood_i = log_marginal_likelihood_[i];
    PM(verbose_ > 0, "done.\n");
    //std::cerr << "done." << std::endl;
}

void Markov_transition_model::pass_test_messages_forward_(
    Prob&                    test_log_marginal_likelihood_i,
    const size_t&            i,
    const Likelihood_matrix& llm
    )
{
    /*
     Pass test message forward to get the test log likelihood
     */
    PMWP(verbose_ > 0,
         "       Passing test messages forward using weak limit truncation for sequence %d ...\n", (i));
    //std::cerr << "        Passing test messages forward using weak limit truncation..." << i << "...";
    Prob_vector pztp1_x1t(pi0());
    for(size_t t = 1; t <= test_T(i); ++t)
    {
        // p_zt_x1t = p(z_t | x_{1:t})
        // llm[t,] = p(x_t | z_t) for each z_t
        // pztp1_x1t = p(z_{t+1} | x_{1:t}
        // (at this stage pztp1_x1t has t = t-1)
        Message_vector p_zt_x1t = llm.get_row(t) + pztp1_x1t;
        Prob_matrix p_ztp1_zt_x1t = kjb::shift_columns_by(A(), p_zt_x1t);
        pztp1_x1t = kjb::log_marginalize_over_rows(p_ztp1_zt_x1t);
    }
    // std::cerr << "done." << std::endl;
    test_log_marginal_likelihood_i = kjb::log_sum(pztp1_x1t.begin(), pztp1_x1t.end());
    PM(verbose_ > 0, "done.\n");
    //std::cerr << "done." << std::endl;
}

/*
void Markov_transition_model::pass_messages_forward_(
    Prob&                    log_marginal_likelihood,
    const Likelihood_matrix& llm
    )
{
    std::cerr << "        Passing messages forward using weak limit truncation...";
    Prob_vector pztp1_x1t(pi0());
    for(size_t t = 1; t <= T(); ++t)
    {
        // p_zt_x1t = p(z_t | x_{1:t})
        // llm[t,] = p(x_t | z_t) for each z_t
        // pztp1_x1t = p(z_{t+1} | x_{1:t}
        // (at this stage pztp1_x1t has t = t-1)
        Message_vector p_zt_x1t = llm.get_row(t) + pztp1_x1t;
        messages(t) = p_zt_x1t; // this gets used for backward sampling
        Prob_matrix p_ztp1_zt_x1t = kjb::shift_columns_by(A(), p_zt_x1t);
        pztp1_x1t = kjb::log_marginalize_over_rows(p_ztp1_zt_x1t);
    }
    // std::cerr << "done." << std::endl;
    log_marginal_likelihood = kjb::log_sum(pztp1_x1t.begin(), pztp1_x1t.end());
    std::cerr << "done." << std::endl;
}
 
void Markov_transition_model::pass_test_messages_forward_(
    Prob&                    test_log_marginal_likelihood,
    const Likelihood_matrix& llm
    )
{
    std::cerr << "        Passing messages forward using weak limit truncation...";
    Prob_vector pztp1_x1t(pi0());
    for(size_t t = 1; t <= test_T(); ++t)
    {
        // p_zt_x1t = p(z_t | x_{1:t})
        // llm[t,] = p(x_t | z_t) for each z_t
        // pztp1_x1t = p(z_{t+1} | x_{1:t}
        // (at this stage pztp1_x1t has t = t-1)
        Message_vector p_zt_x1t = llm.get_row(t) + pztp1_x1t;
        Prob_matrix p_ztp1_zt_x1t = kjb::shift_columns_by(A(), p_zt_x1t);
        pztp1_x1t = kjb::log_marginalize_over_rows(p_ztp1_zt_x1t);
    }
    // std::cerr << "done." << std::endl;
    test_log_marginal_likelihood = kjb::log_sum(pztp1_x1t.begin(), pztp1_x1t.end());
    std::cerr << "done." << std::endl;
 }
 */

void Markov_transition_model::pass_messages_forward_beamily_(
    Prob&                    log_marginal_likelihood_i,
    const size_t&            i
    )
{
    /*
     Pass message forward using beam sampling, refer to the beam sampling paper,
     and update necessary auxiliary variable for backward sampling
     */
    PM(verbose_ > 0, "       Passing messages forward conditioned on slice vars...");
    //std::cerr << "        Passing messages forward conditioned on slice vars...";
    slice_indices_[i][1] = Index_list();
    for(size_t j = 0; j < J(); ++j)
    {
        if(pi0(j) > slice_variables_[i][1])
        {
            slice_indices_[i][1].push_back(j);
        }
    }
    kjb::Index_range indices(slice_indices_[i][1]);
    Message_vector p_zt_x1t =
    log_likelihood_matrix(i, indices, kjb::Index_range(1)).get_row(1);
    messages(i, 1) = p_zt_x1t; // this gets used for backward sampling
    for(size_t t = 2; t <= T(i); ++t)
    {
        const size_t J_slice = indices.size();
        slice_indices_[i][t] = Index_list();
        kjb::Const_matrix_view A_slice = A()(indices, kjb::Index_range::ALL);
        Prob_vector p_zt_x1tm1;             // p(z_t | x_{1:t-1})
        for(size_t jp = 0; jp < J(); ++jp)
        {
            Prob_vector p_zt_ztm1j_x1tm1;  // p(z_t, z_{t-1} = j | x_{1:t-1})
            bool use_jp = false;
            for(size_t j = 0; j < J_slice; ++j)
            {
                if(A_slice(j,jp) > slice_variables_[i][t])
                {
                    use_jp = true;
                    p_zt_ztm1j_x1tm1.push_back(p_zt_x1t[j]);
                }
            }
            if(use_jp)
            {
                p_zt_x1tm1.push_back(
                    kjb::log_sum(p_zt_ztm1j_x1tm1.begin(), p_zt_ztm1j_x1tm1.end()));
                slice_indices_[i][t].push_back(jp);
            }
        }
        indices = kjb::Index_range(slice_indices_[i][t]);
        p_zt_x1t =
        log_likelihood_matrix(i, indices, kjb::Index_range(t)).get_row(1)
        + p_zt_x1tm1;
        messages(i,t) = p_zt_x1t; // this gets used for backward sampling
    }
    log_marginal_likelihood_[i] = kjb::log_sum(p_zt_x1t.begin(), p_zt_x1t.end());
    log_marginal_likelihood_i = log_marginal_likelihood_[i];
    PM(verbose_ > 0, "done.\n");
    //std::cerr << "done." << std::endl;
}

/*
void Markov_transition_model::pass_messages_forward_beamily_(
    Prob&                    log_marginal_likelihood
    )
{
    std::cerr << "        Passing messages forward conditioned on slice vars...";
    slice_indices_[1] = Index_list();
    // std::cerr << "        Passing messages forward for beam sampling..." << std::endl;
    //// filter pi0 to contain only elts > w_0
    // std::cerr << "            Filtering initial distribution...";
    for(size_t j = 0; j < J(); ++j)
    {
        if(pi0(j) > slice_variables_[1])
        {
            slice_indices_[1].push_back(j);
        }
    }
    // std::cerr << "done." << std::endl;
    kjb::Index_range indices(slice_indices_[1]);
    //// select elts of llm[t] corresp. to nonzero elts of pztp1_x1t
    // std::cerr << "            t=1; computing message t..."
    //           << std::endl;
    // // std::cerr << "            Full likelihood vector = "
    // //           << llm.get_row(t) << std::endl;
    // std::cerr << "            Slice indices = "
    //           << slice_indices_[1] << std::endl;
    // std::cerr << "            Filtered likelihood vector = "
    //           << llm.get_row(1)[indices] << std::endl;
    Message_vector p_zt_x1t =
        log_likelihood_matrix(indices, kjb::Index_range(1)).get_row(1);
    messages(1) = p_zt_x1t; // this gets used for backward sampling
    // std::cerr << "            complete message = "
    //           << messages(1) << std::endl;
    // std::cerr << "done." << std::endl;
    for(size_t t = 2; t <= T(); ++t)
    {
        const size_t J_slice = indices.size();
        // p_zt_x1t = p(z_t | x_{1:t})
        // llm[t,] = p(x_t | z_t) for each z_t
        // pztp1_x1t = p(z_{t+1} | x_{1:t}
        // (at this stage pztp1_x1t has t = t-1)

        //// select rows of A() corresp. to rows of p_zt_x1t
        //// then for each entry a_jj' > w_t (where j in elts p_zt_x1t),
        //// multiply p_zt_x1t[j] by a_jj'
        slice_indices_[t] = Index_list();
        kjb::Const_matrix_view A_slice = A()(indices, kjb::Index_range::ALL);
        // std::cerr << "A_slice = " << std::endl;
        // std::cerr << A_slice << std::endl;
        // std::cerr << "    Slice variable = " << slice_variables_[t] << std::endl;
        Prob_vector p_zt_x1tm1;             // p(z_t | x_{1:t-1})
        for(size_t jp = 0; jp < J(); ++jp)
        {
            Prob_vector p_zt_ztm1j_x1tm1;  // p(z_t, z_{t-1} = j | x_{1:t-1})
            bool use_jp = false;
            for(size_t j = 0; j < J_slice; ++j)
            {
                // std::cerr << "    Log transition prob "
                //           << j << "," << jp << "] = " << A_slice(j,jp) << std::endl;
                if(A_slice(j,jp) > slice_variables_[t])
                {
                    use_jp = true;
                    p_zt_ztm1j_x1tm1.push_back(p_zt_x1t[j]);
                }
            }
            if(use_jp)
            {
                p_zt_x1tm1.push_back(
                    kjb::log_sum(p_zt_ztm1j_x1tm1.begin(), p_zt_ztm1j_x1tm1.end()));
                slice_indices_[t].push_back(jp);
            }
        }
        indices = kjb::Index_range(slice_indices_[t]);
        //// select elts of llm[t] corresp. to nonzero elts of pztp1_x1t
        // std::cerr << "            t=" << t << ": computing message t..."
        //           << std::endl;
        // // std::cerr << "            Full likelihood vector = "
        // //           << llm.get_row(t) << std::endl;
        // std::cerr << "            Slice indices = "
        //           << slice_indices_[t] << std::endl;
        // std::cerr << "            Filtered likelihood vector = "
        //           << llm.get_row(t)[indices] << std::endl;
        // std::cerr << "            prior message component = "
        //           << p_zt_x1tm1 << std::endl;
        p_zt_x1t =
            log_likelihood_matrix(indices, kjb::Index_range(t)).get_row(1)
            + p_zt_x1tm1;
        messages(t) = p_zt_x1t; // this gets used for backward sampling
        // std::cerr << "            complete message = "
        //           << messages(t) << std::endl;
        // std::cerr << "done." << std::endl;
    }
    std::cerr << "done." << std::endl;
    log_marginal_likelihood = kjb::log_sum(p_zt_x1t.begin(), p_zt_x1t.end());
}
 */

void Markov_transition_model::sample_z_backwards_(const size_t& i)
{
    /*
     sample state z backward, refer to HMM paper
     backward sampling method
     */
    PM(verbose_ > 0, "        Sampling z backward using weak limit truncation...\n");
    //std::cerr << "        Sampling z backward using weak limit truncation...";
    Prob_vector next_potential(static_cast<int>(J()), 0.0);
    for(int t = T(i); t > 0; --t)
    {
        next_potential += messages(i,t); // this is p(z_t | x_{1:T})
        kjb::Categorical_distribution<> r_z(
                kjb::log_normalize_and_exponentiate(next_potential), 0
                );
        z(i,t) = kjb::sample(r_z);
        // we want p(z_{t-1} | x_{1:T}, z_t)
        // \propto p(z_{t-1} | x_{1:{t-1}}) p(z_{t} | z_{t-1}) p(x_{t:T} | z_t)
        // \propto p(z_{t-1} | x_{1:{t-1}}) p(z_{t} | z_{t-1})
        kjb::Index_range indices(slice_indices_.at(i)[t]);
        next_potential = kjb::log_normalize(A().get_col(z(i,t)));
    }
    PM(verbose_ > 0, "done.\n");
    //std::cerr << "done." << std::endl;
}

/*
void Markov_transition_model::sample_z_backwards_()
{
    std::cerr << "        Sampling z backward using weak limit truncation...";
    Prob_vector next_potential(static_cast<int>(J()), 0.0);
    for(int t = T(); t > 0; --t)
    {
        next_potential += messages(t); // this is p(z_t | x_{1:T})
        kjb::Categorical_distribution<> r_z(
            kjb::log_normalize_and_exponentiate(next_potential), 0
            );
        z(t) = kjb::sample(r_z);
        // we want p(z_{t-1} | x_{1:T}, z_t)
        // \propto p(z_{t-1} | x_{1:{t-1}}) p(z_{t} | z_{t-1}) p(x_{t:T} | z_t)
        // \propto p(z_{t-1} | x_{1:{t-1}}) p(z_{t} | z_{t-1}) 
        kjb::Index_range indices(slice_indices_[t]);
        next_potential = kjb::log_normalize(A().get_col(z(t)));
    }
    std::cerr << "done." << std::endl;
}
 */

void Markov_transition_model::sample_z_backwards_beamily_(const size_t& i)
{
    /*
     sample state z backward by beam sampling, refer to the beam sampling paper
     */
    PMWP(verbose_ > 0, "       Sampling z %d backward conditioned on slice vars...", (i));
    //std::cerr << "        Sampling z " << i << " backward conditioned on slice vars...";
    Prob_vector pzt_ztp1_x1T = messages(i, T(i)); // p(z_t | z_{t+1}, x_{1:T})
    kjb::Index_range indices(slice_indices_[i][T(i)]);
    kjb::Categorical_distribution<size_t> r_z(
        slice_indices_[i][T(i)],
        kjb::log_normalize_and_exponentiate(pzt_ztp1_x1T)
        );
    z(i, T(i)) = kjb::sample(r_z);
    for(int t = T(i) - 1; t >= 1; --t)
    {
        Index_list values;
        Prob_vector probs;
        Prob_vector::const_iterator m_it = messages(i, t).begin();
        for(auto it = slice_indices_[i][t].begin();
            it != slice_indices_[i][t].end(); ++it, ++m_it)
        {
            if(A(*it, z(i,t+1)) > slice_variables_[i][t+1])
            {
                values.push_back(*it);
                probs.push_back(*m_it);
            }
        }
        kjb::Categorical_distribution<size_t> r_z(
            values,
            kjb::log_normalize_and_exponentiate(probs)
        );
        z(i, t) = kjb::sample(r_z);
    }
    PM(verbose_ > 0, "done.\n");
    //std::cerr << "done." << std::endl;
}

/*
void Markov_transition_model::sample_z_backwards_beamily_()
{
    std::cerr << "        Sampling z backward conditioned on slice vars...";
    // std::cerr << "    Beam sampling z backwards...";
    Prob_vector pzt_ztp1_x1T = messages(T()); // p(z_t | z_{t+1}, x_{1:T})
    kjb::Index_range indices(slice_indices_[T()]);
    kjb::Categorical_distribution<size_t> r_z(
        slice_indices_[T()],
        kjb::log_normalize_and_exponentiate(pzt_ztp1_x1T)
        );
    // std::cerr << "        t = " << T() << std::endl;
    // std::cerr << "            Sampling z from indices " << slice_indices_[T()]
    //           << " with probabilities "
    //           << kjb::log_normalize_and_exponentiate(pzt_ztp1_x1T)
    //           << std::endl;
    z(T()) = kjb::sample(r_z);
    for(int t = T() - 1; t >= 1; --t)
    {
        Index_list values;
        Prob_vector probs;
        Prob_vector::const_iterator m_it = messages(t).begin();
        for(auto it = slice_indices_[t].begin();
            it != slice_indices_[t].end(); ++it, ++m_it)
        {
            if(A(*it, z(t+1)) > slice_variables_[t+1])
            {
                values.push_back(*it);
                probs.push_back(*m_it);
            }
        }
        // std::cerr << "        t = " << t << std::endl;
        kjb::Categorical_distribution<size_t> r_z(
            values,
            kjb::log_normalize_and_exponentiate(probs)
            );
        // std::cerr << "            Sampling z from indices " << values
        //           << " with probabilities "
        //           << kjb::log_normalize_and_exponentiate(probs)
        //           << std::endl;
        z(t) = kjb::sample(r_z);
        // std::cerr << "            Chose z = " << z(t) << std::endl;
        // we want p(z_{t-1} | x_{1:T}, z_t)
        // \propto p(z_{t-1} | x_{1:{t-1}}) p(z_{t} | z_{t-1}) p(x_{t:T} | z_t)
        // \propto p(z_{t-1} | x_{1:{t-1}}) p(z_{t} | z_{t-1})
        // std::cerr << "done." << std::endl;
    }
    std::cerr << "done." << std::endl;
}
 */

