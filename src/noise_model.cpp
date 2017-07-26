/* $Id: noise_model.cpp 21443 2017-06-28 18:13:50Z chuang $ */

/*!
 * @file noise_model.cpp
 *
 * @author Colin Dawson 
 */

#include "util.h"
#include "noise_model.h"
#include "emission_model.h"

void Noise_model::set_parent(Emission_model* const p)
{
    parent = p;
    write_path = parent->write_path;
}

/*
Prob_vector Noise_model::log_likelihood_ratios_for_state_changes(
    const Mean_vector&      x,
    const Mean_matrix&      delta,
    const kjb::Index_range& times
    )
{
    size_t S = delta.get_num_rows();
    Prob_vector result((int) S, 0.0);
    for(size_t s = 0; s < S; ++s)
    {
        result[s] =
            log_likelihood_ratio_for_state_change(x, delta.get_row(s), times);
    }
    return result;
}
 */

Prob_vector Noise_model::log_likelihood_ratios_for_state_changes(
    const size_t& i,
    const Mean_vector&      x,
    const Mean_matrix&      delta,
    const kjb::Index_range& times
    )
{
    size_t S = delta.get_num_rows();
    Prob_vector result((int) S, 0.0);
    for(size_t s = 0; s < S; ++s)
    {
        // std::cerr << "        Log likelihood ratio for value "
        //           << s << " is ";
        result[s] =
            log_likelihood_ratio_for_state_change(i, x, delta.get_row(s), times);
        // std::cerr << result[s] << std::endl;
    }
    return result;
}

size_t Noise_model::NF() const { return parent->NF();}

size_t Noise_model::test_NF() const { return parent->test_NF();}

Prob_matrix Noise_model::get_log_likelihood_matrix(
    const size_t&              i,
    const Mean_matrix&         X,
    const kjb::Index_range&    times,
    const size_t&              all_T,
    Prob (Noise_model::* log_likelihood_f)(const size_t& i, const size_t& t, const Mean_vector& mean) const
    ) const
{
    /*
     compute the emission log likelihood for being in each hidden state j with the observed data y
     for each time period t
     */
    size_t J = X.get_num_rows();
    size_t T;
    Time_set time_indices;
    if(times.all())
    {
        T = all_T;
        time_indices.reserve(T);
        for(size_t t = 1; t <= T; ++t)
        {
            time_indices.push_back(t);
        }
    } else {
        T = times.size();
        time_indices = times.expand();
    }
    Likelihood_matrix llm((int) T + 1, (int) J, 0.0);
    for(size_t j = 0; j < J; ++j)
    {
        Mean_vector mean = X.get_row(j);
        // std::cerr << "   X row " << j << " = " << mean << std::endl;
        for (size_t t = 1; t <= T; ++t)
        {
            llm(t, j) = (this->*log_likelihood_f)(i, time_indices[t-1] - 1, mean);
        }
    }
    return llm;
}

/*
Prob_matrix Noise_model::get_log_likelihood_matrix(
    const Mean_matrix&         X,
    const kjb::Index_range&    times,
    const size_t&              all_T,
    Prob (Noise_model::* log_likelihood_f)(const size_t& t, const Mean_vector& mean) const
    ) const
{
    size_t J = X.get_num_rows();
    // std::cerr << "    Computing log likelihoods...";
    size_t T;
    Time_set time_indices;
    if(times.all())
    {
        T = all_T;
        time_indices.reserve(T);
        for(size_t t = 1; t <= T; ++t)
        {
            time_indices.push_back(t);
        }
    } else {
        T = times.size();
        time_indices = times.expand();
    }
    Likelihood_matrix llm((int) T + 1, (int) J, 0.0);
    for(size_t j = 0; j < J; ++j)
    {
        Mean_vector mean = X.get_row(j);
        for (size_t t = 1; t <= T; ++t)
        {
            llm(t, j) = (this->*log_likelihood_f)(time_indices[t-1] - 1, mean);
        }
    }
    return llm;
}
 */

Likelihood_matrix Noise_model::get_conditional_log_likelihood_matrix(
    const size_t&      i,
    const Mean_matrix& X_others,
    const Mean_matrix& delta_x
    )
{
    PM(verbose_ > 0, "    Calculating conditional log likelihood matrix...");
    //std::cerr << "    Calculating conditional log likelihood matrix...";
    // << std::endl;
    Prob_matrix result((int) T(i) + 1, (int) delta_x.get_num_rows());
    // std::cerr << "        Result is " << T() << "x" << delta_x.get_num_rows()
    //           << std::endl;
    for(int j = 0; j < result.get_num_cols(); ++j)
    {
        Mean_vector delta = delta_x.get_row(j);
        for(size_t t = 1; t <= T(i); ++t)
        {
            // std::cerr << "        (" << t << "," << j << ") ";
            Mean_vector mean = X_others.get_row(t-1) + delta;
            result(t,j) = this->log_likelihood(i, t-1, mean);
        }
        // std::cerr << std::endl;
    }
    PM(verbose_ > 0, "done.\n");
    //std::cerr << "done." << std::endl;
    // std::cerr << "Cond-LLM = " << std::endl;
    // std::cerr << result << std::endl;
    return result;
}

/*
Likelihood_matrix Noise_model::get_conditional_log_likelihood_matrix(
    const Mean_matrix& X_others,
    const Mean_matrix& delta_x
    )
{
    std::cerr << "    Calculating conditional log likelihood matrix...";
    // << std::endl;
    Prob_matrix result((int) T() + 1, (int) delta_x.get_num_rows());
    // std::cerr << "        Result is " << T() << "x" << delta_x.get_num_rows()
    //           << std::endl;
    for(int j = 0; j < result.get_num_cols(); ++j)
    {
        Mean_vector delta = delta_x.get_row(j);
        for(size_t t = 1; t <= T(); ++t)
        {
            // std::cerr << "        (" << t << "," << j << ") ";
            Mean_vector mean = X_others.get_row(t-1) + delta;
            result(t,j) = this->log_likelihood(t-1,mean);
        }
        // std::cerr << std::endl;
    }
    std::cerr << "done." << std::endl;
    // std::cerr << "Cond-LLM = " << std::endl;
    // std::cerr << result << std::endl;
    return result;
}
 */
