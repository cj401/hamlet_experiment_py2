/* $Id: linear_emission_model.cpp 21443 2017-06-28 18:13:50Z chuang $ */

/*!
 * @file linear_emission_model.cpp
 *
 * @author Colin Dawson 
 */

#include "util.h"
#include "linear_emission_model.h"
#include "weight_prior.h"
#include "noise_model.h"
#include "hdp_hmm_lt.h"
#include <boost/make_shared.hpp>

/* /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ */ 

Emission_model_ptr Linear_emission_parameters::make_module() const
{
    return boost::make_shared<Linear_emission_model>(wp, np);
}

Linear_emission_model::Linear_emission_model(
    const Weight_param_ptr       weight_parameters,
    const Noise_param_ptr        noise_parameters
    ) : Emission_model(noise_parameters),
        wp_(weight_parameters->make_module())
{}

void Linear_emission_model::set_up_verbose_level(const size_t verbose)
{
    Base_class::set_up_verbose_level(verbose);
    wp_->set_up_verbose_level(verbose);
}

void Linear_emission_model::initialize_resources()
{
    PM(verbose_ > 0, "Allocating resrouces for emission model...\n");
    //std::cerr << "Allocating resources for emission model..." << std::endl;
    Base_class::initialize_resources();
    wp_->set_parent(this);
    wp_->initialize_resources();
    theta_star() = State_matrix_list((int) NF());
    for (size_t i = 0; i < NF(); i++)
    {
        theta_star(i) = State_matrix((int) T(i), (int) W().get_num_rows(), 0.0);
    }
    PM(verbose_ > 0, "done.\n");
    //std::cerr << "done." << std::endl;
    /*
     add in to initialize X_star_
     */
    X_star_ = Mean_matrix_list((int) NF());
    for (size_t i = 0; i < NF(); i++)
    {
        X_star_.at(i) = Mean_matrix(T(i), K(), 0.0);
    }
}

/*
void Linear_emission_model::initialize_resources()
{
    std::cerr << "Allocating resources for emission model..." << std::endl;
    Base_class::initialize_resources();
    wp_->set_parent(this);
    wp_->initialize_resources();
    theta_star() = State_matrix((int) T_, (int) W().get_num_rows(), 0.0);
    // if(includes_reference_state())
    // {
    //     std::cerr << "    Adding bias column to thetastar...";
    //     theta_star().set_col(W().get_num_rows() - 1, State_type((int) T_, 1.0));
    //     std::cerr << "done." << std::endl;
    // }
    // std::cerr << "    Allocating X...";
    // X_ = Mean_matrix(J(), K(), 0.0);
}
 */

void Linear_emission_model::initialize_params()
{
    PM(verbose_ > 0, "Initializing linear emission parameters...\n");
    //std::cerr << "Initializing linear emission parameters..." << std::endl;
    wp_->initialize_params();
    // sync_means_();
    Base_class::initialize_params();
};

void Linear_emission_model::input_previous_results(const std::string& input_path, const std::string& name)
{
    PM(verbose_ > 0, "Initializing linear emission parameters...\n");
    //std::cerr << "Initializing Linear emission parameters..." << std::endl;
    wp_->input_previous_results(input_path, name);
    Base_class::input_previous_results(input_path, name);
}

void Linear_emission_model::generate_data(const std::string& path)
{
    std::cerr << "    Writing weights to file " << path + "weights.txt" << std::endl;
    Weight_matrix W_star = W();
    if(includes_reference_state())
    {
        W_star.vertcat(
            kjb::create_row_matrix(wp_->get_bias()));
    }
    W_star.write((path + "weights.txt").c_str());
    sync_means_();
    generate_observations(T_, path);
};

void Linear_emission_model::set_up_results_log() const
{
    wp_->set_up_results_log();
    Base_class::set_up_results_log();
}

void Linear_emission_model::write_state_to_file(const std::string& name) const
{
    wp_->write_state_to_file(name);
    Base_class::write_state_to_file(name);
};

void Linear_emission_model::update_params()
{
    PM(verbose_ > 0, "Updating emission parameters...\n");
    //std::cerr << "Updating emission parameters..." << std::endl;
    sync_means_();
    Base_class::update_params();
    wp_->update_params();
    sync_means_();
}

const State_matrix& Linear_emission_model::theta_star(const size_t& i) const
{
    return parent->theta_star(i);
}

State_matrix& Linear_emission_model::theta_star(const size_t& i)
{
    return parent->theta_star(i);
}

const State_matrix_list& Linear_emission_model::theta_star() const
{
    return parent->theta_star();
}

State_matrix_list& Linear_emission_model::theta_star()
{
    return parent->theta_star();
}

size_t Linear_emission_model::dimension_map(const size_t& dprime) const
{
    return parent->dimension_map(dprime);
}

// const State_matrix& Linear_emission_model::theta_prime() const
// {
//     return parent->theta_prime();
// }

Likelihood_matrix Linear_emission_model::get_log_likelihood_matrix(
    const size_t&       i,
    const State_matrix& theta,
    const kjb::Index_range& times
    ) const
{
    Mean_matrix X = theta * W();
    if(includes_reference_state())
    {
        X.shift_rows_by(wp_->get_bias());
    }
    return nm_->get_log_likelihood_matrix(i, X, times);
}

/*
Likelihood_matrix Linear_emission_model::get_log_likelihood_matrix(
    const State_matrix& theta,
    const kjb::Index_range& times
    ) const
{
    Mean_matrix X = theta * W();
    if(includes_reference_state())
    {
        X.shift_rows_by(wp_->get_bias());
    }
    return nm_->get_log_likelihood_matrix(X, times);
}
 */

Likelihood_matrix Linear_emission_model::get_conditional_log_likelihood_matrix(
    const size_t&      i,
    const size_t&      d_first,
    const size_t&      range_size
    ) const
{
    bool include_zeroes = wp_->includes_bias();
    kjb::Index_range other_dimensions; // dimensions we are conditioning on
    if(d_first > 0)
    {
        kjb::Index_range before_dimensions(0, d_first - 1);
        other_dimensions.concat(before_dimensions);
    }
    if(d_first + range_size <= D_prime())
    {
        kjb::Index_range after_dimensions(d_first + range_size, D_prime() - 1);
        other_dimensions.concat(after_dimensions);
    }
    // the mean with all considered dimensions zeroed out
    Mean_matrix X_others =
    theta_star(i)(kjb::Index_range(true), other_dimensions) *
    W()(other_dimensions, kjb::Index_range(true));
    if(include_zeroes)
    {
        X_others.shift_rows_by(wp_->get_bias());
    }
    // the matrix of changes to the output for each possible value of the
    // considered dimensions (we assume exactly one is 1, the rest zero)
    Mean_matrix delta_x((int) range_size + (int) include_zeroes, (int) K(), 0.0);
    // if there is a reference state, the first value is "no change" from X_others
    // otherwise, all values use some weights
    delta_x.replace(
        (int) include_zeroes, 0,
        W()(kjb::Index_range(d_first, d_first + range_size - 1),
                        kjb::Index_range(true)));
    return nm_->get_conditional_log_likelihood_matrix(i, X_others, delta_x);
}

/*
Likelihood_matrix Linear_emission_model::get_conditional_log_likelihood_matrix(
    const size_t&      d_first,
    const size_t&      range_size
    ) const
{
    bool include_zeroes = wp_->includes_bias();
    // std::cerr << "Linear_emission_model::get_conditional_log_likelihood("
    //           << d_first << "," << range_size  << "," << include_zeroes
    //           << ")" << std::endl;
    // std::cerr << "D_prime() = " << D_prime() << std::endl;
    kjb::Index_range other_dimensions; // dimensions we are conditioning on
    if(d_first > 0)
    {
        kjb::Index_range before_dimensions(
            0, d_first - 1);
        other_dimensions.concat(before_dimensions);
    }
    if(d_first + range_size <= D_prime())
    {
        kjb::Index_range after_dimensions(d_first + range_size, D_prime() - 1);
        other_dimensions.concat(after_dimensions);
    }
    // the mean with all considered dimensions zeroed out
    Mean_matrix X_others =
        theta_star()(kjb::Index_range(true), other_dimensions) *
        W()(other_dimensions, kjb::Index_range(true));
    // if there is a reference state, need to take into account the bias term
    if(include_zeroes)
    {
        X_others.shift_rows_by(wp_->get_bias());
    }
    // the matrix of changes to the output for each possible value of the
    // considered dimensions (we assume exactly one is 1, the rest zero)
    Mean_matrix delta_x((int) range_size + (int) include_zeroes, (int) K(), 0.0);
    // if there is a reference state, the first value is "no change" from X_others
    // otherwise, all values use some weights
    delta_x.replace(
        (int) include_zeroes, 0,
        W()(kjb::Index_range(d_first, d_first + range_size - 1),
            kjb::Index_range(true)));
    // std::cerr << "X_others is" << X_others.get_num_rows() << "x"
    //           << X_others.get_num_cols() << std::endl;
    // std::cerr << "delta_x = " << std::endl;
    // std::cerr << delta_x << std::endl;
    // std::cerr << "theta_star_others = " << std::endl;
    // std::cerr << theta_star()(kjb::Index_range(true), other_dimensions) << std::endl;
    // std::cerr << "W_others = " << std::endl;
    // std::cerr << W()(other_dimensions, kjb::Index_range(true)) << std::endl;
    // std::cerr << "X_others = " << std::endl;
    // std::cerr << X_others << std::endl;
    // std::cerr << "delta_x is" << delta_x.get_num_rows() << "x"
    //           << delta_x.get_num_cols() << std::endl;
    return nm_->get_conditional_log_likelihood_matrix(X_others, delta_x);
}
 */

Likelihood_matrix Linear_emission_model::get_test_log_likelihood_matrix(
    const size_t&       i,
    const State_matrix& theta) const
{
    Mean_matrix X = theta * W();
    if(includes_reference_state())
    {
        X.shift_rows_by(wp_->get_bias());
    }
    return nm_->get_test_log_likelihood_matrix(i, X);
};

Mean_matrix Linear_emission_model::build_augmented_mean_matrix_for_d
(
 const size_t& i,
 const size_t& first_index,
 const size_t& range_size
 )
{
    kjb::Index_range range(first_index, first_index + range_size - 1);
    Mean_matrix X_d =
        theta_star(i)(kjb::Index_range(true), range) *
            W()(range, kjb::Index_range(true));
    assert(X_d.get_num_rows() == (int) T(i));
    assert(X_d.get_num_cols() == (int) K());
    return X_d;
}

/*
Mean_matrix Linear_emission_model::build_augmented_mean_matrix_for_d
(
    const size_t& first_index,
    const size_t& range_size
)
{
    kjb::Index_range range(first_index, first_index + range_size - 1);
    Mean_matrix X_d =
        theta_star()(kjb::Index_range(true), range) *
        W()(range, kjb::Index_range(true));
    assert(X_d.get_num_rows() == (int) T());
    assert(X_d.get_num_cols() == (int) K());
    return X_d;
}
 */

void Linear_emission_model::insert_latent_dimension(
    const size_t& d,
    const size_t& new_pos)
{
    wp_->insert_latent_dimension(d, new_pos);
    // sync_means_();
}

void Linear_emission_model::remove_latent_dimension(const size_t& old_pos)
{
    wp_->remove_latent_dimension(old_pos);
    // sync_means_();
}

void Linear_emission_model::replace_latent_dimension(
    const size_t& d,
    const size_t& pos)
{
    wp_->replace_latent_dimension(d, pos);
    // sync_means_();
}

double Linear_emission_model::log_likelihood_ratio_for_state_change(
    const size_t& i,
    const State_type& theta_current,
    const double& delta,
    const kjb::Index_range& indices,
    const size_t& d
    ) const
{
    if(indices.size() == 0) return 0.0;
    Mean_vector diff_vector = delta * W().get_row(d); // K-vec
    Mean_vector x = theta_current * W();
    if(includes_reference_state())
    {
        x = x + wp_->get_bias();
    }
    return nm_->log_likelihood_ratio_for_state_change(i, x,diff_vector,indices);
}

/*
double Linear_emission_model::log_likelihood_ratio_for_state_change(
    const State_type& theta_current,
    const double& delta,
    const kjb::Index_range& indices,
    const size_t& d
    ) const
{
    if(indices.size() == 0) return 0.0;
    Mean_vector diff_vector = delta * W().get_row(d); // K-vec
    Mean_vector x = theta_current * W();
    if(includes_reference_state())
    {
        x = x + wp_->get_bias();
    }
    return nm_->log_likelihood_ratio_for_state_change(x,diff_vector,indices);
}
 */

Prob_vector Linear_emission_model::log_likelihood_ratios_for_state_range(
    const size_t&     i,
    const State_type& theta_current,
    const size_t& d,
    const size_t& first_index,
    const size_t& range_size,
    const kjb::Index_range& indices,
    bool include_new
    ) const
{
    if(indices.size() == 0) return Prob_vector((int) range_size + (int) include_new, 0.0);
    Mean_matrix diff_matrix = W().submatrix(first_index, 0, range_size, K_); // range-size x K
    if(include_new)
    {
        // std::cerr << "        Proposing a new weight vector...";
        Prob_matrix w_new = kjb::create_row_matrix(wp_->propose_weight_vector(d));
        diff_matrix.vertcat(w_new);
        // std::cerr << "done." << std::endl;
    }
    Mean_vector x_old = theta_current * W();
    // std::cerr << "x_old = " << x_old << std::endl;
    Prob_vector result =
        nm_->log_likelihood_ratios_for_state_changes(i, x_old, diff_matrix, indices);
    // std::cerr << "likelihood ratios = " << result << std::endl;
    return result;
}

/*
Prob_vector Linear_emission_model::log_likelihood_ratios_for_state_range(
    const State_type& theta_current,
    const size_t& d,
    const size_t& first_index,
    const size_t& range_size,
    const kjb::Index_range& indices,
    bool include_new
    ) const
{
    // use when we want to consider adding one of several weight vectors
    // to the mean
    if(indices.size() == 0) return Prob_vector((int) range_size + (int) include_new, 0.0);
    // Row s of diff_matrix consists of a set of weights that would be added to
    // the reference mean (x_old) if theta_{j,d} takes on state s.
    Mean_matrix diff_matrix = W().submatrix(first_index, 0, range_size, K_); // range-size x K
    // If we are considering a brand new value of s, sample the weights from the prior
    if(include_new)
    {
        // std::cerr << "        Proposing new weight vector = " << std::endl;
        Prob_matrix w_new = kjb::create_row_matrix(wp_->propose_weight_vector(d));
        // std::cerr << new_weights << std::endl;
        // std::cerr << "        Existing weights are " << std::endl;
        // std::cerr << diff_matrix << std::endl;
        diff_matrix.vertcat(w_new);
    }
    // Get the mean assuming that theta_{j,d} has the reference value of zero
    Mean_vector x_old = theta_current * W();
    return nm_->log_likelihood_ratios_for_state_changes(x_old, diff_matrix, indices);
}
 */

