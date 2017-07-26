
/* $Id: categorical_state_model.cpp 21509 2017-07-21 17:39:13Z cdawson $ */

/*!
 * @file categorical_state_model.cpp
 *
 * @author Colin Dawson 
 */

#include "util.h"
#include "categorical_state_model.h"
#include "similarity_model.h"
#include "emission_model.h"
#include "hdp_hmm_lt.h"
#include <third_party/underflow_utils.h>
#include <boost/make_shared.hpp>
#include <boost/bind.hpp>
#include <algorithm>

Categorical_state_parameters::Categorical_state_parameters(
    const Parameters&    params,
    const std::string&   name
    ) : State_parameters(params, name),
        fixed_alpha(params.exists(name, "alpha") || params.exists(name, "alpha_file")),
        identical_alpha_priors(params.exists(name, "a_alpha")),
        alpha(
            !fixed_alpha || params.exists(name, "alpha_file") ? NULL_CONC :
            params.get_param_as<double>(
                name, "alpha", bad_alpha_prior(), valid_conc_param)),
        a_alpha(
            fixed_alpha || !identical_alpha_priors ? NULL_SHAPE :
            params.get_param_as<double>(
                name, "a_alpha", bad_alpha_prior(), valid_shape_param)),
        b_alpha(
            fixed_alpha || !identical_alpha_priors ? NULL_RATE :
            params.get_param_as<double>(
                name, "b_alpha", bad_alpha_prior(), valid_rate_param)),
        alpha_file(
            !fixed_alpha || params.exists(name, "alpha") ? "" :
            params.get_param_as<std::string>(":experiment", "params_path") +
            params.get_param_as<std::string>(name, "alpha_file")),
        alpha_prior_file(
            fixed_alpha || identical_alpha_priors ? "" :
            params.get_param_as<std::string>(":experiment", "params_path") + 
            params.get_param_as<std::string>(name, "alpha_prior_file"))
{
    IFT((a_alpha > 0 && b_alpha > 0) || alpha > 0 || !alpha_file.empty() ||
        !alpha_prior_file.empty(),
        kjb::IO_error, bad_alpha_prior());
}

State_model_ptr Categorical_state_parameters::make_module() const
{
    return boost::make_shared<Categorical_state_model>(this);
}

Categorical_state_model::Categorical_state_model(
    const Categorical_state_parameters * const hyperparams
    ) : State_model(hyperparams),
        alpha_file(hyperparams->alpha_file),
        alpha_prior_file(hyperparams->alpha_prior_file),
        a_alpha_(hyperparams->alpha_prior_file.empty() ?
                 Scale_vector((int) hyperparams->D, hyperparams->a_alpha) :
                 Scale_vector(kjb::Matrix(alpha_prior_file).get_col(0))),
        b_alpha_(hyperparams->alpha_prior_file.empty() ?
                 Scale_vector((int) hyperparams->D, hyperparams->b_alpha) :
                 Scale_vector(kjb::Matrix(alpha_prior_file).get_col(1))),
        fixed_alpha(hyperparams->fixed_alpha),
        alpha_((int) hyperparams->D, hyperparams->alpha),
        entity_counts_((int) hyperparams->D, State_count_v()),
        feature_dimensions_((int) hyperparams->D, 0),
        cum_feature_dimensions_((int) hyperparams->D + 1, 0)
{}

void Categorical_state_model::initialize_params()
{
    PM(verbose_ > 0, "Initializing categorical state model...\n");
    //std::cerr << "Initializing categorical state model..." << std::endl;
    // Initialize alpha from the prior
    if(!alpha_file.empty())
    {
        alpha_ = Scale_vector(alpha_file);
    }
    if(!fixed_alpha)
    {
        PM(verbose_ > 0, "    Initializing alpha = ");
        //std::cerr << "    Initializing alpha = ";
        for(size_t d = 0; d < D(); ++d)
        {
            Gamma_dist r_alpha(a_alpha_[d], 1.0 / b_alpha_[d]);
            alpha_[d] = kjb::sample(r_alpha);
            // (*alpha_it) = 1.0;
        }
    } else {
        PM(verbose_ > 0, "    Using fixed alpha = ");
        //std::cerr << "    Using fixed alpha = ";
    }
    if (verbose_ > 0) {std::cerr << alpha_ << std::endl;}
    //std::cerr << alpha_ << std::endl;
    // Initialize theta from the prior conditioned on alpha
    //std::cerr << "    Initializing theta (" << J() << " x " << D() << ")";
    PMWP(verbose_ > 0, "    Initializing theta (%d x %d)", (J())(D()));
    for(size_t d = 0; d < D(); ++d)
    {
        //std::cerr << std::endl;
        //std::cerr << "        Sampling number of values for feature " << d << "...";
        PMWP(verbose_ > 0, "\n        Sampling number of values for feature %d...", (d));
        kjb::Chinese_restaurant_process r_tables(alpha_.at(d), J());
        // tables is a vector of vectors, where the s-th vector is a list
        // of states assigned to "table" s
        kjb::Crp::Type tables = kjb::sample(r_tables);
        //std::cerr << "done. (S = " << tables.size() << ")" << std::endl;
        PMWP(verbose_ > 0, "done. (S = %d)\n", (tables.size()));
        // feature_dimensions_[d] is number of distinct states for dim d
        feature_dimensions_.at(d) = tables.size();
        // entity_counts_[d][s] stores number of states out of J whose d
        // dimension is in state s
        //std::cerr << "done. (S = " << tables.size() << ")" << std::endl;
        //std::cerr << "        Writing S to object...";
        PMWP(verbose_ > 0, "done. (S = %d)\n", (tables.size()));
        PM(verbose_ > 0, "        Writing S to object...");
        entity_counts_[d] = Prob_vector((int) tables.size(), 0.0);
        for(size_t s = 0; s < tables.size(); ++s)
        {
            entity_counts_[d][s] = (double) tables[s].size();
        }
        PM(verbose_ > 0, "done.\n");
        //std::cerr << "done." << std::endl;
        size_t table_number = 0;
        // std::cerr << "        Building dummy variable matrix..." << std::endl;
        // Build a "dummy variable" matrix with 1 in position (j, s) when
        // state j has value s on feature d; 0 elsewhere (so, exactly one
        // non-zero entry per row)
        State_matrix dummy_matrix((int) J(), (int) tables.size(), 0.0);
        for(kjb::Crp::Type::const_iterator table_it = tables.begin();
            table_it != tables.end(); ++table_it)
        {
            // std::cerr << "            'Table' number " << table_number
            //           << " has states " << *table_it
            //           << std::endl;
            for(std::vector<size_t>::const_iterator table_iit = (*table_it).begin();
                table_iit != (*table_it).end(); ++table_iit)
            {
                // std::cerr << "                State " << *table_iit
                //           << " assigned to table " << table_number << std::endl;
                // the main theta_ matrix just records the integer s in position
                // (j,d)
                theta_(*table_iit, d) = table_number;
                dummy_matrix(*table_iit, table_number) = 1.0;
            }
            ++table_number;
        }
        // for the first dimension, copy the dummy_matrix to dummy_theta;
        // after the first, concatenate the current dummary matrix to the
        // existing one
        // std::cerr << "dummy_matrix[" << d << "] =" << std::endl;
        // std::cerr << kjb::floor(dummy_matrix) << std::endl;
        if(d == 0)
        {
            dummy_theta_ = dummy_matrix;
        } else {
            dummy_theta_.horzcat(dummy_matrix);
        }
        // std::cerr << "done." << std::endl;
        // std::cerr << std::endl;
        // std::cerr << "    ";
        PM(verbose_ > 0, "done.\n");
        //std::cerr << "done." << std::endl;
    }
    sync_cumulative_feature_dimensions();
    // std::cerr << "Dummy theta = " << std::endl;
    // std::cerr << kjb::floor(dummy_theta_) << std::endl;
    theta_prime_ = &dummy_theta_;
    D_prime_ = dummy_theta_.get_num_cols();
}

void Categorical_state_model::initialize_resources()
{
    Base_class::initialize_resources();
}

void Categorical_state_model::input_previous_results(const std::string& input_path, const std::string& name)
{
    std::cerr << "Inputting information for categorical state model..." << std::endl;
    Base_class::input_previous_results(input_path, name);
    if (!fixed_alpha)
    {
        std::cerr << "Inputting information for prob vector alpha_..." << std::endl;
        alpha_ = Prob_vector(input_to_vector<double>(input_path, "alpha_theta.txt", name));
        std::cerr << "done." << std::endl;
    }
}

void Categorical_state_model::set_up_results_log() const
{
    Base_class::set_up_results_log();
    if(!fixed_alpha)
    {
        std::ofstream ofs;
        ofs.open(write_path + "alpha_theta.txt", std::ofstream::out);
        ofs << "iteration value" << std::endl;
        ofs.close();
    }
    create_directory_if_nonexistent(write_path + "mean_by_state");
}

/*
void Categorical_state_model::write_state_to_file(const std::string& name) const
{
    const std::string filestem = write_path + name;
    std::ofstream ofs;
    kjb::Matrix(theta()).submatrix(0,0,J(), D()).floor().write(
        (write_path + "theta/" + name + ".txt").c_str());
    kjb::Matrix(theta_star()).submatrix(0,0,T(), D_prime_).floor().write(
        (write_path + "thetastar/" + name + ".txt").c_str());
    if(!fixed_alpha)
    {
        ofs.open(write_path + "alpha_theta.txt", std::ofstream::out | std::ofstream::app);
        ofs << name << " " << alpha_ << std::endl;
        ofs.close();
    }
    build_augmented_mean_matrix().write(
        (write_path + "mean_by_state/" + name + ".txt").c_str());
}
 */

void Categorical_state_model::write_state_to_file(const std::string& name) const
{
    const std::string filestem = write_path + name;
    std::ofstream ofs;
    kjb::Matrix(theta()).submatrix(0,0,J(), D()).floor().write(
        (write_path + "theta/" + name + ".txt").c_str());
    if (NF() == 1)
    {
        kjb::Matrix(theta_star(0)).submatrix(0,0,T(0), D_prime_).floor().write(
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
    if(!fixed_alpha)
    {
        ofs.open(write_path + "alpha_theta.txt", std::ofstream::out | std::ofstream::app);
        ofs << name << " " << alpha_ << std::endl;
        ofs.close();
    }
    if (NF() == 1)
    {
        build_augmented_mean_matrix(0).write(
            (write_path + "mean_by_state/" + name + ".txt").c_str());
    }
    else
    {
        create_directory_if_nonexistent((write_path + "mean_by_state/" + name).c_str());
        boost::format fmt("%03i");
        for (size_t i = 0; i < NF(); i++)
        {
            std::string file_name = (fmt % (i+1)).str() + ".txt";
            build_augmented_mean_matrix(i).write(
                (write_path + "mean_by_state/" + name + "/" + file_name).c_str());
        }
    }
}

kjb::Matrix Categorical_state_model::build_augmented_mean_matrix(const size_t& i) const
{
    kjb::Matrix result =
    emission_model()->build_augmented_mean_matrix_for_d(i, 0, feature_dimensions_[0]);
    for(size_t d = 1; d < D(); ++d)
    {
        result.horzcat(
            emission_model()->build_augmented_mean_matrix_for_d(i,
                cum_feature_dimensions_[d], feature_dimensions_[d]));
    }
    return result;
}

/*
kjb::Matrix Categorical_state_model::build_augmented_mean_matrix() const
{
    kjb::Matrix result =
        emission_model()->build_augmented_mean_matrix_for_d(0,feature_dimensions_[0]);
    for(size_t d = 1; d < D(); ++d)
    {
        result.horzcat(
            emission_model()->build_augmented_mean_matrix_for_d(
                cum_feature_dimensions_[d],
                feature_dimensions_[d]));
    }
    return result;
}
 */

void Categorical_state_model::update_params()
{
    //std::cerr << "Updating state parameters..." << std::endl;
    PM(verbose_ > 0, "Updating state parameters...\n");
    update_theta_();
    if(!fixed_alpha)
    {
        update_alpha_();
    }
}

void Categorical_state_model::add_ground_truth_eval_header() const
{
    //// Doesn't make a lot of sense to do this here.
}

void Categorical_state_model::compare_state_sequence_to_ground_truth(
    const std :: string &,
    const std :: string &) const
{
    //// Doesn't make a lot of sense to do this here.
}

void Categorical_state_model::update_alpha_()
{
    //TODO: Write this
    PM(verbose_ > 0, "    Updating alpha...");
    //std::cerr << "    Updating alpha...";
    for(size_t d = 0; d < D(); ++d)
    {
        double shape = a_alpha_[d] + feature_dimensions_[d] - 1;
        threshold_a_double(shape, write_path, "alpha_theta");
        Beta_dist r_t(alpha_.at(d) + 1, J());
        double minus_log_t = -log(kjb::sample(r_t));
        double rate = b_alpha_[d] + minus_log_t;
        kjb::Categorical_distribution<int> r_indicator(
            0,1, shape, J() * rate);
        // threshold_a_double(scale, write_path, "alpha_theta");
        Conc_dist r_alpha(shape + kjb::sample(r_indicator), 1.0 / rate);
        alpha_[d] = kjb::sample(r_alpha);
    }
    PM(verbose_ > 0, "done.\n");
    //std::cerr << "done." << std::endl;
}

void Categorical_state_model::update_theta_(const size_t & j, const size_t & d)
{
    // std::cerr << std::endl;
    // std::cerr << "Updating theta[" << j << "," << d << "]" << std::endl;
    // std::cerr << "Old theta = " << std::endl;
    // std::cerr << kjb::floor(theta_) << std::endl;
    /// Actually update theta from its conditional posterior
    int& S = feature_dimensions_[d]; // current number of distinct values
    // std::cerr << "    Current S = " << S << std::endl;
    const int theta_old = get_theta(j,d); // old theta value ( == s)
    // std::cerr << "    old theta[j,d] = " << theta_old << std::endl;
    const double t = similarity_model()->get_model()->get_theta()(j,d);
    // std::cerr << "    Similarity model has theta = " << t << std::endl;
    if(theta_old != t)
    {
        std::cerr << "ERROR: Sim model has theta = " << t
                  << " but State model has theta = " << theta_old
                  << std::endl;
        assert(false);
    }
    // old_index is the dummy variable column for state s in dimension d
    size_t old_index = cum_feature_dimensions_[d] + theta_old;
    // std::cerr << "    Dimension d state theta_old is in column " << old_index << std::endl;
    // std::cerr << "    Feature dimensions = " << feature_dimensions_ << std::endl;
    // std::cerr << "    Cumulative dimensions = " << cum_feature_dimensions_ << std::endl;
    // we are changing theta_old to something else, so decrement the count
    // for feature s
    // std::cerr << "        Old entity_counts[" << d << "] = "
    //           << kjb::floor(entity_counts_[d]) << std::endl;
    size_t old_count = --entity_counts(d, theta_old);
    // std::cerr << "        Temporary entity_counts[" << d << "] = "
    //           << kjb::floor(entity_counts_[d]) << std::endl;
    // vector of counts associated with states so far
    Prob_vector prior = entity_counts(d);
    // std::cerr << "        'Prior' entity count is " << prior << std::endl;
    // need a mass of alpha for prior of "new state"
    // std::cerr << "        Innovation parameter is " << alpha_.at(d) << std::endl;
    prior.push_back(alpha_.at(d));
    Prob_vector log_prior = kjb::ew_log(prior);
    // Prob_vector log_prior((int) entity_counts(d).size() + 1, 1.0);
    // std::cerr << "log prior = " << log_prior << std::endl;
    Prob_vector lt_component =
        similarity_model()->log_likelihood_for_state_range(j, d, S + 1);
    // std::cerr << "lt component = " << lt_component << std::endl;
    // calculate the baseline mean vector for state j (affects X)
    // emission_model()->shift_mean_for_state_j(j, -1, old_index);
    Prob_vector log_likelihood =
        parent->log_likelihood_ratios_for_state_range(
            j, d, cum_feature_dimensions_[d], feature_dimensions_[d], true);
    // std::cerr << "log likelihood = " << kjb::floor(log_likelihood) << std::endl;
    Prob_vector log_posterior = log_prior + lt_component + log_likelihood;
    // std::cerr << "log posterior = " << log_posterior << std::endl;
    kjb::Categorical_distribution<> r_theta(
        kjb::log_normalize_and_exponentiate(log_posterior), 0
        );
    int theta_new = kjb::sample(r_theta);
    // std::cerr << "Chose theta = " << theta_new << std::endl;
    // std::cerr << "theta[" << j << "," << d << "] was " << theta(j,d) << ","
    //           << " now " << theta_new << std::endl;

    /// Do relevant followup bookkeeping
    if(theta_old == theta_new)
    {
        // std::cerr << "Theta is same as old value, so restoring entity count"
        //           << std::endl;
        ++entity_counts(d,theta_old);
        // std::cerr << "    Theta unchanged." << std::endl;
        // std::cerr << "    New theta_prime = " << std::endl;
        // std::cerr << kjb::floor(dummy_theta_) << std::endl;
        return; // nothing else to do
    }
    /// Only get to below if theta changed
    theta_(j,d) = theta_new;
    // std::cerr << "    Getting similarity model in sync with new theta..." << std::endl;
    similarity_model()->sync_after_theta_update(j,d,theta_old, theta_new);
    size_t new_index = cum_feature_dimensions_[d] + theta_new;
    // std::cerr << "    New index = " << new_index << std::endl;
    // had a 1 in position (j,s); replace with zero
    dummy_theta_(j,old_index) = 0;
    // If we created a new state, we have to adjust the dimensions
    // and indices of various things
    if(theta_new == S)
    {
        // std::cerr << "    Selected new value for theta." << std::endl;
        if(old_count > 0) // the case where we aren't just replacing an old value
        {
            // increment the number of distinct values for feature d
            // std::cerr << "         Incrementing feature count in position " << d << std::endl;
            ++S;
            // std::cerr << "        Incrementing cumulative feature counts." << std::endl;
            sync_cumulative_feature_dimensions();
            entity_counts_[d].push_back(1);
            // std::cerr << "        Augmenting theta_prime matrix." << std::endl;
            // std::cerr << "        Old dummy theta = " << std::endl;
            // std::cerr << kjb::floor(dummy_theta_) << std::endl;
            assert((size_t) dummy_theta_.get_num_rows() == J());
            assert((size_t) dummy_theta_.get_num_cols() == D_prime_);
            dummy_theta_.resize(J(), D_prime_ + 1, 0.0);
            // std::cerr << "        Resized dummy theta = " << std::endl;
            // std::cerr << kjb::floor(dummy_theta_) << std::endl;
            assert(dummy_theta_.get_num_rows() == J());
            assert((size_t) dummy_theta_.get_num_cols() == D_prime_ + 1);
            dummy_theta_.insert_zero_column(new_index);
            // std::cerr << "        New dummy theta = " << std::endl;
            // std::cerr << kjb::floor(dummy_theta_) << std::endl;
            dummy_theta_(j,new_index) = 1;
            // std::cerr << "        Augmenting weight matrix." << std::endl;
            D_prime_++;
            emission_model()->insert_latent_dimension(d, new_index);
        } else {  // otherwise we can just reuse this index
            // std::cerr << "    Replacing old theta." << std::endl;
            theta_(j,d) = theta_old;
            similarity_model()->sync_after_theta_update(j,d,theta_new, theta_old);
            new_index = old_index;
            dummy_theta_(j,new_index) = 1;
            emission_model()->replace_latent_dimension(d, new_index);
            entity_counts(d, theta_old) = 1;
        }
    } else if(old_count == 0) {
        // std::cerr << "    Killing off old theta value " << theta_old
        //           << " in dimension " << d
        //           << std::endl;
        // in this case we are killing a feature without creating a new one
        // std::cerr << "        Removing theta dummy column " << old_index << std::endl;
        // std::cerr << "        Old dummy theta = " << std::endl;
        // std::cerr << kjb::floor(dummy_theta_) << std::endl;
        dummy_theta_(j,new_index)++;
        dummy_theta_.remove_column(old_index);
        // std::cerr << "        Removing weight vector " << std::endl;
        D_prime_--;
        emission_model()->remove_latent_dimension(old_index);
        // std::cerr << "        Removing zero entity count in position "
        //           << d << "," << theta_old << std::endl;
        // std::cerr << "        Old entity_counts = " << kjb::floor(entity_counts_[d]) << std::endl;
        assert(entity_counts(d,theta_old) == 0);
        ++entity_counts(d,theta_new);
        entity_counts_[d].erase(entity_counts_[d].begin() + theta_old);
        // std::cerr << "        New entity_counts = " << kjb::floor(entity_counts_[d]) << std::endl;
        // std::cerr << "        Decrementing feature count in position " << d << std::endl;
        --feature_dimensions_[d];
        // std::cerr << "        Decrementing cumulative feature counts " << d << std::endl;
        sync_cumulative_feature_dimensions();
        // std::cerr << "        Decrementing theta values above old value " << std::endl;
        for(size_t jj = 0; jj < J(); ++jj)
        {
            assert(theta(jj,d) != theta_old);
            if(theta(jj,d) > theta_old) {--theta(jj,d);}
        }
        similarity_model()->get_model()->initialize_params(theta());
        if(theta_new > theta_old) {--new_index;}
    } else {
        // std::cerr << "    Selected existing value for theta." << std::endl;
        dummy_theta_(j,new_index) = 1;
        ++entity_counts(d, theta_new); // need to restore the count that was subtracted
    }
    // std::cerr << "New theta = " << std::endl;
    // std::cerr << kjb::floor(theta_) << std::endl;
    // std::cerr << "New theta_prime = " << std::endl;
    // std::cerr << kjb::floor(dummy_theta_) << std::endl;
    assert(D_prime_ == dummy_theta_.get_num_cols());
    // rebuild the Phi and Delta matrices based on the update
    // std::cerr << "    Syncing phi and delta " << std::endl;
    // similarity_model()->sync_after_theta_update(j,d,theta_old, theta_new);
    // update the latent mean
    // std::cerr << "    Syncing mean matrix " << std::endl;
    // emission_model()->shift_mean_for_state_j(j, 1, new_index);
}

void Categorical_state_model::update_theta_()
{
    //std::cerr << "    Updating theta..." << std::endl;
    PM(verbose_ > 0, "    Updating theta...\n");
    similarity_model()->sync_before_theta_update();
    for(size_t j = 0; j < J(); ++j)
    {
        for(size_t d = 0; d < D(); ++d)
        {
            update_theta_(j,d);
        }
    }
    // std::cerr << theta_ << std::endl;
    //std::cerr << "    Done updating theta." << std::endl;
    PM(verbose_ > 0, "    Done updating theta.\n");
}

Prob_vector Categorical_state_model::get_theta_crp_prior(
    const size_t& d,
    const size_t& current_s
    )
{
    Prob_vector result = kjb::ew_log(entity_counts_.at(d));
    result.push_back(log(alpha_.at(d)));
    result[current_s]--;
    return result;
}

Prob_vector Categorical_state_model::get_theta_transition_log_likelihood(
    const size_t& j,
    const size_t& d
    )
{
    int S = feature_dimensions_[d];
    Prob_vector result(S + 1, 0.0);
    for(int s = 0; s <= S; ++s)
    {
        result[s] = similarity_model()->log_likelihood_for_state(j,d,s);
    }
    return result;
}

void Categorical_state_model::sync_cumulative_feature_dimensions()
{
    // std::cerr << "Syncing feature dimensions" << std::endl;
    // assert(cum_feature_dimensions_.begin() != cum_feature_dimensions_.end());
    Count_vector::const_iterator input_it = feature_dimensions_.begin();
    Count_vector::iterator output_it = cum_feature_dimensions_.begin();
    int accumulator = 0;
    while(output_it != cum_feature_dimensions_.end() && input_it != feature_dimensions_.end())
    {
        accumulator += (*input_it++);
        *(++output_it) = accumulator;
    }
    // std::cerr << "Feature dimensions = " << feature_dimensions_ << std::endl;
    // std::cerr << "Cumulative feature dimensions = " << cum_feature_dimensions_ << std::endl;
}
