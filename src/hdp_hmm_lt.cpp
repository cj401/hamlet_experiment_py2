/* $Id: hdp_hmm_lt.cpp 21443 2017-06-28 18:13:50Z chuang $ */

/*!
 * @file hdp_hmm_lt.cpp
 *
 * @author Colin Dawson 
 */

#include "hdp_hmm_lt.h"
#include "util.h"

HDP_HMM_LT::HDP_HMM_LT(
    const Transition_prior_param_ptr   transition_prior_parameters,
    const Dynamics_param_ptr           dynamics_parameters,
    const State_param_ptr              state_parameters,
    const Emission_param_ptr           emission_parameters,
    const Similarity_param_ptr         similarity_parameters,
    const std::string&                 write_path,
    const unsigned long int            random_seed
    ) : Base_class(write_path, emission_parameters),
        transition_prior_(transition_prior_parameters->make_module()),
        dynamics_model_(dynamics_parameters->make_module()),
        state_model_(state_parameters->make_module()),
        similarity_model_(similarity_parameters->make_module())
{
    kjb::seed_sampling_rand(random_seed);
    initialize_gnu_rng(random_seed);
    J_ = transition_prior_->J_;
    D_ = state_model_->D_;
}

HDP_HMM_LT::HDP_HMM_LT(
    const Transition_prior_ptr transition_prior,
    const Dynamics_model_ptr   dynamics_model,
    const State_model_ptr      state_model,
    const Emission_model_ptr   emission_model,
    const Similarity_model_ptr similarity_model,
    const std::string&         write_path
    ) : Base_class(write_path, emission_model),
        transition_prior_(transition_prior),
        dynamics_model_(dynamics_model),
        state_model_(state_model),
        similarity_model_(similarity_model)
{
    PM(verbose_ > 0, "Creating HDP-HMM from existing modules\n");
    //std::cerr << "Creating HDP-HMM from existing modules" << std::endl;
    PM(verbose_ > 0, "    Linking component models...");
    //std::cerr << "    Linking component models...";
    PM(verbose_ > 0, "transition...");
    //std::cerr << "transition...";
    transition_prior_->set_parent(this);
    PM(verbose_ > 0, "dynamics..");
    //std::cerr << "dynamics...";
    dynamics_model_->set_parent(this);
    PM(verbose_ > 0, "state...");
    //std::cerr << "state...";
    state_model_->set_parent(this);
    PM(verbose_ > 0, "similarity...");
    //std::cerr << "similarity...";
    similarity_model_->set_parent(this);
    PM(verbose_ > 0, "emission...");
    //std::cerr << "emission...";
    emission_model_->set_parent(this);
    PM(verbose_ > 0, "done.\n");
    //std::cerr << "done." << std::endl;
    J_ = transition_prior_->J_;
    D_ = state_model_->D_;
    PMWP(verbose_ > 0, "    Set (J,D) = (%d, %d)\n", (J_)(D_));
    //std::cerr << "    Set (J,D) = ("
    //          << J_ << "," << D_ << ")" << std::endl;
    PM(verbose_ > 0, "HDP-HMM created.");
    //std::cerr << "HDP-HMM created." << std::endl;
}

HDP_HMM_LT::~HDP_HMM_LT()
{
    // gsl_rng_free(gnu_rng);
}

void HDP_HMM_LT::generate_test_sequence(
    const size_t&      num_sequences,
    const size_t&      T,
    const std::string& path
    )
{
    test_NF_ = num_sequences;
    test_T_ = T_list((int) num_sequences, (int) T);
    test_T_.at(0)=T;
    emission_model_->set_test_T(test_T_);
    dynamics_model_->sample_labels_from_prior(test_T_);
    emission_model_->generate_test_observations(test_T_, path + "test_");
    add_test_data(path, num_sequences);
}

/*
 void HDP_HMM_LT::generate_test_sequence(
 const size_t&      T,
 const std::string& path
 )
 {
 T_[0] = T; test_T_ = T;
 emission_model_->set_test_T(T);
 dynamics_model_->sample_labels_from_prior();
 emission_model_->generate_test_observations(T, path + "test_");
 add_test_data(path);
 }
 */

void HDP_HMM_LT::resample_state_and_similarity_models()
{
    state_model_->update_params();
    sync_theta_star_();
    similarity_model_->update_params();
}

/*
void HDP_HMM_LT::resample_transition_model(const Prob_matrix& log_likelihood_matrix)
{
    transition_prior_->update_params();
    sync_transition_matrix_();
    dynamics_model_->update_params(log_likelihood_matrix);
    sync_partition_();
    sync_theta_star_();
    transition_prior_->sync_transition_counts();
    transition_prior_->update_auxiliary_data();
}
 */

void HDP_HMM_LT::resample_transition_model_stage_one()
{
    transition_prior_->update_params();
    sync_transition_matrix_();
}

void HDP_HMM_LT::resample_transition_model_stage_two()
{
    sync_partition_();
    sync_theta_star_();
    transition_prior_->sync_transition_counts();
    transition_prior_->update_auxiliary_data();
}

void HDP_HMM_LT::resample()
{
    resample_state_and_similarity_models();
    resample_transition_model_stage_one();
    if (dynamics_model_->sampling_method == "weak_limit")
    {
        for (size_t i = 0; i < NF(); i++)
        {
            dynamics_model_->update_params(i, log_likelihood_matrix(i));
        }
    }
    else
    {
        KJB_THROW_2(kjb::Not_implemented, "Can only handle weak limit sampling method now!");
        /*
        for (size_t i = 0; i < NF(); i++)
        {
            dynamics_model_->update_params(i, Likelihood_matrix(i));
        }
         */
    }
    resample_transition_model_stage_two();
    // sync_theta_star_();
    emission_model_->update_params();
#ifdef DEBUGGING
    // std::cerr << "Latent State Sequence = " << std::endl;
    // std::cerr << theta_star_.floor() << std::endl;
#endif
}

/*
void HDP_HMM_LT::resample()
{
    resample_state_and_similarity_models();
    resample_transition_model(
        dynamics_model_->sampling_method == "weak_limit" ?
        log_likelihood_matrix() :
        Likelihood_matrix()
        );
    sync_theta_star_();
    emission_model_->update_params();
#ifdef DEBUGGING
    // std::cerr << "Latent State Sequence = " << std::endl;
    // std::cerr << theta_star_.floor() << std::endl;
#endif
}
 */

void HDP_HMM_LT::set_up_results_log() const
{
    PMWP(verbose_ > 0, "Trying to create directory %s\n", (write_path.c_str()));
    //std::cerr << "Trying to create directory " << write_path
    //          << std::endl;
    create_directory_if_nonexistent(write_path);
    // empty_directory(write_path);
    transition_prior_->set_up_results_log();
    dynamics_model_->set_up_results_log();
    state_model_->set_up_results_log();
    similarity_model_->set_up_results_log();
    emission_model_->set_up_results_log();
    create_directory_if_nonexistent(write_path + "thetastar");
    std::ofstream ofs;
    ofs.open(write_path + "train_log_likelihood.txt", std::ofstream::out);
    ofs << "iteration value" << std::endl;
    ofs.close();
    ofs.open(write_path + "test_log_likelihood.txt", std::ofstream::out);
    ofs << "iteration value" << std::endl;
    ofs.close();
}

void HDP_HMM_LT::input_previous_results(const std::string& input_path, const std::string& name)
{
    PMWP(verbose_ > 0, "Inputting information from iteration %s\n", (name.c_str()));
    //std::cerr << "Inputting information from iteration " << name << " ..." << std::endl;
    state_model_->input_previous_results(input_path, name);
    similarity_model_->input_previous_results(input_path, name);
    transition_prior_->input_previous_results(input_path, name);
    sync_transition_matrix_();
    dynamics_model_->input_previous_results(input_path, name);
    sync_partition_();
    sync_theta_star_();
    transition_prior_->sync_transition_counts();
    transition_prior_->update_auxiliary_data();
    sync_theta_star_();
    emission_model_->input_previous_results(input_path, name);
}

void HDP_HMM_LT::write_state_to_file(
    const std::string& name
    ) const
{
    write_state_and_similarity_states_to_file(name);
    write_transition_model_state_to_file(name);
    emission_model_->write_state_to_file(name);
    // Base_class::write_state_to_file(name);
    compute_and_record_marginal_likelihoods(name);
}

void HDP_HMM_LT::write_state_and_similarity_states_to_file(
    const std::string& name
    ) const
{
    state_model_->write_state_to_file(name);
    similarity_model_->write_state_to_file(name);
}

void HDP_HMM_LT::write_transition_model_state_to_file(
    const std::string& name
    ) const
{
    transition_prior_->write_state_to_file(name);
    dynamics_model_->write_state_to_file(name);
}

void HDP_HMM_LT::compute_and_record_marginal_likelihoods(
    const std::string& name
    ) const
{
    /*
     compute the log likelihood for both train and test data
     call the pass message forward function in dynamic model
     */
    //std::cerr << "Computing and recording marginal likelihoods..." << std::endl;
    PM(verbose_ > 0, "Computing and recording marginal likelihoods...\n");
    // std::cerr << "    NF = " << NF() << std::endl;
    // std::cerr << "    T(0) = " << T(0) << std::endl;
    // std::cerr << "    log_likelihood_matrix(0) = "  << log_likelihood_matrix(0)
    //           << std::endl;
    double log_likelihood = 0;
    size_t time = 0;
    for (size_t i = 0; i < NF(); i++)
    {
        log_likelihood +=
            dynamics_model_->get_log_marginal_likelihood(i, log_likelihood_matrix(i));
        time += T(i);
    }
    std::ofstream ofs;
    ofs.open(write_path + "train_log_likelihood.txt", std::ofstream::out | std::ofstream::app);
    ofs << name << " " << log_likelihood / time << std::endl;
    ofs.close();
    if(test_data_exists)
    {
        PM(verbose_ > 0, "    Computing and recording test marginal likelihood...");
        //std::cerr << "    Computing and recording test marginal likelihood...";
        size_t test_time = 0;
        double test_log_likelihood = 0;
        for (size_t i = 0; i < test_NF(); i++)
        {
            test_log_likelihood +=
                dynamics_model_->get_test_log_marginal_likelihood(i, test_log_likelihood_matrix(i));
            test_time += test_T(i);
        }
        ofs.open(write_path + "test_log_likelihood.txt",
                 std::ofstream::out | std::ofstream::app);
        ofs << name << " " << test_log_likelihood / test_time << std::endl;
        ofs.close();
        PM(verbose_ > 0, "done.\n");
        //std::cerr << "done." << std::endl;
    }
    PM(verbose_ > 0, "Finished computing log marginal likelihood.\n");
    //std::cerr << "Finished computing log marginal likelihood." << std::endl;
}

/*
void HDP_HMM_LT::compute_and_record_marginal_likelihoods(
    const std::string& name
    ) const
{
    std::ofstream ofs;
    std::cerr << "Computing and recording marginal likelihoods..." << std::endl;
    ofs.open(write_path + "train_log_likelihood.txt", std::ofstream::out | std::ofstream::app);
    ofs << name << " "
        << dynamics_model_->get_log_marginal_likelihood(log_likelihood_matrix()) / T()
        << std::endl;
    ofs.close();
    if(test_data_exists)
    {
        ofs.open(write_path + "test_log_likelihood.txt",
                 std::ofstream::out | std::ofstream::app);
        ofs << name << " "
            << dynamics_model_->get_log_marginal_likelihood(
                test_log_likelihood_matrix()) / test_T() << std::endl;
        ofs.close();
    }
    // std::cerr << "done." << std::endl;
}
 */

void HDP_HMM_LT::initialize_parent_links()
{
    transition_prior_->set_parent(this);
    dynamics_model_->set_parent(this);
    state_model_->set_parent(this);
    similarity_model_->set_parent(this);
}

void HDP_HMM_LT::set_up_verbose_level(const size_t verbose)
{
    Base_class::set_up_verbose_level(verbose);
    transition_prior_->set_up_verbose_level(verbose);
    dynamics_model_->set_up_verbose_level(verbose);
    state_model_->set_up_verbose_level(verbose);
    similarity_model_->set_up_verbose_level(verbose);
}

void HDP_HMM_LT::initialize_resources_()
{
    PM(verbose_ > 0, "    Allocating resources for state model...\n");
    //std::cerr << "    Allocating resources for state model..." << std::endl;
              // << "(J,D) = (" << J() << "," << D_ << ")" << std::endl;
    state_model_->set_parent(this);
    assert(state_model_->parent == this);
    state_model_->initialize_resources();
    PM(verbose_ > 0, "    State model allocated.\n");
    //std::cerr << "    State model allocated." << std::endl;
    similarity_model_->set_parent(this);
    assert(similarity_model_->parent == this);
    similarity_model_->initialize_resources();
    transition_prior_->set_parent(this);
    assert(transition_prior_->parent == this);
    transition_prior_->initialize_resources();
    dynamics_model_->set_parent(this);
    assert(dynamics_model_->parent == this);
    assert(dynamics_model_->T() == T_);
    dynamics_model_->initialize_resources();
    initialize_partition_();
    PM(verbose_ > 0, "Allocating A...");
    //std::cerr << "Allocating A...";
    A_ = Prob_matrix(J(), J(), 0.0);
    PM(verbose_ > 0, "done.\n");
    //std::cerr << "done." << std::endl;
    Base_class::initialize_resources_();
}

void HDP_HMM_LT::initialize_params_()
{
    initialize_state_and_similarity_models();
    initialize_transition_model();
    emission_model_->initialize_params();
}

void HDP_HMM_LT::initialize_state_and_similarity_models()
{
    state_model_->initialize_params();
    similarity_model_->initialize_params();
}

void HDP_HMM_LT::initialize_transition_model()
{
    transition_prior_->initialize_params();
    sync_transition_matrix_();
    dynamics_model_->initialize_params();
    transition_prior_->sync_transition_counts();
    transition_prior_->update_auxiliary_data();
    sync_partition_();
    sync_theta_star_();
}

void HDP_HMM_LT::initialize_partition_()
{
    /*
     initialize the partition, grouping all time period with same hidden state
     together, used in sampling emission module parameter
     */
    PM(verbose_ > 0, "Initializaing partition...");
    //std::cerr << "Initilizing partition...";
    partition_map_list_ = Partition_map_list(NF());
    for (size_t i = 0; i < NF(); i++)
    {
        partition_map_list_[i].clear();
        for (size_t j = 0; j < J(); ++j)
        {
            partition_map_list_[i].insert(std::make_pair(j, Time_set(0)));
        }
    }
    PM(verbose_ > 0, "done.\n");
    //std::cerr << "done." << std::endl;
}

/*
void HDP_HMM_LT::initialize_partition_()
{
    std::cerr << "Initializing partition...";
    partition_map_.clear();
    for(size_t j = 0; j < J(); ++j)
    {
        partition_map_.insert(std::make_pair(j, Time_set(0)));
    }
    std::cerr << "done." << std::endl;
}
 */

size_t HDP_HMM_LT::D_prime() const {return state_model_->D_prime();}

double HDP_HMM_LT::log_likelihood_ratio_for_state_change(
    const size_t& j,
    const double& delta,
    const size_t& d
    ) const
{
    double result;
    for (size_t i = 0; i < NF(); i++)
    {
        result += emission_model_->log_likelihood_ratio_for_state_change(
            i, theta_prime(j), delta, partition_map(i,j), d);
    }
    return result;
}

/*
double HDP_HMM_LT::log_likelihood_ratio_for_state_change(
    const size_t& j,
    const double& delta,
    const size_t& d
    ) const
{
    return emission_model_->log_likelihood_ratio_for_state_change(
        theta_prime(j), delta, partition_map(j), d);
}
 */

Prob_vector HDP_HMM_LT::log_likelihood_ratios_for_state_range(
    const size_t& j,
    const size_t& d,
    const size_t& first_index,
    const size_t& range_size,
    bool include_new
    ) const
{
    Prob_vector results((int) range_size + (int) include_new, 0.0);
    for (size_t i = 0; i < NF(); i++)
    {
        // std::cerr << "    Getting log likelihood ratios from stream "
        //           << i << " for each possible new value of state "
        //           << j << " in dimension "
        //           << d << " which encompasses columns "
        //           << first_index << " to " << first_index + range_size - 1
        //           << " of dummy_theta (rows of W).  State currently used at positions {"
        //           << partition_map(i,j) << "} in stream " << i
        //           << std::endl;
        // if(include_new)
        // {
        //     std::cerr << "    Also considering a new state in position "
        //               << first_index + range_size << std::endl;
        // }
        results += emission_model_->log_likelihood_ratios_for_state_range(
            i, theta_prime(j), d, first_index, range_size, partition_map(i,j), include_new);
    }
    return results;
}

/*
Prob_vector HDP_HMM_LT::log_likelihood_ratios_for_state_range(
    const size_t& j,
    const size_t& d,
    const size_t& first_index,
    const size_t& range_size,
    bool include_new
    ) const
{
    return emission_model_->log_likelihood_ratios_for_state_range(
        theta_prime(j), d, first_index, range_size, partition_map(j), include_new);
}
 */

void HDP_HMM_LT::sync_partition_()
{
    /*
     synchronize the partition, grouping all the time period with the same
     latent state together, used in sampling emission module parameter
     */
    //std::cerr << "Updating partition...";
    PM(verbose_ > 0, "Updating partition...");
    for (size_t i = 0; i < NF(); ++i)
    {
        for (size_t j = 0; j < J(); ++j)
        {
            partition_map_list_[i].at(j).clear();
        }
        for (size_t t = 1; t <= T(i); ++t)
        {
            partition_map_list_[i].at(z(i,t)).push_back(t-1);
        }
        PM(verbose_ > 0, "done.\n");
        //std::cerr << "done." << std::endl;
    }
}

/*
void HDP_HMM_LT::sync_partition_()
{
    std::cerr << "Updating partition...";
    for(size_t j = 0; j < J(); ++j)
    {
        partition_map_.at(j).clear();
    }
    for(size_t t = 1; t <= T(); ++t)
    {
        partition_map_.at(z(t)).push_back(t-1);
    }
    // for(Partition_map::const_iterator it = partition_map_.begin(); it != partition_map_.end(); ++it)
    // {
    //     std::cerr << it->first << ": " << it->second << std::endl;
    // }
    std::cerr << "done." << std::endl;
}
 */

void HDP_HMM_LT::sync_theta_star_()
{
    /*
     synchronizing the thetastar
     N by D matrix with each row being the location of hidden state j
     */
    PM(verbose_ > 0, "Synchronizing theta*...");
    //std::cerr << "Synchronizing theta*...";
    for (size_t i = 0; i < NF(); i++)
    {
        // std::cerr << "    Theta* " << i << " has dimensions "
        // << theta_star(i).get_num_rows() << " by "
        // << theta_star(i).get_num_cols() << std::endl;
        theta_star(i).realloc(T(i), D_prime());
        for(size_t j = 0; j < J(); ++j)
        {
            Time_set times = partition_map(i, j);
            for (Time_set::const_iterator it = times.begin(); it != times.end(); ++it)
            {
                theta_star(i).set_row((*it), theta_prime(j));
            }
        }
        // std::cerr << "        theta* " << i << " has dimensions "
        // << theta_star(i).get_num_rows() << " by "
        // << theta_star(i).get_num_cols() << std::endl;
    }
    PM(verbose_ > 0, "done.\n");
    //std::cerr << "done." << std::endl;
}

/*
void HDP_HMM_LT::sync_theta_star_()
{
    std::cerr << "Synchronizing theta*...";
    std::cerr << std::endl;
    // std::cerr << "    theta = " << std::endl;
    // std::cerr << kjb::floor(theta()) << std::endl;
    // std::cerr << "    theta_prime = " << std::endl;
    // std::cerr << "    theta* = " << theta_star_ << std::endl;
    std::cerr << "    Theta* has dimensions "
              << theta_star_.get_num_rows() << " by "
              << theta_star_.get_num_cols() << std::endl;
    // std::cerr << kjb::floor(*(state_model_->theta_prime_)) << std::endl;
    theta_star_.realloc(T(), D_prime());
    for(size_t j = 0; j < J(); ++j)
    {
        Time_set times = partition_map(j);
        // std::cerr << "Partition(" << j << ") = " << times << std::endl;
        for (Time_set::const_iterator it = times.begin(); it != times.end(); ++it) 
        {
            // std::cerr << "    Current theta* = " << std::endl;
            // std::cerr << theta_star_ << std::endl;
            // std::cerr << "    Setting row " << *it << " to " << kjb::floor(theta_prime(j))
            //           << std::endl;
            theta_star_.set_row((*it), theta_prime(j));
        }
    }
    std::cerr << "        theta* has dimensions "
              << theta_star_.get_num_rows() << " by "
              << theta_star_.get_num_cols() << std::endl;
    std::cerr << "done." << std::endl;
}
 */

void HDP_HMM_LT::sync_transition_matrix_()
{
    /*
     synchronizing the transition matrix A_ used in sampling hidden sequence
     A_ is in log and unnormalized a_{j,j'} = pi_{j,j'} * phi_{j,j'}
     Refer to equation (8) in the paper for more details
     
     pi() and phi() from transition_prior module
     */
    PM(verbose_ > 0, "Synchronizing transition_matrix...");
    //std::cerr << "Synchronizing transition_matrix...";
    A_ = pi() + Phi();
    dynamics_model_->mask_A();
    A_ = kjb::log_normalize_rows(A_);
    PM(verbose_ > 0, "done.\n");
    //std::cerr << "done." << std::endl;
}
