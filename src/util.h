/* $Id: util.h 21509 2017-07-21 17:39:13Z cdawson $ */

#ifndef UTIL_H_
#define UTIL_H_

/*!
 * @file util.h
 *
 * @author Colin Dawson 
 */
#include <l_cpp/l_stdio_wrap.h>
#include <l_cpp/l_util.h>
#include <l_cpp/l_int_matrix.h>
#include <l_cpp/l_index.h>
#include <m_cpp/m_matrix.h>
#include <gsl_cpp/gsl_rng.h>
#include <third_party/rtnorm.hpp>
#include <third_party/underflow_utils.h>
#include <prob_cpp/prob_util.h>
#include <prob_cpp/prob_distribution.h>
#include <prob_cpp/prob_sample.h>
#include <boost/random.hpp>
#include <boost/bind.hpp>
#include <boost/unordered_map.hpp>
//need to include boost algorithm string
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <fstream>
//include for the macro
#include <boost/preprocessor/seq.hpp>
//end change
#include <map>
#include <cmath>

#ifdef DEBUGGING
#define PM(condition, msg)                          \
    do                                              \
    {                                               \
        if (condition)                              \
        {                                           \
            printf(msg);                            \
        }                                           \
    }                                               \
    while(0)

#define PMWP(condition, msg, params)                \
    do                                              \
    {                                               \
        if (condition)                              \
        {                                           \
            printf(msg,                             \
                   BOOST_PP_SEQ_ENUM(params));      \
        }                                           \
    }                                               \
    while(0)
#else
#define PM(condition, msg)
#define PMWP(condition, msg, params)
#endif

#define NULL_SHAPE -1.0
#define NULL_RATE -1.0
#define NULL_CONC -1.0
#define NULL_BETA_PARAM -1.0
#define NULL_STRING ""

class HMM_base;
class HDP_HMM_LT;
class State_model;
class Emission_model;
class Transition_model;
struct Transition_prior_parameters;
class Transition_prior;
struct Dynamics_parameters;
class Dynamics_model;
class Similarity_model;

typedef kjb::Matrix Mean_matrix;
typedef kjb::Vector Mean_vector;
typedef kjb::Vector Scale_vector;
typedef boost::shared_ptr<Transition_prior> Transition_prior_ptr;
typedef boost::shared_ptr<State_model> State_model_ptr;
typedef boost::shared_ptr<Emission_model> Emission_model_ptr;
typedef boost::shared_ptr<Similarity_model> Similarity_model_ptr;
typedef boost::shared_ptr<Dynamics_model> Dynamics_model_ptr;
typedef boost::shared_ptr<const Transition_prior> Transition_prior_const_ptr;
typedef boost::shared_ptr<const State_model> State_model_const_ptr;
typedef boost::shared_ptr<const Emission_model> Emission_model_const_ptr;
typedef boost::shared_ptr<const Similarity_model> Similarity_model_const_ptr;
typedef boost::shared_ptr<const Dynamics_model> Dynamics_model_const_ptr;

typedef std::vector<Mean_matrix> Mean_matrix_list;

/// Variable types
typedef double Prob;
typedef double Conc;
typedef int Count;
typedef double Cts_duration;
typedef double Distance;
typedef size_t Dsct_duration;
typedef int State_indicator;
typedef size_t Time_index;
typedef std::vector<Time_index> Time_set;
typedef kjb::Matrix State_matrix;
typedef kjb::Vector State_type;
typedef State_matrix::Value_type Coordinate;
typedef kjb::Int_vector  State_sequence;
typedef kjb::Vector Prob_vector;
typedef kjb::Const_matrix_vector_view Const_prob_row_view;
typedef kjb::Matrix_vector_view Prob_row_view;
typedef kjb::Matrix Prob_matrix;
typedef std::vector<Count> Count_vector;
typedef kjb::Int_matrix Count_matrix;
typedef kjb::Int_matrix Binary_matrix;
typedef kjb::Int_vector Binary_vector;
typedef std::vector<Cts_duration> Time_array;
typedef kjb::Vector Likelihood_vector;
typedef kjb::Matrix Likelihood_matrix;
typedef kjb::Matrix Distance_matrix;
typedef boost::unordered_map<State_indicator, Time_set> Partition_map;

typedef std::vector<State_sequence> State_sequence_list;
typedef kjb::Int_vector T_list;
typedef std::vector<State_matrix> State_matrix_list;
typedef std::vector<Partition_map> Partition_map_list;

/// Distribution types
// typedef boost::gamma_distribution<> Gamma_dist;
// typedef boost::poisson_distribution<> Poisson_dist;
// typedef boost::bernoulli_distribution<> Bernoulli_dist;
typedef kjb::Gamma_distribution Gamma_dist;
typedef kjb::Poisson_distribution Poisson_dist;
typedef kjb::Bernoulli_distribution Bernoulli_dist;
typedef kjb::Beta_distribution Beta_dist;
typedef kjb::Normal_distribution Normal_dist;
typedef Gamma_dist        Conc_dist;
typedef Gamma_dist        Rate_dist;
typedef Gamma_dist        Precision_dist;
typedef Gamma_dist        Dur_dist;
typedef Poisson_dist      Count_dist;

/*
void score_binary_states(
    const std::string& results_path,
    const std::string& ground_truth_path,
    const std::string& label,
    const State_matrix& state_matrix
    );
 */
void score_binary_states(
    const std::string& results_path,
    const std::string& ground_truth_path,
    const std::string& label,
    const State_matrix_list& state_matrix_list
    );

void add_binary_gt_eval_header(const std::string& results_path);

inline double log_of_one_more_than(const double& u)
{
    return LogOnePlusX(u);
}

inline double log_normalize_by(const double& x, const double& logmax) {return x - logmax;}

template<class InputIt>
void log_normalize_vector_in_place(InputIt first, InputIt last)
{
    const double total_mass = kjb::log_sum(first, last);
    std::transform(first, last, first, boost::bind(log_normalize_by, _1, total_mass));
}

void threshold_a_double(
    double& x,
    const std::string& write_path,
    const std::string& message
    );

/**
 * @brief create a matrix whose rows are binary representations of ints from 0 to 2^(num_cols - 1)
 *
 * Noticeable lag starts to happen when num_cols is >= about 20
 */
Binary_matrix generate_matrix_of_binary_range(const size_t& num_cols);

template<class InputIt, class OutputIt>
void sample_log_from_dirichlet_posterior_with_symmetric_prior(
    const double&  prior_weight_per_value,
    const InputIt  params_update_first,
    const InputIt  params_update_last,
    const OutputIt d_first,
    const OutputIt d_last,
    const std::string& write_path
    )
{
    OutputIt oit = d_first;
    double val;
    for(InputIt it = params_update_first; it != params_update_last; ++it, ++oit)
    {
        double shape = prior_weight_per_value + (*it);
        threshold_a_double(shape, write_path, "beta");
        Gamma_dist r_gamma(shape, 1.0);
        val = log(kjb::sample(r_gamma));
        // (*oit) = val < thresh ? thresh : val;
        (*oit) = val;
    }
    log_normalize_vector_in_place(d_first, d_last);
}

void sample_mv_normal_vectors_from_row_means(
    kjb::Matrix&        result,
    const kjb::Matrix&  means,
    const kjb::Matrix&  cov
    );

template<typename T>
std::ostream& operator<<(std::ostream& os, std::vector<T> d)
{
    for (typename std::vector<T>::const_iterator d_it = d.begin(); d_it != d.end(); ++d_it) 
    {
        os << *d_it << " ";
    }
    return os;
}

template<typename T>
std::ostream& operator<<(std::ostream& os, std::vector<std::vector<T> > d)
{
    os << "[" << std::endl;
    for (typename std::vector<std::vector<T> >::const_iterator r_it = d.begin(); r_it != d.end(); ++r_it) 
    {
        os << "(";
        for(typename std::vector<T>::const_iterator e_it = (*r_it).begin();
            e_it != (*r_it).end(); ++e_it)
        {
            os << *e_it << ", ";
        }
        os << ")" << std::endl;
    }
    os << "]";
    return os;
}

//new function to input a vector from a file
template<typename T>
T input_to_value(
    const std::string& write_path,
    const std::string& file_name,
    const std::string& name)
{
    //input data
    std::string iteration_input;
    std::vector <std::string> input_vector;
    std::ifstream input_file((write_path + file_name).c_str());
    while (getline(input_file, iteration_input))
    {
        input_vector.push_back(iteration_input);
    }
    input_file.close();
    //find iteration number
    std::vector<std::string> split_vector;
    T data_value;
    for (std::vector<std::string>::reverse_iterator rit = input_vector.rbegin();
         rit != input_vector.rend(); ++rit)
    {
        boost::split(split_vector, *rit, boost::is_any_of(" "));
        if (split_vector[0] == name)
        {
            data_value = boost::lexical_cast<T>(split_vector[1]);
            break;
        }
    }
    return data_value;
}

template<typename T>
std::vector<T> input_to_vector(
    const std::string& write_path,
    const std::string& file_name,
    const std::string& name)
{
    //input data
    std::string iteration_input;
    std::vector <std::string> input_vector;
    std::ifstream input_file((write_path + file_name).c_str());
    while (getline(input_file, iteration_input))
    {
        input_vector.push_back(iteration_input);
    }
    input_file.close();
    //find iteration number
    std::vector<std::string> split_vector;
    std::vector<T> data_vector;
    for (std::vector<std::string>::reverse_iterator rit = input_vector.rbegin();
         rit != input_vector.rend(); ++rit)
    {
        boost::split(split_vector, *rit, boost::is_any_of(" "));
        if (split_vector[0] == name)
        {
            for (std::vector<std::string>::iterator it = std::next(split_vector.begin());
                 it != split_vector.end(); ++it)
            {
                try{
                    data_vector.push_back(boost::lexical_cast<T>(*it));
                } catch(boost::bad_lexical_cast &){
                    continue;
                }
            }
            break;
        }
    }
    return data_vector;
}

/* /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ */ 

static gsl_rng* gnu_rng;

void initialize_gnu_rng(const unsigned long int& seed);
double sample_left_truncated_normal(double lower);
double sample_right_truncated_normal(double upper);

int create_directory_if_nonexistent(const std::string& dir_name);
int empty_directory(const std::string& dir_name);

#endif

