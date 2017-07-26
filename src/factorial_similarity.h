// /* $Id: factorial_similarity.h 20751 2016-07-10 20:02:13Z cdawson $ */

// #ifndef FACTORIAL_SIMILARITY_H_
// #define FACTORIAL_SIMILARITY_H_

// /*!
//  * @file factorial_similarity.h
//  *
//  * @author Colin Dawson 
//  */

// #include "similarity_model.h"
// #include "state_model.h"
// #include "dirichlet_transition_prior.h"
// #include "hdp_transition_prior.h"
// #include "parameters.h"

// class Factorial_similarity;
// struct Factorial_similarity_parameters;

// struct Factorial_similarity_parameters : public Similarity_model_parameters
// {
//     const bool   fixed_mean;
//     const double p_mean;
//     const boost::shared_ptr<Transition_prior_parameters> tpp;
    
//     Factorial_similarity_parameters(
//         const Parameters&  params,
//         const std::string& name = "Binary_factorial"
//         ) : Similarity_model_parameters(),
//             fixed_mean(params.exists(name, "p_mean")),
//             p_mean(
//                 !fixed_mean ? NULL_BETA_PARAM :
//                 params.get_param_as<double>(
//                     name, "p_mean", bad_beta_prior(), valid_prob_param)),
//             tpp(fixed_mean ?
//                 boost::make_shared<Dirichlet_transition_prior_parameters>(
//                     params, 2, Prob_vector(1.0 - p_mean, p_mean)) :
//                 boost::make_shared<HDP_transition_prior_parameters>(
//                     params, 2))
//     {}

//     virtual ~Factorial_similarity_parameters() {}

//     virtual Similarity_model_ptr make_module() const;

//     static const std::string& bad_alpha_prior()
//     {
//         static const std::string msg = 
//             "ERROR: For a Factorial Binary HMM model,"
//             " config file must specify either alpha > 0,"
//             " or both positive a_alpha (shape) and b_alpha (rate) "
//             " hyperparameters.";
//         return msg;
//     }
    
//     static const std::string& bad_beta_prior()
//     {
//         static const std::string msg = 
//             "ERROR: For a Factorial Binary HMM model,"
//             " config file must specify one of p_mean in [0,1],"
//             " gamma > 0, or both positive a_gamma (shape) and b_gamma (rate) "
//             " hyperparameters.";
//         return msg;
//     }
// };

// class Factorial_similarity : public Similarity_model
// {
//     typedef Factorial_similarity Self;
//     typedef Similarity_model Base_class;
//     typedef Factorial_similarity_parameters Params;
//     typedef boost::shared_ptr<Transition_prior> Chain_prior_ptr;
// public:
//     Factorial_similarity(const Params* const hyperparams)
//         : fixed_mean_(hyperparams->fixed_mean),
//           p_mean_(hyperparams->p_mean),
//           tpp_(hyperparams->tpp)
//     {}

//     virtual ~Factorial_similarity() {}

//     virtual void initialize_resources();

//     virtual void initialize_params();

//     virtual void update_params();
    
//     virtual void set_up_results_log() const {};

//     virtual void write_state_to_file(const std::string&) const {};

// protected:
//     State_sequence get_state_stream(const size_t& d);
//     void sync_lambda_();
//     void sync_phi_();
//     const Chain_prior_ptr chain_prior(const size_t& d) const
//     {
//         return chain_priors_.at(d);
//     }

// public:
//     const bool fixed_mean_;
//     const double p_mean_;
//     const boost::shared_ptr<Transition_prior_parameters> tpp_;

//     std::vector<Chain_prior_ptr> chain_priors_;
//     Prob_matrix lambda_;
// };

// #endif
