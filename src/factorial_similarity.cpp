// /* $Id: factorial_similarity.cpp 20751 2016-07-10 20:02:13Z cdawson $ */

// /*!
//  * @file factorial_similarity.cpp
//  *
//  * @author Colin Dawson 
//  */

// #include "hdp_hmm_lt.h"
// #include "factorial_similarity.h"
// #include "hdp_transition_prior.h"
// #include "underflow_utils.h"
// #include <boost/make_shared.hpp>

// Similarity_model_ptr Factorial_similarity_parameters::make_module() const
// {
//     return boost::make_shared<Factorial_similarity>(this);
// }

// State_sequence Factorial_similarity::get_state_stream(const size_t& d)
// {
//     return kjb::floor(parent->theta_star().get_col(d));
// };

// void Factorial_similarity::initialize_resources()
// {
//     Base_class::initialize_resources();
//     for(size_t d = 0; d < D(); ++d)
//     {
//         chain_priors_.push_back(tpp_->make_module());
//         std::cerr << "    Allocating chain " << d << std::endl;
//         (*chain_priors_.rbegin())->initialize_resources();
//     }
//     lambda_ = Prob_matrix((int) D(), 3);
// }

// void Factorial_similarity::initialize_params()
// {
//     std::cerr << "Initializing Factorial HMM params..." << std::endl;
//     for(size_t d = 0; d < D(); ++d)
//     {
//         std::cerr << "    Initializing chain " << d << std::endl;
//         chain_priors_[d]->initialize_params();
//     }
//     lambda_.set_col(1, Prob_vector((int) D(), 0.0));
//     sync_lambda_();
//     sync_phi_();
// }

// void Factorial_similarity::update_params()
// {
//     std::cerr << "Updating Factorial HMM params..." << std::endl;
//     for(size_t d = 0; d < D(); ++d)
//     {
//         std::cerr << "    Updating chain " << d << std::endl;
//         chain_prior(d)->sync_transition_counts(get_state_stream(d));
//         chain_prior(d)->update_auxiliary_data();
//         chain_prior(d)->update_params();
//     }
//     sync_lambda_();
//     sync_phi_();
// }

// void Factorial_similarity::sync_lambda_()
// {
//     for(size_t d = 0; d < D(); ++d)
//     {
//         lambda_(d,0) = chain_prior(d)->pi(1,0) - chain_prior(d)->pi(1,1);
//         lambda_(d,2) = chain_prior(d)->pi(0,1) - chain_prior(d)->pi(0,0);
//     }
// }

// void Factorial_similarity::sync_phi_()
// {
//     for(size_t j = 0; j < J(); ++j)
//     {
//         for(size_t jp = 0; jp < J(); ++jp)
//         {
//             Phi(j,jp) = 0;
//             State_type d_theta = (theta_prime(jp) - theta_prime(j)) += 1;
//             for(size_t d = 0; d < D(); ++d)
//             {
//                 Phi(j,jp) += lambda_(d, d_theta[d]);
//             }
//         }
//     }
// }
