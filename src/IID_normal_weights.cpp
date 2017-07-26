// /* $Id: IID_normal_weights.cpp 21021 2017-01-06 01:08:30Z chuang $ */

// /*!
//  * @file IID_normal_weights.cpp
//  *
//  * @author Colin Dawson
//  */

// #include "IID_normal_weights.h"
// #include <m_cpp/m_matrix.h>
// #include <boost/make_shared.hpp>

// Weight_prior_ptr IID_normal_weights_parameters::make_module() const
// {
//     return boost::make_shared<IID_normal_weights>(this);
// }

// const State_matrix& IID_normal_weights::theta_star() const
// {
//     return parent->theta_star();
// }

// void IID_normal_weights::sample_h_and_mu_from_prior_()
// {
    // std::cerr << "    Generating h_w and mu_w...";
    // Precision_dist r_h_w(a_h_w_, 1.0 / b_h_w_);
    // Precision_dist r_h_b(a_h_b_, 1.0 / b_h_b_);
    // Normal_dist r_mu_w(0.0, 1.0 / h_mu_w_);
    // Normal_dist r_mu_b(0.0, 1.0 / h_mu_b_);
    // h_w_ = kjb::sample(r_h_w);
    // h_b_ = kjb::sample(r_h_b);
    // mu_w_ = kjb::sample(r_mu_w);
    // mu_b_ = kjb::sample(r_mu_b);
    // std::cerr << "done.";
// }

// IID_normal_weights::Weight_vector IID_normal_weights::sample_w_row_from_prior(
//     const size_t&) const
// {
//     kjb::Vector w_sigma2 = kjb::Vector((int) K(), sigma_2_w_);
//     kjb::Vector prior_mean((int) K(), 0.0);
//     kjb::Matrix prior_covariance = kjb::create_diagonal_matrix(w_sigma2);
//     kjb::MV_gaussian_distribution r_wk(prior_mean, prior_covariance);
//     return kjb::sample(r_wk);
// }

// void IID_normal_weights::sample_W_from_prior_()
// {
//     W_ = kjb::Matrix(D_prime(), K(), 0.0);
//     std::cerr << "    Generating W...";
//     for(size_t d = 0; d < D_prime(); ++d)
//     {
//         W_.set_row(d, sample_w_row_from_prior(d));
//     }
//     if(include_bias_)
//     {
//         kjb::Vector b_sigma2 = kjb::Vector((int) K(), sigma_2_b_);
//         kjb::Vector b_mean = kjb::Vector((int) K(), 0.0);
//         kjb::Matrix b_covariance = kjb::create_diagonal_matrix(b_sigma2);
//         kjb::MV_gaussian_distribution r_bias(b_mean, b_covariance);
//         b_ = kjb::sample(r_bias);
//     }
//     std::cerr << "done." << std::endl;
// }

// void IID_normal_weights::update_W_()
// {
//     std::cerr << "    Updating W...";
//     kjb::Vector w_h = kjb::Vector((int) D_prime(), 1.0 / sigma_2_w_);
//     kjb::Vector b_h = include_bias_ ? kjb::Vector(1, 1.0 / sigma_2_b_) : kjb::Vector();
//     kjb::Matrix prior_precision =
//         kjb::create_diagonal_matrix(kjb::cat_vectors(w_h, b_h));
//     kjb::Matrix theta_star_transpose =
//         kjb::matrix_transpose(theta_star()); // (D+1) x T
//     kjb::Matrix likelihood_precision =
//         (theta_star_transpose * theta_star()); // (D+1) x (D+1)
//     kjb::Matrix data_projection =
//         theta_star_transpose * parent->noisy_data(); // (D+1) x K
//     kjb::Matrix posterior_covariance =
//         kjb::matrix_inverse(prior_precision + likelihood_precision);
//     kjb::Matrix posterior_mean =
//         posterior_covariance * data_projection;
//     Weight_matrix W_new((int) D_prime() + include_bias_, (int) K());
//     for(size_t k = 0; k < K(); ++k)
//     {
//         kjb::MV_gaussian_distribution r_wk(
//             posterior_mean.get_col(k),
//             1.0 / noise_parameters().at(k,0) * posterior_covariance
//             );
//         W_new.set_col(k, kjb::sample(r_wk));
//     }
//     if(include_bias_)
//     {
//         W_ = W_new.submatrix(0, 0, D_prime(), K());
//         b_ = W_new.get_row(D_prime());
//     } else {
//         W_ = W_new;
//     }
//     std::cerr << "done." << std::endl;
// }

// void IID_normal_weights::update_h_and_mu_()
// {
    // kjb::Vector sum_ws = kjb::sum_matrix_cols(W_);
    // assert(sum_ws.size() == D() + 1);
    // double sum_w = std::accumulate(sum_ws.begin(), sum_ws.end() - 1, 0.0);
    // double sum_b = sum_ws[D()];
    // kjb::Matrix w_mean_matrix((int) D(), (int) K(), mu_w_);
    // kjb::Matrix b_mean_matrix(1, (int) K(), mu_b_);
    // w_mean_matrix.vertcat(b_mean_matrix); // D+1 x K
    // kjb::Vector sum_sq_devs =
    //     kjb::sum_matrix_cols(kjb::get_squared_elements(W_ - w_mean_matrix));
    // double w_sse = std::accumulate(sum_sq_devs.begin(), sum_sq_devs.end() - 1, 0.0);
    // double b_sse = sum_sq_devs[D()];
    // Precision_dist r_w_h(a_h_w_ + K() * D() / 2.0, 2.0 / (2.0 * b_h_w_ + w_sse));
    // Precision_dist r_b_h(a_h_b_ + K() / 2.0, 2.0 / (2.0 * b_h_b_ + b_sse));
    // h_w_ = kjb::sample(r_w_h);
    // h_b_ = kjb::sample(r_b_h);
    // double post_mu_w_h = h_mu_w_ + D() * K() * h_w_;
    // double post_mu_b_h = h_mu_b_ + K() * h_b_;
    // Normal_dist r_mu_w(h_w_ * sum_w / post_mu_w_h, 1.0 / post_mu_w_h);
    // Normal_dist r_mu_b(h_b_ * sum_b / post_mu_b_h, 1.0 / post_mu_b_h);
    // mu_w_ = kjb::sample(r_mu_w);
    // mu_b_ = kjb::sample(r_mu_b);
    // std::cerr << "w ~ N(" << mu_w_ << "," << sqrt(1.0 / h_w_) << ")" << std::endl;
    // std::cerr << "b ~ N(" << mu_b_ << "," << sqrt(1.0 / h_b_) << ")" << std::endl;
// }

