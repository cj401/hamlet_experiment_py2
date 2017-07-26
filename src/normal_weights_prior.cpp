/* $Id: normal_weights_prior.cpp 21443 2017-06-28 18:13:50Z chuang $ */

/*!
 * @file Normal_weights_prior.cpp
 *
 * @author Colin Dawson 
 */

#include "util.h"
#include "normal_weights_prior.h"
#include <m_cpp/m_matrix.h>
#include <boost/make_shared.hpp>

Weight_prior_ptr Normal_weights_prior_parameters::make_module() const
{
    return boost::make_shared<Normal_weights_prior>(this);
}

void Normal_weights_prior::initialize_params()
{
    PM(verbose_ > 0, "    Initializing weights...\n");
    //std::cerr << "    Initializing weights..." << std::endl;
    // sample_h_and_mu_from_prior_();
    if(equal_variances)
    {
        PM(verbose_ > 0, "         Initializing exchangeable prior covariance matrix...");
        //std::cerr << "        Initializing exchangeable prior covariance matrix...";
        prior_variances_ =
            Weight_matrix(D_prime(), K(), sigma_2_w);
        if(include_bias_)
        {
            prior_variances_.vertcat(Weight_matrix(1, (int) K(), sigma_2_b));
        }
        prior_precision_matrices_ =
            create_constant_precision_matrices_(sigma_2_w, sigma_2_b);
        PM(verbose_ > 0, "done.\n");
        //std::cerr << "done." << std::endl;
        // std::cerr << "Prior variances = " << std::endl;
        // std::cerr << prior_variances_ << std::endl;
        // std::cerr << "Prior precision [0]= " << std::endl;
        // std::cerr << prior_precision_matrices_[0] << std::endl;
    } else {
        PMWP(verbose_ > 0,
             "       Initializing prior covariance matrix from file %s...",
             (prior_variance_file.c_str()));
        //std::cerr << "        Initializing prior covariance matrix from file "
        //          << prior_variance_file << "...";
        prior_variances_ = Weight_matrix(prior_variance_file);
        prior_precision_matrices_ = read_and_format_precision_matrices_();
        IFT(prior_variances_.get_num_rows() == (int) D() + include_bias_,
            kjb::IO_error, "Normal_weights prior_variances_file has incorrect"
            " number of rows.  Should equal D param of state model, plus one if"
            " a bias weight is included.");
        IFT(prior_variances_.get_num_cols() == (int) K(),
            kjb::IO_error, "Normal_weights prior_variances_file has incorrect"
            " number of columns.  Should equal number of observable dimensions.");
        PM(verbose_ > 0, "done.\n");
        //std::cerr << "done." << std::endl;
    }
    if(zero_mean)
    {
        PM(verbose_ > 0, "      Initiailizing prior means to zero...");
        //std::cerr << "        Initializing prior means to zero...";
        prior_means_ =
            Weight_matrix(D() + (int) include_bias_, K(), 0.0);
        PM(verbose_ > 0, "done.\n");
        //std::cerr << "done." << std::endl;
        // std::cerr << "Prior means = " << std::endl;
        // std::cerr << prior_means_ << std::endl;
    } else {
        PM(verbose_ > 0, "        Initializing prior means from file..");
        //std::cerr << "        Initializing prior means from file...";
        prior_means_ = Weight_matrix(prior_mean_file);
        IFT(prior_means_.get_num_rows() == (int) D() + include_bias_,
            kjb::IO_error, "Normal_weights prior_means_file has incorrect"
            " number of rows.  Should equal D param of state model, plus one if"
            " a bias weight is included.");
        IFT(prior_means_.get_num_cols() == (int) K(),
            kjb::IO_error, "Normal_weights prior_means_file has incorrect"
            " number of columns.  Should equal number of observable dimensions.");
        PM(verbose_ > 0, "done.\n");
        //std::cerr << "done." << std::endl;
    }
    sample_W_from_prior_();
    std::cerr << std::endl;
}

void Normal_weights_prior::input_previous_results(const std::string& input_path, const std::string& name)
{
    PM(verbose_ > 0, "Inputting information for normal weight prior...\n");
    //std::cerr << "Inputting information for normal weight prior..." << std::endl;
    PM(verbose_ > 0, "Inputting information for W_...\n");
    //std::cerr << "Inputting information for W_" << std::endl;
    W_ = Weight_matrix((input_path + "W/" + name + ".txt").c_str());
    if(include_bias_)
    {
        b_ = W_.get_row(D_prime());
        W_ = W_.submatrix(0,0,D_prime(), K());
    }
    PM(verbose_ > 0, "done.\n");
    //std::cerr << "done." << std::endl;
}

const State_matrix& Normal_weights_prior::theta_star(const size_t& i) const
{
    return parent->theta_star(i);
}

/*
const State_matrix& Normal_weights_prior::theta_star() const
{
    return parent->theta_star();
}
 */

Normal_weights_prior::Weight_vector
Normal_weights_prior::sample_w_row_from_prior(const size_t& d) const
{
    kjb::Matrix prior_covariance =
        kjb::create_diagonal_matrix(prior_variances_.get_row(d));
    kjb::Vector prior_mean = prior_means_.get_row(d);
    // std::cerr << "        Prior mean[ " << d << "] = " << std::endl;
    // std::cerr << prior_mean << std::endl;
    // std::cerr << "        Prior variance[ " << d << "] = " << std::endl;
    // std::cerr << prior_covariance << std::endl;
    kjb::MV_gaussian_distribution r_wk(prior_mean, prior_covariance);
    return kjb::sample(r_wk);
}

size_t Normal_weights_prior::dimension_map(const size_t& dprime) const
{
    return parent->dimension_map(dprime);
}

void Normal_weights_prior::insert_latent_dimension(
    const size_t& d,
    const size_t& new_pos
    )
{
    // std::cerr << "        Adding a latent dimension (now "
    //           << D_prime() << ")." << std::endl;
    Base_class::insert_latent_dimension(d, new_pos);
    // std::cerr << "        Syncing prior dimensions...";
    sync_prior_dimensions();
    // std::cerr << "done." << std::endl;
}

void Normal_weights_prior::remove_latent_dimension(const size_t& old_pos)
{
    // std::cerr << "        Removing a latent dimension (now "
    //           << D_prime() << ")." << std::endl;
    Base_class::remove_latent_dimension(old_pos);
    // std::cerr << "        Syncing prior dimensions...";
    sync_prior_dimensions();
    // std::cerr << "done." << std::endl;
}

void Normal_weights_prior::replace_latent_dimension(const size_t& d, const size_t& pos)
{
    Base_class::replace_latent_dimension(d, pos);
}

void Normal_weights_prior::sync_prior_dimensions()
{
    //TODO: Handle unequal cases
    // std::cerr << "    Synching weight prior dimensions after theta dimension change."
    //           << std::endl;
    if(equal_variances)
    {
        prior_precision_matrices_ =
            create_constant_precision_matrices_(sigma_2_w, sigma_2_b);
    } else {
        // std::cerr << "            Updating covariance matrix dimensions...";
        prior_precision_matrices_ =
            read_and_format_precision_matrices_();
        // std::cerr << "done." << std::endl;
    }
    if(zero_mean)
    {
        prior_mean_matrix_ =
            Weight_matrix(D_prime() + (int) include_bias_, K(), 0.0);
    } else {
        // std::cerr << "            Updating mean matrix dimensions...";
        prior_mean_matrix_.resize((int) D_prime(), (int) K());
        for(int dprime = 0; dprime < D_prime(); ++dprime)
        {
            int d = dimension_map(dprime);
            prior_mean_matrix_.set_row(dprime, prior_means_.get_row(d));
        }
        // std::cerr << "done." << std::endl;
    }
}

void Normal_weights_prior::sample_W_from_prior_()
{
    W_ = Weight_matrix(D_prime() + (int) include_bias_, K(), 0.0);
    PM(verbose_ > 0, "    Generating W...");
    //std::cerr << "    Generating W...";
    for(size_t dprime = 0; dprime < D_prime() + (int) include_bias_; ++dprime)
    {
        W_.set_row(dprime, sample_w_row_from_prior(dimension_map(dprime)));
    }
    if(include_bias_)
    {
        b_ = W_.get_row(D_prime());
        W_ = W_.submatrix(0,0,D_prime(), K());
    }
    PM(verbose_ > 0, "done.\n");
    //std::cerr << "done." << std::endl;
}

void Normal_weights_prior::update_W_()
{
    /*
     Sample W using standard methods for Bayesian linear regression
     Refer to section 4.1 "Sampling W and Sigma" in th paper
     */
    PM(verbose_ > 0, "    Updating W...");
    //std::cerr << "    Updating W...";
    kjb::Matrix theta_star_all = theta_star(0);
    kjb::Matrix noisy_data_all = parent->noisy_data(0);
    for (size_t i = 1; i < NF(); i++)
    {
        theta_star_all.vertcat(theta_star(i));
        noisy_data_all.vertcat(parent->noisy_data(i));
    }
    // std::cerr << "         Concatenated theta* is "
    //           << theta_star_all.get_num_rows() << " x "
    //           << theta_star_all.get_num_cols() << std::endl;
    // std::cerr << "         Concatenated data is "
    //           << noisy_data_all.get_num_rows() << " x "
    //           << noisy_data_all.get_num_cols() << std::endl;
    kjb::Matrix theta_star_all_transpose =
        kjb::matrix_transpose(theta_star_all);
    kjb::Matrix likelihood_precision =
        (theta_star_all_transpose * theta_star_all);
    kjb::Matrix data_projection =
        (theta_star_all_transpose * noisy_data_all);
    kjb::Matrix mle_weights = likelihood_precision * data_projection;
    Weight_matrix W_new((int) D_prime() + include_bias_, (int) K());
    // std::cerr << "W_new is " << D_prime() << " x " << K() << std::endl;
    for(size_t k = 0; k < K(); ++k)
    {
        // std::cerr << "Output Dimension " << k << ":" << std::endl;
        // std::cerr << "    Prior precision is "
        //           << prior_precision_matrices_.at(k).get_num_rows()
        //           << " x " << prior_precision_matrices_.at(k).get_num_cols()
        //           << std::endl;
        // std::cerr << "    Likelihood precision is "
        //           << likelihood_precision.get_num_rows()
        //           << " x " << likelihood_precision.get_num_cols()
        //           << std::endl;
        // std::cerr << "    D_prime = " << D_prime() << std::endl;
        // std::cerr << "    Posterior precision = " << std::endl;
        kjb::Matrix posterior_precision = 
            prior_precision_matrices_.at(k) +
            noise_parameters().at(k,0) * likelihood_precision;
        // std::cerr << posterior_precision << std::endl;
        kjb::Matrix posterior_covariance = kjb::matrix_inverse(posterior_precision);
        // std::cerr << "    Posterior covariance = " << std::endl;
        // std::cerr << posterior_covariance << std::endl;
        // std::cerr << "    Prior mean = " << std::endl;
        // std::cerr << prior_means_.get_col(k) << std::endl;
        // std::cerr << "    MLE_weights = " << std::endl;
        // std::cerr << mle_weights.get_col(k) << std::endl;
        // std::cerr << "    Reweighted Prior Mean = "
        //           << prior_precision_matrices_[k] * prior_means_.get_col(k)
        //           << std::endl;
        // std::cerr << "    Reweighted MLE = "
        //           << noise_parameters().at(k,0) * mle_weights
        //           << std::endl;
        // std::cerr << "    Posterior mean = " << std::endl;
        kjb::Vector posterior_mean(
            posterior_covariance *
              (prior_precision_matrices_[k] * prior_mean_matrix_.get_col(k) +
               noise_parameters().at(k,0) * data_projection.get_col(k)));
        // std::cerr << posterior_mean << std::endl;
        kjb::MV_gaussian_distribution r_wk(
            posterior_mean,
            posterior_covariance
            );
        W_new.set_col(k, kjb::sample(r_wk));
    }
    if(include_bias_)
    {
        W_ = W_new.submatrix(0, 0, D_prime(), K());
        b_ = W_new.get_row(D_prime());
    } else {
        W_ = W_new;
    }
    PM(verbose_ > 0, "done.\n");
    //std::cerr << "done." << std::endl;
}

/*
void Normal_weights_prior::update_W_()
{
    std::cerr << "    Updating W...";
    kjb::Matrix theta_star_transpose =
        kjb::matrix_transpose(theta_star()); // (D+1) x T
    kjb::Matrix likelihood_precision =
        (theta_star_transpose * theta_star()); // (D+1) x (D+1)
    kjb::Matrix data_projection =
        theta_star_transpose * parent->noisy_data(); // (D+1) x K
    kjb::Matrix mle_weights = likelihood_precision * data_projection;
    Weight_matrix W_new((int) D_prime() + include_bias_, (int) K());
    for(size_t k = 0; k < K(); ++k)
    {
        kjb::Matrix posterior_covariance =
            kjb::matrix_inverse(
                prior_precision_matrices_.at(k) +
                noise_parameters().at(k,0) * likelihood_precision
                );
        kjb::Vector posterior_mean(
            posterior_covariance *
            (prior_precision_matrices_[k] * prior_means_.get_col(k) +
             noise_parameters().at(k,0) * mle_weights));
        kjb::MV_gaussian_distribution r_wk(
            posterior_mean,
            posterior_covariance
            );
        W_new.set_col(k, kjb::sample(r_wk));
    }
    if(include_bias_)
    {
        W_ = W_new.submatrix(0, 0, D_prime(), K());
        b_ = W_new.get_row(D_prime());
    } else {
        W_ = W_new;
    }
    std::cerr << "done." << std::endl;
}
 */

Normal_weights_prior::Weight_matrix_vec
Normal_weights_prior::read_and_format_precision_matrices_()
{
    Weight_matrix_vec result;
    for(int k = 0; k < prior_variances_.get_num_cols(); ++k)
    {
        Weight_matrix prior_precision =
            kjb::create_diagonal_matrix((int) D_prime(), 1.0);
        for(int dprime = 0; dprime < D_prime(); ++dprime)
        {
            prior_precision(dprime, dprime) =
                1.0 / prior_variances_(dimension_map(dprime), k);
        }
        result.push_back(prior_precision);
    }
    return result;
}

Normal_weights_prior::Weight_matrix_vec
Normal_weights_prior::create_constant_precision_matrices_(
    const double& sigma_2_w,
    const double& sigma_2_b
    )
{
    Weight_matrix_vec result;
    for(size_t k = 0; k < K(); ++k)
    {
        Weight_vector w_h = Weight_vector((int) D_prime(), 1.0 / sigma_2_w);
        Weight_vector b_h =
            include_bias_ ?
            Weight_vector(1, 1.0 / sigma_2_b)
            : Weight_vector();
        Weight_matrix prior_precision =
            kjb::create_diagonal_matrix(kjb::cat_vectors(w_h, b_h));
        result.push_back(prior_precision);
    }
    return result;
}

// void Normal_weights_prior::update_h_and_mu_()
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

