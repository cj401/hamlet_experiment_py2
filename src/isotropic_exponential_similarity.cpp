/* $Id: isotropic_exponential_similarity.cpp 21509 2017-07-21 17:39:13Z cdawson $ */

/*!
 * @file isotropic_exponential_similarity.cpp
 *
 * @author Colin Dawson 
 */

#include "util.h"
#include "isotropic_exponential_similarity.h"
#include "state_model.h"
#include "hdp_hmm_lt.h"
#include <third_party/arms.h>
#include <boost/make_shared.hpp>
#include <boost/bind.hpp>
#include <functional>

double hamming_distance(const double& x, const double& y)
{
    return (int) x == (int) y ? 0.0 : 1.0;
}

double l1_distance(const double& x, const double& y) {return std::abs(x - y);}

double squared_distance(const double& x, const double& y) {return (x - y) * (x - y);}

double (*interpret_metric(const std::string& s))(const double&, const double&)
{
    if(s == "l1") return &l1_distance;
    if(s == "squared") return &squared_distance;
    if(s == "hamming") return &hamming_distance;
    else KJB_THROW_3(kjb::Illegal_argument, "%s is not a known metric name",
                     (s.c_str()));
}

Similarity_model_ptr Isotropic_exponential_similarity_parameters::make_module() const
{
    return boost::make_shared<Isotropic_exponential_similarity>(this);
}

Isotropic_exponential_similarity::Isotropic_exponential_similarity(
    const Params* const hyperparams
    ) : fixed_lambda(hyperparams->fixed_lambda),
        b_lambda_(hyperparams->b_lambda),
        metric_(interpret_metric(hyperparams->metric)),
        lambda_(hyperparams->lambda),
        C_(), kernel_(hyperparams->kernel)
        //lambda_posterior_linear_coef_(0),
{
    /*
     pick option of distance metric and kernel
     (distance metric: hamming, l1, and squared)
     (kernel: Laplacian vs. cauchy)
     lambda_: either fixed or sample from ~exponential(b_lambda_)
     */
    if (kernel_ != "isotropic_exponential" && kernel_ != "cauchy")
    {
        KJB_THROW_3(kjb::Illegal_argument, "%s is not a known kernel name",
                    (kernel_.c_str()));
    }
    std::cerr << "        Distance metric is " << hyperparams->metric
              << " and the Kernel is " << kernel_
              << std::endl;
}

void Isotropic_exponential_similarity::initialize_resources()
{
    PM(verbose_ > 0, "Allocating resources for similarity model...\n");
    //std::cerr << "Allocating resources for similarity model..." << std::endl;
    
    // std::cerr << "    Allocating Phi...";
    // Phi_ = Prob_matrix(J(), J(), 0.0);
    // std::cerr << "done." << std::endl;
    // // std::cerr << "    Allocating C...";
    // // C_ = std::vector<Count_matrix>(2, Count_matrix(J(), D(), 0));
    // // std::cerr << "done." << std::endl;
    // std::cerr << "    Allocating Delta...";
    // Delta_ = Distance_matrix(J(), J(), 0.0);
    // std::cerr << "done." << std::endl;
    // anti_Phi_ = Prob_matrix(J(), J(), 0.0);
    params = new_model_object();
}

void Isotropic_exponential_similarity::initialize_params()
{
    /*
     initialize parameter lambda in laplacian similarity function (section 4.1 "The Model" in the paper)
     or lambda in cauchy similarity function (not include in the paper)
     
     sample it from a exponential distribution with rate b_lambda_
     */
    PM(verbose_ > 0, "Initializing isotropic exponential similarity kernel...\n");
    //std::cerr << "Initializing isotropic exponential similarity kernel..." << std::endl;
    // Initialize lambda from the prior
    if(fixed_lambda)
    {
        PMWP(verbose_ > 0, "    Using fixed lambda = %.6f\n", (lambda_));
        //std::cerr << "    Using fixed lambda = " << lambda_ << std::endl;
    } else {
        PM(verbose_ > 0, "    Initializing lambda = ");
        //std::cerr << "    Initializing lambda = ";
        Gamma_dist r_lambda(1.0, 1.0 / b_lambda_);
        lambda_ = kjb::sample(r_lambda);
        PMWP(verbose_ > 0, "%.6f", (lambda_));
        //std::cerr << lambda_ << std::endl;
    }
    // std::cerr << "    Initializing Delta and Phi using theta = " << std::endl;
    // std::cerr << theta() << std::endl;
    params->initialize_params(theta());
    PM(verbose_ > 0, "done.\n");
    //std::cerr << "done." << std::endl;
}

void Isotropic_exponential_similarity::input_previous_results(const std::string& input_path, const std::string& name)
{
    PM(verbose_ > 0, "Inputting information for metric similarity model...\n");
    //std::cerr << "Inputting information for metric similarity model..." << std::endl;
    if (input_path == write_path)
    {
        PM(verbose_ > 0, "Inputting information for lambda_...\n");
        //std::cerr << "Inputting information for lambda_..." << std::endl;
        lambda_ = input_to_value<double>(input_path, "lambda.txt", name);
        PMWP(verbose_ > 0, "lambda = %.6f\n", (lambda_));
        //std::cerr << "lambda_ = " << lambda_ << std::endl;
    }
    params->initialize_params(theta());
    PM(verbose_ > 0, "done.\n");
    //std::cerr << "done." << std::endl;
}

void Isotropic_exponential_similarity::update_params()
{
    PM(verbose_ > 0, "Updating similarity parameters...\n");
    //std::cerr << "Updating similarity parameters..." << std::endl;
    if(!fixed_lambda)
    {
        update_lambda_();
        params->sync_Phi();
    }
    PM(verbose_ > 0, "Done updating simialrity parameters.\n");
    //std::cerr << "Done updating similarity parameters." << std::endl;
}

void Isotropic_exponential_similarity::set_up_results_log() const
{
    Base_class::set_up_results_log();
    std::ofstream ofs;
    ofs.open(write_path + "lambda.txt", std::ofstream::out);
    ofs << "iteration value" << std::endl;
    ofs.close();
}

void Isotropic_exponential_similarity::write_state_to_file(const std::string& name) const
{
    Base_class::write_state_to_file(name);
    const std::string filestem = write_path + name;
    std::ofstream ofs;
    ofs.open(write_path + "lambda.txt", std::ofstream::out | std::ofstream::app);
    ofs << name << " " << lambda_ << std::endl;
    ofs.close();
}

Distance_matrix Isotropic_exponential_similarity::get_Delta(
    const State_matrix& theta
    ) const
{
    /*
     distance matrix delta, delta_{j,j'} = d(l_{j}, l_{j'}) where l_{j} is the location for state j
     refer to section 4.1 "The Model" in the paper
     Phi_{j,j'} inversely depends on distance between the state j and j'
     The larger the distance is, the smaller the similarity measure is.
     */
    size_t J = theta.get_num_rows();
    size_t D = theta.get_num_cols();
    Distance_matrix result(J, J, 0.0);
    for(size_t j = 0; j < J; ++j)
    {
        result(j,j) = 0.0;
        for(size_t jp = 0; jp < j; ++jp)
        {
            double entry = 0.0;
            for(size_t d = 0; d < D; ++d)
            {
                if(metric_(theta(j,d), theta(jp,d)) < 0.0)
                    KJB_THROW_2(kjb::Cant_happen, "Negative distance?");
                entry += metric_(theta(j,d), theta(jp,d));
            }
            result(j,jp) = result(jp,j) = entry;
        }
    }
    return result;
}

Prob Isotropic_exponential_similarity::get_Phi_j_jp(
    const double& lambda,
    const Distance& Delta_j_jp) const
{
    /*
     Delta_{j,j'} is the distance between state j and j'
     Phi_{j,j'} is the similarity measure between state j and j'
     For isotropic_exponential, phi_{j,j'}=exp(- lambda_ * delta_{j,j'})
     Here, we store all data in log for numerical stability
     Refer to section 4.1 "The Model" for more detail
     
     Alternative kernel is Cauchy kernel, which
     phi_{j,j'} = 1 / (1 + lambda_ * delta_{j,j'})
     */
    if (kernel_ == "isotropic_exponential")
        return -lambda * Delta_j_jp;
    else if (kernel_ == "cauchy")
        return -log(1 + lambda * Delta_j_jp);
    else 
        KJB_THROW_2(kjb::IO_error, "Unknown kernel type.  Parameter "
                    " Isotropic_exponential_similarity kernel must be one of"
                    " 'isotropic_exponential' or 'cauchy'");
}

Prob_matrix Isotropic_exponential_similarity::get_Phi(
    const double& lambda,
    const Distance_matrix& Delta
    ) const
{
    /*
     compute the Phi(), J by J matrix, where entry (j,j') is the similarity measure (rate of successful jump) 
     between state j and state j', call function get_phi_j_jp
     refer to section 4.1 "The Model" for more details
     */
    const size_t J = Delta.get_num_rows();
    Prob_matrix result(J, J);
    for(size_t j = 0; j < J; ++j)
    {
        result(j,j) = 0.0;
        for(size_t jp = 0; jp < j; ++jp)
        {
            result(j,jp) = result(jp,j) = get_Phi_j_jp(lambda, Delta(j,jp));
        }
    }
    return result;
}

void Isotropic_exponential_similarity::sync_before_theta_update() const
{
    PM(verbose_ > 0, "        Syncing before theta update...");
    //std::cerr << "        Syncing before theta update...";
    sync_C_();
    PM(verbose_ > 0, "done.\n");
    //std::cerr << "done." << std::endl;
}

void Isotropic_exponential_similarity::sync_after_theta_update(
    const size_t&     j,
    const size_t&     d,
    const Coordinate& theta_old,
    const Coordinate& theta_new
    ) const
{
    params->sync_after_theta_update(j, d, theta_old, theta_new);
    sync_C_jd_(j,d, theta_old, theta_new);
}

Distance Isotropic_exponential_similarity::get_change_to_Delta(
    const size_t&       jp,
    const size_t&       d,
    const Coordinate&   theta_old,
    const Coordinate&   theta_new,
    const State_matrix& theta
    ) const
{
    // std::cerr << "    Computing change to Delta contributed from [" << jp << "," << d << "]"
    //           << std::endl;
    // std::cerr << "        theta_new = " << theta_new << std::endl;
    // std::cerr << "        theta_old = " << theta_old << std::endl;
    // std::cerr << "        theta(jp,d) = " << theta(jp,d) << std::endl;
    return metric_(theta_new, theta(jp,d)) - metric_(theta_old, theta(jp,d));
}

Similarity_model::Model_ptr Isotropic_exponential_similarity::new_model_object() const
{
    Model_ptr result = boost::make_shared<HMC_interface>(this);
    // result->initialize_params(theta);
    return result;
}

void Isotropic_exponential_similarity::sync_C_jd_(
    const size_t&     j,
    const size_t&     d,
    const Coordinate& theta_old,
    const Coordinate& theta_new
    ) const
{
    for(size_t jp = 0; jp < J(); ++jp)
    {
        size_t n = N(j,jp) + N(jp,j);
        C(jp,d, theta_old) -= n;
        C(jp,d, theta_new) += n;
    }
}

/*
void Isotropic_exponential_similarity::sync_C_() const
{
    // std::cerr << "        Syncing successful transition counts...";
    for(std::map<size_t, Count_matrix>::iterator it = C_.begin();
        it != C_.end(); ++it)
    {
        it->second.zero_out();
    }
    for (size_t j = 0; j < J(); ++j)
    {
        for (size_t jp = 0; jp < j; ++jp)
        {
            size_t n = N(j,jp) + N(jp,j);
            for(size_t d = 0; d < D(); ++d)
            {
                C(j,d,get_theta(jp,d)) += n;
                C(jp,d,get_theta(j,d)) += n;
            }
        }
    }
    // std::cerr << "done." << std::endl;
}
 */

void Isotropic_exponential_similarity::sync_C_() const
{
    // std::cerr << "        Syncing successful transition counts...";
    for(std::map<size_t, Count_matrix>::iterator it = C_.begin();
        it != C_.end(); ++it)
    {
        it->second.zero_out();
    }
    for (size_t j = 0; j < J(); ++j)
    {
        for (size_t jp = 0; jp < j; ++jp)
        {
            size_t n = N(j,jp) + N(jp,j);
            for(size_t d = 0; d < D_prime(); ++d)
            {
                C(j,d,get_theta(jp,d)) += n;
                C(jp,d,get_theta(j,d)) += n;
            }
        }
    }
    // std::cerr << "done." << std::endl;
}

Prob Isotropic_exponential_similarity::log_likelihood_for_state(
    const size_t& j,
    const size_t& d,
    const double& theta_jd
    ) const
{
    /*
     update log likelihood for state if element in the state get changed (for example,
     in binary state model, cocktail party data, section 4.1 in the paper)
     
     called in base class (similarity model)
     */
    Prob log_likelihood = 0.0;
    if (kernel_ == "isotropic_exponential")
    {
        log_likelihood = lambda_ * C(j,d,theta_jd);
        for (size_t jp = 0; jp < J(); ++jp)
        {
            // if there are no failed transitions, they don't factor in
            if(Q(j,jp) + Q(jp,j) == 0) continue;
            double new_distance_to_theta_jp =
                Delta(jp,j)
                + metric_(theta_jd, get_theta(jp,d))
                - metric_(get_theta(j,d), get_theta(jp,d));
            log_likelihood += (Q(j,jp) + Q(jp,j)) *
                log(1 - exp(-lambda_ * new_distance_to_theta_jp));
        }
    }
    else if (kernel_ == "cauchy")
    {
        for (size_t jp = 0; jp < J(); ++jp)
        {
            if(Q(j,jp) + Q(jp,j) == 0) continue;
            double new_distance_to_theta_jp =
                Delta(jp,j)
                + metric_(theta_jd, get_theta(jp,d))
                - metric_(get_theta(j,d), get_theta(jp,d));
            log_likelihood += (Q(j,jp) + Q(jp,j)) * log(lambda_ * new_distance_to_theta_jp) -
                (C(j,d,theta_jd) + Q(j,jp) + Q(jp,j)) *
                log(1 + lambda_ * new_distance_to_theta_jp);
        }
    }
    return log_likelihood;
}

/*
Prob Isotropic_exponential_similarity::log_likelihood_gradient_term(
    const Coordinate& theta1,
    const Coordinate& theta2,
    const Count&      successes,
    const Count&      failures,
    const Prob&       log_similarity,
    const Prob&       log_dissimilarity
    ) const
{
    Prob failure_part =
        failures == 0 ? 0 : failures * exp(log_similarity - log_dissimilarity);
    return -lambda_ * metric_(theta1, theta2) * (successes - failure_part);
}
 */

Prob Isotropic_exponential_similarity::log_likelihood_gradient_term(
    const Coordinate& theta1,
    const Coordinate& theta2,
    const Distance&   delta,
    const Count&      successes,
    const Count&      failures,
    const Prob&       log_similarity,
    const Prob&       log_dissimilarity
    ) const
{
    /*
     compute the gradient for log likelihood used in Hamiltonian Monte Carlo (HMC) in 
     updating the location l_{j} (equation (25) in the paper)
     
     refer to Section 4.3 "Modifications to Model and Inference" for more details.
     */
    if (kernel_ == "isotropic_exponential")
    {
        Prob failure_part =
            failures == 0 ? 0 : failures * exp(log_similarity - log_dissimilarity);
        return -lambda_ * metric_(theta1, theta2) * (successes - failure_part);
    }
    else if (kernel_ == "cauchy")
    {
        /*
        std::cerr << "delta: " << delta << " successes: " << successes << " failures: "
                  << failures << " metric: " << metric_(theta1, theta2) << " lambda: "
                  << lambda_ << std::endl;
         */
        Prob numerator_part = 2 * metric_(theta1, theta2) * (successes * lambda_ * delta + failures);
        Prob denominator_part = delta * (1 + lambda_ * delta);
        //std::cerr << "numerator: " << numerator_part << " denominator: " << denominator_part << std::endl;
        return numerator_part / denominator_part;
    } else {
        KJB_THROW_2(kjb::IO_error, "Unknown kernel type.  Parameter "
                    " Isotropic_exponential_similarity kernel must be one of"
                    " 'isotropic_exponential' or 'cauchy'");
    }
}

Prob Isotropic_exponential_similarity::get_anti_Phi_j_jp(
    const double& lambda,
    const Distance& Delta_j_jp
    ) const
{
    /*
     anti_phi_{j,j'} = 1 - phi_{j,j'}, which is the rate of the unsuccessful jumps
     part of the posterior likelihood
     */
    if (kernel_ == "isotropic_exponential")
        return log(1 - exp(-lambda * Delta_j_jp));
    else if (kernel_ == "cauchy")
        return log(lambda * Delta_j_jp) - log(1 + lambda * Delta_j_jp);
    else 
        KJB_THROW_2(kjb::IO_error, "Unknown kernel type.  Parameter "
                    " Isotropic_exponential_similarity kernel must be one of"
                    " 'isotropic_exponential' or 'cauchy'");
}

Prob_matrix Isotropic_exponential_similarity::get_anti_Phi(
    const double&          lambda,
    const Distance_matrix& Delta
    ) const
{
    /*
     compute the anti_Phi(), J by J matrix, where entry (j,j') is rate of unsuccessful jump from
     state j to state j', call function get_anti_Phi_j_jp
     */
    size_t J = Delta.get_num_rows();
    Prob_matrix result(J,J, 0.0);
    for(size_t j = 0; j < J; ++j)
    {
        for(size_t jp = 0; jp < j; ++jp)
        {
            if(std::abs(Delta(j,jp)) > FLT_EPSILON)
                result(j,jp) = result(jp,j) = get_anti_Phi_j_jp(lambda, Delta(j,jp));
        }
    }
    // std::cerr << "anti_Phi = " << std::endl;
    // std::cerr << result << std::endl;
    return result;
}
    
/*
double lambda_log_density(double lambda, Isotropic_exponential_similarity* model)
{
    Prob_matrix anti_Phi_by_Q =
        kjb::ew_multiply(
            model->get_anti_Phi(lambda, model->params->get_Delta()),
            model->Q_part());
    // std::cerr << "        Anti-Phi x Q = " << anti_Phi_by_Q << std::endl;
    return -model->lambda_posterior_linear_coef_ * (lambda) +
            kjb::sum_matrix_rows(anti_Phi_by_Q).sum_vector_elements();
}
 */

double lambda_log_density(double lambda, Isotropic_exponential_similarity* model)
{
    /*
     function for computing the log posterior density for lambda used in 
     adjective rejection sampling
     posterior for laplacian (isotropic_exponential) kernel is presented in equation (24) 
     in the paper
     */
    if (model->kernel_ == "isotropic_exponential")
    {
        kjb::Matrix Delta_by_N = kjb::ew_multiply(model->params->get_Delta(), model->N_part());
        double lambda_posterior_linear_coef_ = model->b_lambda_ +
            kjb::sum_matrix_rows(Delta_by_N).sum_vector_elements();
        Prob_matrix anti_Phi_by_Q =
            kjb::ew_multiply(
                model->get_anti_Phi(lambda, model->params->get_Delta()),
                model->Q_part());
        // std::cerr << "        Anti-Phi x Q = " << anti_Phi_by_Q << std::endl;
        return -lambda_posterior_linear_coef_ * (lambda) +
            kjb::sum_matrix_rows(anti_Phi_by_Q).sum_vector_elements();
    }
    else if (model->kernel_ == "cauchy")
    {
        Prob Prior_part = -model->b_lambda_ * lambda * log(model->b_lambda_);
        Prob_matrix Phi_by_N =
        kjb::ew_multiply(
                model->get_Phi(lambda, model->params->get_Delta()),
                model->N_part());
        Prob_matrix anti_Phi_by_Q =
        kjb::ew_multiply(
                model->get_anti_Phi(lambda, model->params->get_Delta()),
                model->Q_part());
        return Prior_part -
            kjb::sum_matrix_rows(Phi_by_N).sum_vector_elements() +
            kjb::sum_matrix_rows(anti_Phi_by_Q).sum_vector_elements();
    } else {
        KJB_THROW_2(kjb::IO_error, "Unknown kernel type.  Parameter "
                    " Isotropic_exponential_similarity kernel must be one of"
                    " 'isotropic_exponential' or 'cauchy'");
    }
}

void Isotropic_exponential_similarity::update_lambda_()
{
    /*
     sample lambda_, the parameter in the kernel function (laplacian or cauchy)
     use Adaptive Rejection Sampling (arms_simple)
     posterior density for laplacian kernel is presented in
     section 4.1 "Sampling lambda" (equation 24)
     */
    PM(verbose_ > 0, "    Updating lamdba = ");
    //std::cerr << "    Updating lambda = ";
    N_part_ = kjb::Int_matrix(N());
    Q_part_ = Q();
    N_part_.resize(J(), J());
    Q_part_.resize(J(), J());
    double xl = 0.0, xr = 1000.0, old_lambda = lambda_;
    double (*fp)(double, void*) = (double (*)(double, void*)) &lambda_log_density;
    if (kernel_ == "isotropic_exponential")
        arms_simple(4, &xl, &xr, fp, this, 0, &old_lambda, &lambda_);
    else if (kernel_ == "cauchy")
        arms_simple(4, &xl, &xr, fp, this, 1, &old_lambda, &lambda_);
    PMWP(verbose_ > 0, "%.6f\n", (lambda_));
    //std::cerr << lambda_ << std::endl;
}

/*
void Isotropic_exponential_similarity::update_lambda_()
{
    std::cerr << "    Updating lambda = ";
    kjb::Int_matrix N_part(N());
    Q_part_ = Q();
    N_part.resize(J(), J());
    Q_part_.resize(J(), J());
    kjb::Matrix Delta_by_N = kjb::ew_multiply(params->get_Delta(), N_part);
    lambda_posterior_linear_coef_ = b_lambda_ +
        kjb::sum_matrix_rows(Delta_by_N).sum_vector_elements();
    double xl = 0.0, xr = 1000.0, old_lambda = lambda_;
    double (*fp)(double, void*) = (double (*)(double, void*)) &lambda_log_density;
    arms_simple(4, &xl, &xr, fp, this, 0, &old_lambda, &lambda_);
    std::cerr << lambda_ << std::endl;
    // std::cerr << "done." << std::endl;
}
 */

