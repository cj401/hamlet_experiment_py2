/* $Id: normal_weights_prior.h 21103 2017-01-22 16:27:51Z chuang $ */

#ifndef NORMAL_WEIGHTS_PRIOR_H_
#define NORMAL_WEIGHTS_PRIOR_H_

/*!
 * @file Normal_weights_prior.h
 *
 * @author Colin Dawson 
 */

#include "parameters.h"
#include "weight_prior.h"
#include "linear_emission_model.h"
#include <string>

class Normal_weights_prior;

struct Normal_weights_prior_parameters : public Weight_prior_parameters
{
    typedef Weight_prior_parameters Base_class;
    const bool equal_variances;
    const bool zero_mean;
    const double sigma_2_w;
    const double sigma_2_b;
    const std::string prior_mean_file;
    const std::string prior_variance_file;

    Normal_weights_prior_parameters(
        const Parameters& params,
        const std::string& name = "Normal_weights_prior"
        ) : Base_class(params, name),
            equal_variances(params.exists(name, "sigma_2_w")),
            zero_mean(!params.exists(name, "prior_mean_file")),
            sigma_2_w(
                !equal_variances ? 0.0 :
                params.get_param_as<double>(
                    name, "sigma_2_w", bad_weight_prior(), valid_scale_param)),
            sigma_2_b(
                !equal_variances ? 0.0 :
                params.get_param_as<double>(
                    name, "sigma_2_b", bad_weight_prior(), valid_scale_param)),
            prior_mean_file(
                equal_variances ? "" :
                params.get_param_as<std::string>(":experiment", "params_path") +
                params.get_param_as<std::string>(name, "prior_mean_file")),
            prior_variance_file(
                equal_variances ? "" :
                params.get_param_as<std::string>(":experiment", "params_path") +
                params.get_param_as<std::string>(name, "prior_variance_file"))
    {
        IFT((sigma_2_w > 0.0 && sigma_2_b > 0.0) ||
            (!prior_mean_file.empty() && !prior_variance_file.empty()),
            kjb::IO_error, bad_weight_prior());
    }

    static const std::string& bad_weight_prior()
    {
        static const std::string msg =
            "ERROR: For a Normal weights prior, config file must either specify "
            "positive sigma_2_w and positive sigma_2_b (if bias weight included), "
            "or specify prior_mean_file and prior_variance_file.";
        return msg;
    }

    virtual ~Normal_weights_prior_parameters() {}

    virtual Weight_prior_ptr make_module() const;
};

class Normal_weights_prior : public Weight_prior
{
public:
    /*------------------------------------------------------------
     * TYPEDEFS
     *------------------------------------------------------------*/

    typedef Normal_weights_prior Self;
    typedef Normal_weights_prior_parameters Params;
    typedef Weight_prior Base_class;
    typedef Linear_emission_model Parent;
    typedef std::vector<Weight_matrix> Weight_matrix_vec;
public:
    Normal_weights_prior(const Params* const hyperparameters)
        : Base_class(hyperparameters),
          equal_variances(hyperparameters->equal_variances),
          zero_mean(hyperparameters->zero_mean),
          sigma_2_w(hyperparameters->sigma_2_w),
          sigma_2_b(hyperparameters->sigma_2_b),
          prior_mean_file(hyperparameters->prior_mean_file),
          prior_variance_file(hyperparameters->prior_variance_file),
          prior_means_(),
          prior_variances_(),
          prior_precision_matrices_()
    {}

    /*------------------------------------------------------------
     * INITIALIZERS
     *------------------------------------------------------------*/
    virtual void initialize_params();

    virtual void generate_data(const std::string&)
    {
        sample_W_from_prior_();
    }
    
    virtual void input_previous_results(const std::string& input_path, const std::string& name);

    /*------------------------------------------------------------
     * INFERENCE INTERFACE
     *------------------------------------------------------------*/
    virtual void update_params()
    {
        update_W_();
    }

    void insert_latent_dimension(const size_t& d, const size_t& new_pos);
    void remove_latent_dimension(const size_t& old_pos);
    void replace_latent_dimension(const size_t& d, const size_t& pos);

    Weight_vector propose_weight_vector(const size_t& d) const
    {
        return sample_w_row_from_prior(d);
    }

    Weight_vector sample_w_row_from_prior(const size_t& d) const;
    
    /*------------------------------------------------------------
     * DISPLAY FUNCTIONS
     *------------------------------------------------------------*/
    virtual void write_state_to_file(const std::string& name) const
    {
        Weight_matrix W_star = W_;
        if(include_bias_) W_star.vertcat(kjb::create_row_matrix(b_));
        W_star.write((write_path + "W/" + name + ".txt").c_str());
    }

    /*------------------------------------------------------------
     * ACCESSORS TO OTHER MODULES
     *------------------------------------------------------------*/
    virtual size_t dimension_map(const size_t& dprime) const;
    
protected:
    /*
    const State_matrix& theta_star() const;
     */
    const State_matrix& theta_star(const size_t& i) const;
    void sample_W_from_prior_();
    void update_W_();
    void sync_prior_dimensions();

    Weight_matrix_vec create_constant_precision_matrices_(
        const double& sigma_2_w,
        const double& sigma_2_b
        );
    
    Weight_matrix_vec read_and_format_precision_matrices_();

    const bool equal_variances;
    const bool zero_mean;
    const bool sigma_2_w;
    const bool sigma_2_b;
    const std::string prior_mean_file;
    const std::string prior_variance_file;

    Weight_matrix prior_means_;
    Weight_matrix prior_variances_;
    Weight_matrix prior_mean_matrix_;
    Weight_matrix_vec prior_precision_matrices_;
};

#endif

