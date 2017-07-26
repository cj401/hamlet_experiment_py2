// /* $Id: IID_normal_weights.h 21021 2017-01-06 01:08:30Z chuang $ */

// #ifndef IID_NORMAL_WEIGHTS_H_
// #define IID_NORMAL_WEIGHTS_H_

// /*!
//  * @file IID_normal_weights.h
//  *
//  * @author Colin Dawson
//  */

// #include "parameters.h"
// #include "weight_prior.h"
// #include "linear_emission_model.h"
// #include <string>

// class IID_normal_weights;

// struct IID_normal_weights_parameters : public Weight_prior_parameters
// {
//     typedef Weight_prior_parameters Base_class;
//     const double sigma_2_w;
//     const double sigma_2_b;

//     IID_normal_weights_parameters(
//         const Parameters& params,
//         const std::string& name = "IID_normal_weights"
//         ) : Base_class(params, name),
//             sigma_2_w(params.get_param_as<double>(name, "sigma_2_w")),
//             sigma_2_b(params.get_param_as<double>(name, "sigma_2_b"))
//     {}

//     virtual ~IID_normal_weights_parameters() {}

//     virtual Weight_prior_ptr make_module() const;
// };

// class IID_normal_weights : public Weight_prior
// {
// public:
//     /*------------------------------------------------------------
//      * TYPEDEFS
//      *------------------------------------------------------------*/

//     typedef IID_normal_weights Self;
//     typedef IID_normal_weights_parameters Params;
//     typedef Weight_prior Base_class;
//     typedef Linear_emission_model Parent;
// public:
//     IID_normal_weights(const Params* const hyperparameters)
//         : Base_class(hyperparameters),
//           sigma_2_w_(hyperparameters->sigma_2_w),
//           sigma_2_b_(hyperparameters->sigma_2_b)
//     {}

//     virtual void initialize_params()
//     {
//         // sample_h_and_mu_from_prior_();
//         sample_W_from_prior_();
//     }

//     virtual void generate_data(const std::string&)
//     {
        // sample_h_and_mu_from_prior_();
//         sample_W_from_prior_();
//     }

//     virtual void update_params()
//     {
//         update_W_();
        // update_h_and_mu_();
//     }

//     virtual void write_state_to_file(const std::string& name) const
//     {
//         Weight_matrix W_star = W_;
//         if(include_bias_) W_star.vertcat(kjb::create_row_matrix(b_));
//         W_star.write((write_path + "W/" + name + ".txt").c_str());
//     }
//     Weight_vector propose_weight_vector(const size_t& d) const
//     {
//         return sample_w_row_from_prior(d);
//     }
//     Weight_vector sample_w_row_from_prior(const size_t& d) const;
    
// protected:
//     const State_matrix& theta_star() const;
    // void sample_h_and_mu_from_prior_();
//     void sample_W_from_prior_();
//     void update_W_();
    // void update_h_and_mu_();

    // const double a_h_w_;
    // const double b_h_w_;
    // const double a_h_b_;
    // const double b_h_b_;
    // const double h_mu_w_;
    // const double h_mu_b_;
//     const double sigma_2_w_;
//     const double sigma_2_b_;

    // double h_w_;
    // double h_b_;
    // double mu_w_;
    // double mu_b_;
// };

// #endif

