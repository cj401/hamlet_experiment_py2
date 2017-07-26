
/* $Id: main.cpp 18871 2015-04-17 00:14:22Z cdawson $ */

/*!
 * @file main.cpp
 *
 * @author Colin Dawson and Clayton T. Morrison
 */

#include "prob_cpp/prob_sample.h"

#include "program_options_and_parameters.h"

#include "hdp_hmm_lt.h"
#include "factorial_hmm.h"
#include "binary_state_model.h"
#include "categorical_state_model.h"
#include "continuous_state_model.h"
#include "isotropic_exponential_similarity.h"
#include "probit_noise_model.h"
#include "normal_noise_model.h"
#include "markov_transition_model.h"
#include "semimarkov_transition_model.h"
#include "known_weights.h"
#include "normal_weights_prior.h"
#include "hdp_transition_prior.h"
#include "known_transition_matrix.h"
#include "dirichlet_transition_prior.h"
#include "normal_mean_prior.h"
#include "dirichlet_mean_prior.h"
#include "categorical_noise_model.h"
#include "mean_emission_model.h"

#include <boost/random.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/format.hpp>

#include <iostream>
#include <fstream>

int main(int argc, char *argv[])
{
    try
    {
        Parameters params;
        bool generate = false;
        //bool verbose = false;
        size_t verbose = 0;
        bool resume = false;
        bool resume_to_old_result_dir = false;
    
        if (process_program_options_and_read_parameters
            (argc, argv,
             params,
             generate,
             verbose,
             resume) == STOP_ON_HELP)
        {
            return SUCCESS;
        }
        
        std::string resume_path;

        std::string data_path =
            params.get_param_as<std::string>(":experiment", "data_path");
        std::string params_path =
            params.get_param_as<std::string>(":environment", "parameters_dir")
            + params.get_param_as<std::string>(":experiment", "parameters_subdir");
        size_t num_data_files =
            params.exists(":experiment", "num_data_files") ?
            params.get_param_as<size_t>(":experiment", "num_data_files") :
            1;
        size_t num_test_files =
            params.exists(":experiment", "num_test_files") ?
            params.get_param_as<size_t>(":experiment", "num_test_files") :
            1;
        std::string results_path;
        size_t iterations;
        size_t start_iteration;
        size_t generate_sequence_length;
        size_t generate_test_sequence_length = 0;
        if(generate)
        {
            generate_sequence_length =
                params.get_param_as<size_t>(":generate", "sequence_length");
            if(params.exists(":generate", "test_sequence_length"))
            {
                generate_test_sequence_length =
                    params.get_param_as<size_t>(":generate", "test_sequence_length");
            }
        } else {
            results_path =
                params.get_param_as<std::string>(":experiment", "results_path");
            iterations =
                params.get_param_as<size_t>(":experiment", "iterations");
        }
        
        std::string overall_model_type =
            params.exists(":MODEL", "MODEL_TYPE") ?
            params.get_param_as<std::string>(":MODEL", "MODEL_TYPE") :
            "JOINT";
        std::string transition_prior_module =
            params.get_param_as<std::string>(":MODULE", "TRANSITION_PRIOR");
        std::string dynamics_module =
            params.get_param_as<std::string>(":MODULE", "DYNAMICS");
        std::string state_module =
            params.get_param_as<std::string>(":MODULE", "STATE");
        std::string emission_module =
            params.get_param_as<std::string>(":MODULE", "EMISSION");
            // params.get_param_as<std::string>(":MODULE", "EMISSION");
        std::string similarity_module = "Isotropic_exponential";
            // params.get_param_as<std::string>(":MODULE", "SIMILARITY");

        //// Set random seed

        size_t random_seed = params.get_param_as<size_t>(":experiment", "random_seed_value");

        //// Specify the model
    
        Transition_prior_param_ptr tp;
        Dynamics_param_ptr dp;
        State_param_ptr sp;
        Emission_param_ptr ep;
        Similarity_param_ptr sim_p;

        if(overall_model_type == "FACTORIAL")
        {
            params.add(state_module + "_state_model", "D", 1);
        }
        // transition_prior
        if(transition_prior_module == "HDP")
        {
            tp = boost::make_shared<HDP_transition_prior_parameters>(params);
        } else if(transition_prior_module == "Dirichlet") {
            tp = boost::make_shared<Dirichlet_transition_prior_parameters>(params);
        } else if(transition_prior_module == "Binary_factorial") {
            similarity_module = "Binary_factorial";
            params.add("Binary_state_model", "combinatorial_theta", 1);
            params.add(
                "Known_transition_matrix", "J",
                pow(2, params.get_param_as<size_t>("Binary_state_model", "D")));
            IFT(dynamics_module == "HMM", kjb::Not_implemented,
                "ERROR: Binary_factorial transition prior must be accompanied by "
                "HMM dynamics.  HSMM dynamics are not yet implemented.");
            tp = boost::make_shared<Known_transition_parameters>(params);
        } else if(transition_prior_module == "Known_transition_matrix") {
            if(!generate)
            {
                params.add("Known_transition_matrix", "transition_matrix_file", data_path);
                params.add("Known_transition_matrix", "initial_distribution_file", data_path);
            }
            tp = boost::make_shared<Known_transition_parameters>(params);
        } else {
            const std::string msg =
                "Parameters ERROR: Unrecognized or not implemented "
                "transition prior name " + transition_prior_module + ".\n"
                "Config file must contain a line ':MODULE TRANSITION_PRIOR <value>' "
                " where <value> is one of HDP, Dirichlet, or Known_transition_matrix.";
            KJB_THROW_2(kjb::IO_error, msg);
        }
    
        // dynamics
        if(dynamics_module == "HMM")
        {
            dp = boost::make_shared<Markov_transition_parameters>(params);
        } else if(dynamics_module == "HSMM") {
            dp = boost::make_shared<Semimarkov_transition_parameters>(params);
        } else {
            const std::string msg =
                "Parameters ERROR: Unrecognized or not implemented "
                "dynamics module name " + dynamics_module + ".\n"
                "Config file must contain a line ':MODULE DYNAMICS <value>' "
                " where <value> is one of HMM or HSMM.";
            KJB_THROW_2(kjb::IO_error, msg);
        }
    
        // state
        if(state_module == "Binary")
        {
            sp = boost::make_shared<Binary_state_parameters>(params);
        } else if(state_module == "CRP") {
            sp = boost::make_shared<Categorical_state_parameters>(params);
        } else if(state_module == "Continuous") {
            sp = boost::make_shared<Continuous_state_parameters>(params);
        } else {
            const std::string msg =
                "Parameters ERROR: Unrecognized or not implemented "
                "state module name " + state_module + ".\n"
                "Currently, only {Binary,CRP,Continuous} are implemented.";
            KJB_THROW_2(kjb::IO_error, msg);
        }

        // similarity
        if(similarity_module == "Isotropic_exponential")
        {
            sim_p = boost::make_shared<Isotropic_exponential_similarity_parameters>(params);
        } // else if(similarity_module == "Binary_factorial") {
        //     sim_p = boost::make_shared<Factorial_similarity_parameters>(params);
        // } 
        else {
            const std::string msg =
                "Parameters ERROR: Unrecognized or not implemented "
                "similarity module name " + similarity_module + ".\n"
                "Currently, only :MODULE SIMILARITY Isotropic_exponential is implemented.";
            KJB_THROW_2(kjb::IO_error, msg);
        }

        // emission
        typedef boost::shared_ptr<Noise_model_parameters> Noise_param_ptr;
        std::string noise_model;
        std::string weights_model;
        std::string means_prior;
        Noise_param_ptr np;
        noise_model = params.get_param_as<std::string>(":MODULE", "NOISE");
        if(noise_model == "Probit")
        {
            np = boost::make_shared<Probit_noise_parameters>(params);
        } else if (noise_model == "Normal") {
            np = boost::make_shared<Normal_noise_parameters>(params);
        } else {
            const std::string msg =
                "Parameters ERROR: Unrecognized or not implemented "
                "noise module name " + noise_model + ".\n"
                "When using a linear emission model, config file must contain a line "
                "':MODULE NOISE <value>', "
                " where <value> is one of Normal or Probit.";
            KJB_THROW_2(kjb::IO_error, msg);
        }
        if(emission_module == "Linear")
        {
            typedef boost::shared_ptr<Weight_prior_parameters> Weight_param_ptr;
            Weight_param_ptr wp;
            weights_model = params.get_param_as<std::string>(":MODULE", "WEIGHTS_PRIOR");

            if(weights_model == "Known")
            {
                if(!generate) params.add("Known_weights", "weights_path", data_path);
                wp = boost::make_shared<Known_weights_parameters>(params);
            //} else if(weights_model == "IID_normal") {
              //  wp = boost::make_shared<IID_normal_weights_parameters>(params);
            } else if(weights_model == "Normal") {
                wp = boost::make_shared<Normal_weights_prior_parameters>(params);
            } else {
                const std::string msg =
                    "Parameters ERROR: Unrecognized or not implemented "
                    "weights module name " + emission_module + ".\n"
                    "When using a linear emission model, config file must contain a line "
                    "':MODULE WEIGHTS_PRIOR <value>', "
                    " where <value> is one of Known_weights, Normal, or IID_normal.";
                KJB_THROW_2(kjb::IO_error, msg);
            }
            ep = boost::make_shared<Linear_emission_parameters>(wp, np);
        } else if(emission_module == "Means") {
            typedef boost::shared_ptr<Noise_model_parameters> Noise_param_ptr;
            typedef boost::shared_ptr<Mean_prior_parameters> Mean_prior_param_ptr;
            Mean_prior_param_ptr mp;
            noise_model = params.get_param_as<std::string>(":MODULE", "NOISE");
            means_prior = params.get_param_as<std::string>(":MODULE", "MEANS");
            if(means_prior == "Normal")
            {
                mp = boost::make_shared<Normal_mean_prior_params>(params);
            } else {
            const std::string msg =
                "Parameters ERROR: Unrecognized or not implemented "
                "means prior " + means_prior + ".\n"
                "Currently, only :MODULE EMISSION Normal is implemented.";
            KJB_THROW_2(kjb::IO_error, msg);
            }
            
            ep = boost::make_shared<Mean_emission_parameters>(mp, np);
        } else if(emission_module == "Dirichlet_multinomial") {
            Mean_prior_param_ptr mp;
            mp = boost::make_shared<Dirichlet_mean_prior_parameters>(params);
            np = boost::make_shared<Categorical_noise_model_parameters>(params);
            ep = boost::make_shared<Mean_emission_parameters>(mp, np);
        } else {
            const std::string msg =
                "Parameters ERROR: Unrecognized or not implemented "
                "emission module name " + emission_module + ".\n"
                "Currently, only :MODULE EMISSION {Linear/Means} are implemented.";
            KJB_THROW_2(kjb::IO_error, msg);
        }

        /// Instantiate the model
        std::cerr << "INITIALIZING " << overall_model_type << " MODEL:" << std::endl;
        std::cerr << "    " << transition_prior_module << "-"
                  << dynamics_module << " with " << sp->D << " dimensional "
                  << state_module << " state space and "
                  << emission_module << " emissions."
                  << std::endl;
        if(emission_module == "Linear")
        {

            std::cerr << "    Linear weights are assumed " << weights_model
                      << " and error model is " << noise_model
                      << std::endl;
        }
        std::string write_path = generate ? data_path : results_path;
        boost::shared_ptr<HMM_base> model;
        if(overall_model_type == "JOINT")
        {
            model = boost::make_shared<HDP_HMM_LT>(
                tp, dp, sp, ep, sim_p, write_path, random_seed);
        } else if(overall_model_type == "FACTORIAL")
        {
            size_t D = params.get_param_as<size_t>(":FACTORIAL", "NUM_CHAINS");
            std::cerr << "Instantiating factorial HMM with " << D << " chains...";
            model = boost::make_shared<Factorial_HMM>(
                D, tp, dp, sp, ep, write_path, random_seed);
        }
        model->initialize_parent_links();
        model->set_up_verbose_level(verbose);

        if(generate)
        {
            std::cerr << "Generating (training) data...";
            size_t generate_observables_dimension = params.get_param_as<size_t>(
                ":generate","observables_dimension");
            model->generate_data(
                num_data_files,
                generate_sequence_length,
                generate_observables_dimension,
                data_path
                );
            model->set_up_results_log();
            model->write_state_to_file("train");
            if(generate_test_sequence_length > 0)
            {
                std::cerr << "Generating test data..." << std::endl;
                model->generate_test_sequence(
                    num_test_files,
                    generate_test_sequence_length,
                    data_path);
            }
            model->write_state_to_file("test");
        }
        else
        {
            std::cerr << "Adding " << num_data_files << " sequence(s) from directory "
                      << data_path << "..." << std::endl;
            model->add_data(data_path, num_data_files);
            std::cerr << "Data successfully added." << std::endl;
            std::cerr << std::endl;
            
            if (resume)
            {
                resume_path = params.get_param_as<std::string>(":experiment", "resume_path");
                if (resume_path == results_path)
                    resume_to_old_result_dir = true;
                else
                    model->set_up_results_log();
            }
            else
            {
                model->set_up_results_log();
            }
            
            // Set up ground truth evaluation
            std::string gt_eval_filename;
            std::ofstream gt_eval_stream;
            bool do_gt_eval = params.get_param_as<bool>(":experiment", "do_ground_truth_eval");
            bool do_test_set_eval = params.get_param_as<bool>(":experiment", "do_test_set_eval");
            if(do_gt_eval)
            {
                model->add_ground_truth_eval_header();
            }
            if(do_test_set_eval)
            {
                model->add_test_data(data_path, num_test_files);
                std::cerr << "Done adding test data." << std::endl;
            }
            
            boost::format fmt("%05i");
            if (resume)
            {
                size_t resume_iteration = params.get_param_as<size_t>(":experiment","resume_iteration");
                std::cerr << "Resume results from iteration " << (fmt % resume_iteration).str() << " in " << resume_path << std::endl;
                model->input_previous_results(resume_path, (fmt % resume_iteration).str());
                if (resume_to_old_result_dir)
                    start_iteration = resume_iteration + 1;
                else
                {
                    model->write_state_to_file((fmt % 0).str());
                    start_iteration = 1;
                }
            }
            else
            {
                model->write_state_to_file((fmt % 0).str());
                start_iteration = 1;
            }

            for(size_t i = start_iteration; i <= iterations; ++i)
            {
                std::cerr << "ITERATION " << i << ":" << std::endl;
                model->resample();
                if(i % 50 == 0)
                {
                    model->write_state_to_file((fmt % i).str());
                    if(do_gt_eval)
                    {
                        gt_eval_stream << i << " ";
                        model->compare_state_sequence_to_ground_truth(data_path, (fmt % i).str());
                    }
                }
                std::cerr << std::endl;
            }
            gt_eval_stream.close();
            std::cerr << "Data read from " << data_path << "obs.txt" << std::endl;
            std::cerr << "Random seed initialized to " << random_seed << std::endl;
            std::cerr << "Results for " << iterations << " iterations written to "
                      << results_path << std::endl;
        }
    } catch(kjb::Exception e) {
        std::cerr << e.what() << std::endl;
        return 1;
    } catch(std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
        
    return SUCCESS;
}

