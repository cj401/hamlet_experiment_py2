/* $Id: categorical_noise_model.cpp 21468 2017-07-11 19:35:05Z cdawson $ */

/*!
 * @file categorical_noise_model.cpp
 *
 * @author Colin Dawson 
 */

#include "util.h"
#include "categorical_noise_model.h"
#include "emission_model.h"
#include <boost/make_shared.hpp>

/*
Categorical_noise_model_parameters::Categorical_noise_model_parameters(
    const Parameters& params,
    const std::string& name
    ) : K(params.get_param_as<size_t>(name, "K"))
{}
 */

Categorical_noise_model_parameters::Categorical_noise_model_parameters(
    const Parameters& params,
    const std::string& name
    ) : Noise_model_parameters(params, ":experiment"),
        K(params.get_param_as<size_t>(name, "K"))
{}

Noise_model_ptr Categorical_noise_model_parameters::make_module() const
{
    return boost::make_shared<Categorical_noise_model>(this);
}

void Categorical_noise_model::initialize_resources()
{
    Base_class::initialize_resources();
}

void Categorical_noise_model::initialize_params()
{
    Base_class::initialize_params();
}

void Categorical_noise_model::input_previous_results(const std::string& input_path, const std::string& name)
{
    PMWP(verbose_ > 0,
         "Inputting information from iteration %s for categorical noise model...\n",
         (name.c_str()));
    //std::cerr << "Inputting information from iteration " << name << " for categorical_noise_model..." << std::endl;
}

void Categorical_noise_model::add_data(const std::string& path, const size_t& num_files)
{
    /// This function gets called before the parent is initialized
    /// so take care not to access the parent pointer here.
    T_list T_t(num_files);
    Y_ = Data_matrix_list(num_files);
    if (num_files > 1)
    {
        boost::format fmt("%03i");
        for (size_t i = 0; i < num_files; i++)
        {
            std::string file_name = (fmt % (i+1)).str() + ".txt";
            std::cerr << "Adding data from file " << path << "obs/" << file_name << std::endl;
            Noisy_data_matrix Ystar((path + "obs/" + file_name).c_str());
            Y_.at(i) = Ystar.floor();
            IFTD(Y_.at(i).get_num_rows() > 0, kjb::IO_error,
                 "I/O ERROR: Could not find data file %s.",
                 ((path + "obs/" + file_name).c_str()));
            IFTD(Y_.at(i).get_num_cols() == 1, kjb::IO_error,
                 "ERROR: Data set has dimensionality %d, but categorical emission"
                 " can only handle output dimension 1", (Y_.at(i).get_num_cols())(K()));
            T_t.at(i) = Y_.at(i).get_num_rows();
        }
    }
    else
    {
        //std::cerr << "Adding data from file " << path << "obs.txt" << std::endl;
        PMWP(verbose_ > 0, "Adding data from file %sobs.txt\n", (path.c_str()));
        Noisy_data_matrix Ystar((path + "obs.txt").c_str());
        //std::cerr << "    Ystar read in successfully." << std::endl;
        PM(verbose_ > 0, "     Ystar read in successfully.\n");
        Y_.at(0) = Ystar.floor();
        //std::cerr << "    Y created by thresholding Ystar." << std::endl;
        PM(verbose_ > 0, "    Y created by the thresholding Ystar.\n");
        IFTD(Y_.at(0).get_num_rows() > 0, kjb::IO_error,
             "I/O ERROR: Could not find data file %s.",
             ((path + "obs.txt").c_str()));
        IFTD(Y_.at(0).get_num_cols() == 1, kjb::IO_error,
             "ERROR: Data set has dimensionality %d, but categorical emission"
             " can only handle output dimension 1", (Y_.at(0).get_num_cols())(K()));
        T_t.at(0) = Y_.at(0).get_num_rows();
    }
    //std::cerr << "    Setting data size...";
    PM(verbose_ > 0, "     Setting data size...");
    set_data_size(T_t, K_);
    PM(verbose_ > 0, "done.\n");
    //std::cerr << "done." << std::endl;
}

void Categorical_noise_model::add_test_data(
    const std::string& path,
    const size_t& num_files
    )
{
    Y_test_ = Data_matrix_list(num_files);
    T_list test_T_t(num_files);
    if (num_files > 1)
    {
        boost::format fmt("%03i");
        for (size_t i = 0; i < num_files; i++)
        {
            std::string file_name = (fmt % (i+1)).str() + ".txt";
            //std::cerr << "Adding data from file " << path << "obs/" << file_name << std::endl;
            PMWP(verbose_ > 0,
                 "Adding data from file %sobs/%s\n", (path.c_str())(file_name.c_str()));
            Noisy_data_matrix Y_test_star((path + "obs/" + file_name).c_str());
            Y_test_.at(i) = Y_test_star.floor();
            IFTD(Y_test_.at(i).get_num_rows() > 0, kjb::IO_error,
                 "I/O ERROR: Could not find data file %s.",
                 ((path + "obs/" + file_name).c_str()));
            IFTD(Y_test_.at(i).get_num_cols() == 1, kjb::IO_error,
                 "ERROR: Test data set has dimensionality %d, but categorical emission"
                 " can only hande output dimension 1", (Y_test_.at(i).get_num_cols())(K()));
            test_T_t.at(i) = Y_test_.at(i).get_num_rows();
        }
    }
    else
    {
        //std::cerr << "Adding test data from file " << path << "obs.txt" << std::endl;
        PMWP(verbose_ > 0, "Adding test data from file %sobs.txt\n", (path.c_str()));
        Noisy_data_matrix Y_test_star((path + "obs.txt").c_str());
        Y_test_.at(0) = Y_test_star.floor();
        IFTD(Y_test_.at(0).get_num_rows() > 0, kjb::IO_error,
             "I/O ERROR: Could not find data file %s.",
             ((path + "obs.txt").c_str()));
        IFTD(Y_test_.at(0).get_num_cols() == 1, kjb::IO_error,
             "ERROR: Data set has dimensionality %d, but categorical emission"
             " can only handle output dimension 1", (Y_test_.at(0).get_num_cols())(K()));
        test_T_t.at(0) = Y_test_.at(0).get_num_rows();
    }
    set_test_data_size(test_T_t);
    PM(verbose_ > 0, "done.\n");
    //std::cerr << "Done adding test data." << std::endl;
    //write test data to result file?
}

//void Categorical_noise_model::add_data(const std::string& path)
//{
//    std::cerr << "Adding data from file " << path << "obs.txt" << std::endl;
//    Noisy_data_matrix Ystar(path + "obs.txt");
//    Y_ = Ystar.floor();
//    IFTD(Y_.get_num_rows() > 0, kjb::IO_error,
//         "I/O ERROR: Could not find data file %s.",
//         ((path + "obs.txt").c_str()));
//    IFTD(Y_.get_num_cols() == 1, kjb::IO_error,
//         "ERROR: Data set has dimensionality %d, but categorical emission"
//         " can only handle output dimension 1", (Y_.get_num_cols())(K()));
//    set_data_size(Y_.get_num_rows(), K_);
//}

//void Categorical_noise_model::add_test_data(const std::string& path)
//{
//    std::cerr << "Adding test data from file " << path << "obs.txt" << std::endl;
//    Noisy_data_matrix Y_test_star(path + "obs.txt");
//    Y_test_ = Y_test_star.floor();
//    IFTD(Y_test_.get_num_rows() > 0, kjb::IO_error,
//         "I/O ERROR: Could not find data file %s.",
//         ((path + "obs.txt").c_str()));
//    IFTD(Y_test_.get_num_cols() == 1, kjb::IO_error,
//         "ERROR: Test data set has dimensionality %d, but categorical emission"
//         " can only hande output dimension 1", (Y_test_.get_num_cols())(K()));
//    set_test_data_size(Y_test_.get_num_rows());
//    std::cerr << "Done adding test data." << std::endl;
//    Y_test_.write((write_path + "test_obs.txt").c_str());
//}

/*
void Categorical_noise_model::generate_data(const std::string& path)
{
    std::cerr << "Generating data..." << std::endl;
    std::cerr << "    Allocating Y...";
    Y_ = Data_matrix((int) T(), (int) 1);
    std::cerr << "done." << std::endl;
    std::cerr << "    Sampling Y...";
    sample_data_(Y_);
    std::cerr << "done." << std::endl;
    std::cerr << "    Writing observations to file " << path + "obs.txt" << std::endl;
    Y_.write((path + "obs.txt").c_str());
}
 */

void Categorical_noise_model::generate_data(const std::string& path)
{
    std::cerr << "Generating data..." << std::endl;
    //std::cerr << "    Allocating Y...";
    Y_ = Data_matrix_list(NF());
    std::cerr << "    Sampling Y...";
    for (size_t i = 0; i < NF(); i++)
    {
        Y_[i] = Data_matrix((int) T(i), (int) 1);
        sample_data_(Y_[i]);
    }
    //std::cerr << "done." << std::endl;
    std::cerr << "done." << std::endl;
    if (NF() == 1)
    {
        std::cerr << "    Writing observations to file " << path + "obs.txt" << std::endl;
        Y_[0].write((path + "obs.txt").c_str());
    }
    else
    {
        create_directory_if_nonexistent((path + "obs").c_str());
        boost::format fmt("%03i");
        for (size_t i = 0; i < NF(); i++)
        {
            std::string file_name = (fmt % (i+1)).str() + ".txt";
            std::cerr << "    Writing observations to file " << path + "obs/" + file_name << std::endl;
            Y_[i].write((path + "obs/" + file_name).c_str());
        }
    }
}

void Categorical_noise_model::generate_test_data(const std::string& path)
{
    std::cerr << "Generating test data..." << std::endl;
    Y_test_ = Data_matrix_list(test_NF());
    for (size_t i = 0; i < test_NF(); i++)
    {
        Y_test_[i] = Data_matrix((int) test_T(i), (int) 1);
        sample_data_(Y_test_[i]);
    }
    if (test_NF() == 1)
    {
        std::cerr << "    Writing observations to file " << path + "obs.txt" << std::endl;
        Y_test_[0].write((path + "obs.txt").c_str());
    }
    else
    {
        create_directory_if_nonexistent((path + "obs").c_str());
        boost::format fmt("%03i");
        for (size_t i = 0; i < test_NF(); i++)
        {
            std::string file_name = (fmt % (i+1)).str() + ".txt";
            std::cerr << "    Writing observations to file " << path + "obs/" + file_name << std::endl;
            Y_test_[i].write((path + "obs/" + file_name).c_str());
        }
    }
}

/*
void Categorical_noise_model::generate_test_data(const std::string& path)
{
    std::cerr << "Generating test data..." << std::endl;
    Y_test_ = Data_matrix((int) test_T_, (int) 1);
    sample_data_(Y_test_);
    std::cerr << "    Writing observations to file " << path + "obs.txt" << std::endl;
    Y_test_.write((path + "obs.txt").c_str());
}
 */

void Categorical_noise_model::set_up_results_log() const
{
    if(NF() > 1)
    {
        boost::format fmt("%03i");
        create_directory_if_nonexistent((write_path + "obs").c_str());
        for (size_t i = 0; i < NF(); i++)
        {
            std::string file_name = (fmt % (i+1)).str() + ".txt";
            Y_.at(i).write((write_path + "obs/" + file_name).c_str());
        }
    } else {
        Y_.at(0).write((write_path + "obs.txt").c_str());
    }
}

void Categorical_noise_model::write_state_to_file(const std::string&) const
{}

void Categorical_noise_model::update_params()
{}

Noise_model::Noise_parameters Categorical_noise_model::parameters() const
{
    KJB_THROW_2(kjb::Not_implemented, "Categorical_noise_model does not have a"
                " parameter vector to return");
}

double Categorical_noise_model::log_likelihood_ratio_for_state_change(
    const size_t &,
    const Mean_vector&,
    const Mean_vector&,
    const kjb::Index_range&
    )
{
    KJB_THROW_2(kjb::Not_implemented, "log_likelihood_ratio_for_state_change not implemented"
                " for a Categorical noise model");
}

double Categorical_noise_model::log_likelihood(
    const size_t& i,
    const size_t& t,
    const Mean_vector& mean
    ) const
{
    return log_likelihood(Y_.at(i), t, mean);
}

double Categorical_noise_model::test_log_likelihood(
    const size_t& i,
    const size_t& t,
    const Mean_vector& mean
    ) const
{
    return log_likelihood(Y_test_.at(i), t, mean);
}

/*
double Categorical_noise_model::log_likelihood(const size_t & t, const Mean_vector & mean) const
{
    return log_likelihood(Y_, t, mean);
}
double Categorical_noise_model::test_log_likelihood(const size_t & t, const Mean_vector & mean) const
{
    return log_likelihood(Y_test_, t, mean);
}
 */

double Categorical_noise_model::log_likelihood(
    const Data_matrix&    Y,
    const size_t&         t,
    const Mean_vector&    mean
    )
 {
     return mean.at((size_t) Y.at(t,0));
 }

void Categorical_noise_model::sample_data_(Data_matrix& Y)
{
    const Mean_matrix_list& means = parent->X_star();
    assert(Y.get_num_rows() == (int) T(0));
    for(size_t t = 0; t < T(0); ++t)
    {
        kjb::Categorical_distribution<> r_y(kjb::ew_exponentiate(means[0].get_row(t)), 0);
        Y(t,0) = kjb::sample(r_y);
    }
}

/*
void Categorical_noise_model::sample_data_(Data_matrix_list& Y)
{
    const Mean_matrix_list& means = parent->X_star();
    for (size_t i = 0; i < NF(); i++)
    {
        assert(Y.at(i).get_num_rows() == (int) T(i));
        for (size_t t = 0; t < T(i); ++t)
        {
            kjb::Categorical_distribution<> r_y(kjb::ew_exponentiate(means[i].get_row(t)), 0);
            Y.at(i)(t,0) = kjb::sample(r_y);
        }
    }
}
 */
