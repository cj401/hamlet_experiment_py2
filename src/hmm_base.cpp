/* $Id: hmm_base.cpp 21443 2017-06-28 18:13:50Z chuang $ */

/*!
 * @file hmm_base.cpp
 *
 * @author Colin Dawson 
 */

#include "util.h"
#include "hmm_base.h"
#include "emission_model.h"

HMM_base::~HMM_base() {}

void HMM_base::add_data(const std::string & path, const size_t& num_files)
{
    NF_ = num_files;
    emission_model_->add_data(path, NF_);
    T_ = emission_model_->T();
    K_ = emission_model_->K();
    this->initialize_resources_();
    this->initialize_params_();
}

void HMM_base::add_test_data(const std::string& path, const size_t& num_files)
{
    test_NF_ = num_files;
    emission_model_->add_test_data(path + "test_", test_NF_);
    test_T_ = emission_model_->test_T();
    test_data_exists = true;
}

void HMM_base::set_up_verbose_level(const size_t verbose)
{
    verbose_ = verbose;
    emission_model_->set_up_verbose_level(verbose);
}

/*
 void HMM_base::add_data(const std :: string & path)
 {
 emission_model_->add_data(path);
 T_ = emission_model_->T();
 K_ = emission_model_->K();
 this->initialize_resources_();
 this->initialize_params_();
 }
 void HMM_base::add_test_data(const std::string& path)
 {
 emission_model_->add_test_data(path + "test_");
 test_T_ = emission_model_->test_T();
 test_data_exists = true;
 }
*/

// void HMM_base::generate_data(const size_t & T, const size_t & K, const std :: string & name)
// {
//     T_ = T; K_ = K;
//     emission_model_->set_data_size(T, K);
//     this->initialize_resources_();
//     this->initialize_params_();
//     emission_model_->generate_data(name);
//     std::cerr << "    Writing states to file " << name + "states.txt" << std::endl;
//     std::cerr << "    Theta* has dimensions (" << theta_star().get_num_rows()
//               << "," << theta_star().get_num_cols() << ")" << std::endl;
//     create_directory_if_nonexistent(name);
//     sync_theta_star_();
//     theta_star().write((name + "states.txt").c_str());
// }

void HMM_base::generate_data(
    const size_t& num_sequences,
    const size_t& T,
    const size_t& K,
    const std::string& name)
{
    NF_ = num_sequences;
    T_ = T_list((int) num_sequences, (int) T);
    T_.at(0) = T; K_ = K;
    emission_model_->set_data_size(T_, K);
    this->initialize_resources_();
    this->initialize_params_();
    emission_model_->generate_data(name);
    create_directory_if_nonexistent(name);
    std::cerr << "    Writing states to file " << name + "states.txt" << std::endl;
    std::cerr << "    Theta* has dimensions (" << theta_star(0).get_num_rows()
    << "," << theta_star(0).get_num_cols() << ")" << std::endl;
    sync_theta_star_();
    theta_star(0).write((name + "states.txt").c_str());
    /*
    if (NF() == 1)
    {
        std::cerr << "    Writing states to file " << name + "states.txt" << std::endl;
        std::cerr << "    Theta* has dimensions (" << theta_star(0).get_num_rows()
        << "," << theta_star(0).get_num_cols() << ")" << std::endl;
        sync_theta_star_();
        theta_star(0).write((name + "states.txt").c_str());
    }
    else
    {
        create_directory_if_nonexistent((name + "states").c_str());
        boost::format fmt("%03i");
        std::cerr << "    Writing states to file " << name + "states/" << std::endl;
        sync_theta_star_();
        for (size_t i = 0; i < NF(); i++)
        {
            std::cerr << "    Theta* " << i << " has dimensions (" << theta_star(i).get_num_rows()
            << "," << theta_star(i).get_num_cols() << ")" << std::endl;
            std::string file_name = (fmt % (i+1)).str() + ".txt";
            theta_star(i).write((name + "states/" + file_name).c_str());
        }
    }
     */
}

void HMM_base::generate_test_sequence(const size_t&, const size_t&, const std::string&)
{}

void HMM_base::input_previous_results(const std::string& input_path, const std::string& name)
{
    PMWP(verbose_ > 1, "Inputting information from iteration %s...\n", (name.c_str()));
    //std::cerr << "Inputting information from iteration " << name << " ..." << std::endl;
    emission_model_->input_previous_results(input_path, name);
}

void HMM_base::write_state_to_file(const std::string& name) const
{
    if (NF() == 1)
    {
        PMWP(verbose_ > 0, "    Writing states to file %sstates.txt\n", (name.c_str()));
        //std::cerr << "    Writing states to file " << name + "states.txt" << std::endl;
        theta_star(0).write((write_path + name + "states.txt").c_str());
    }
    else
    {
        boost::format fmt("%03i");
        PMWP(verbose_ > 0, "    Writing states to file %sstates\n", (name.c_str()));
        //std::cerr << "    Writing states to file " << name + "states/" << std::endl;
        for (size_t i = 0; i < NF(); i++)
        {
            std::string file_name = (fmt % (i+1)).str() + ".txt";
            theta_star(i).write((write_path + name + "states/" + file_name).c_str());
        }
    }

}

/*
void HMM_base::write_state_to_file(const std::string& name) const
{
    std::cerr << "    Writing states to file " << write_path + name + "states.txt" << std::endl;
    // std::cerr << "        theta* has dimensions (" << theta_star().get_num_rows()
    //           << "," << theta_star().get_num_cols() << ")" << std::endl;
    // std::cerr << "        theta* has dimensions (" << T_
    //           << "," << D_prime() << ")" << std::endl;
    theta_star().write((write_path + name + "states.txt").c_str());
}
 */

void HMM_base::initialize_resources_()
{
    theta_star_ = State_matrix_list(NF());
    for (int i = 0; i < NF(); i++)
    {
        PMWP(verbose_ > 0, "Allocating theta* (%d x %d)\n", (T(i))(this->D_prime()));
        //std::cerr << "Allocating theta* (" << T(i) << " x " << this->D_prime()
        //          << ")" << std::endl;
        theta_star(i) = State_matrix(T(i), this->D_prime(), 0.0);
    }
    emission_model_->set_parent(this);
    emission_model_->initialize_resources();
}

/*
void HMM_base::initialize_resources_()
{
    std::cerr << "Allocating theta* (" << T() << " x " << this->D_prime() << ")"
              << std::endl;
    theta_star_ = State_matrix(T(), this->D_prime(), 0.0);
    // std::cerr << "done." << std::endl;
    emission_model_->set_parent(this);
    emission_model_->initialize_resources();
}
 */

void HMM_base::initialize_params_()
{}

