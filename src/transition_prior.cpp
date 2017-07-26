/* $Id: transition_prior.cpp 21443 2017-06-28 18:13:50Z chuang $ */

/*!
 * @file transition_prior.cpp
 *
 * @author Colin Dawson 
 */

#include "util.h"
#include "transition_prior.h"
#include "similarity_model.h"
#include "hdp_hmm_lt.h"
#include <prob_cpp/prob_util.h>
#include <prob_cpp/prob_distribution.h>
#include <prob_cpp/prob_sample.h>

//#include <fstream>

Transition_prior::Transition_prior(const Params* const hyperparams) :
    J_(hyperparams->J), parent(NULL)
{}

void Transition_prior::initialize_resources()
{
    /*
     Initialize memory space for 
     pi0: probability for the initial state (J by 1 vector)
     pi_: pi_{j,j'} transition not considering the phi_ (equation (8) in paper, from normalized gamma
     process representation of the HDP-HMM, section 2.1)
     A_: a_{j,j'} (tilda_pi_{j,j'} in equation (8), the normalized transition that includes the 
     phi_{j,j'}, the LT component)
     N_: n_{j,j'} number of transition between j to j'
     Q_: q_{j,j'} number of unsuccessful attempts to jump from state j to j'
     n_dot: n_{j}, number of times state j is visited
     */
    PM(verbose_ > 0, "Allocating transition prior...\n");
    //std::cerr << "Allocating transition prior..." << std::endl;
    PM(verbose_ > 0, "    Allocating pi0 and A...");
    //std::cerr << "    Allocating pi0 and A...";
    pi0_ = Prob_vector((int) J(), 0.0);
    pi_ = Prob_matrix((int) J(), (int) J(), 0.0);
    PM(verbose_ > 0, "done.\n");
    //std::cerr << "done." << std::endl;
    PM(verbose_ > 0, "    Allocating N and Q...");
    //std::cerr << "    Allocating N and Q...";
    N_ = Q_ = Count_matrix(J() + 1, J(), 0);
    n_dot_ = Count_vector(J(), 0);
    PM(verbose_ > 0, "done.\n");
    //std::cerr << "done." << std::endl;
}

void Transition_prior::initialize_params()
{
    PM(verbose_ > 0, "Initializing transition prior...\n");
    //std::cerr << "Initializing transition prior..." << std::endl;
}

void Transition_prior::update_params()
{
    PM(verbose_ > 0, "Updating transition params...\n");
    //std::cerr << "Updating transition params..." << std::endl;
}

void Transition_prior::input_previous_results(const std::string& input_path, const std::string& name)
{
    PM(verbose_ > 0, "Inputting information for transition prior model...\n");
    //std::cerr << "Inputting information for transition prior model..." << std::endl;
    PM(verbose_ > 0, "Inputting prob vectro pi0...");
    //std::cerr << "Inputting prob vector pi0..." << std::endl;
    pi0_ = Prob_vector((input_path + "pi0/" + name + ".txt").c_str());
    PM(verbose_ > 0, "done.\n");
    //std::cerr << "pi0_ = " << pi0_ << std::endl;
    PM(verbose_ > 0, "Inputting prob matrix pi_...");
    //std::cerr << "Inputting prob matrix pi_..." << std::endl;
    /*
    std::fstream _file((write_path + "pi/" + name + ".txt").c_str());
    if (!_file)
    {
        std::cout << (write_path + "pi/" + name + ".txt").c_str() << " not exist" << std::endl;
    }
    else
    {
        std::cout << (write_path + "pi/" + name + ".txt").c_str() << " exist" << std::endl;
    }
    std::cerr << "'" << (write_path + "pi/" + name + ".txt").c_str() << "'" << std::endl;
     */
    pi_ = Prob_matrix((input_path + "pi/" + name + ".txt").c_str());
    //std::cerr << "pi_ = " << pi_ << std::endl;
    PM(verbose_ > 0, "done.\n");
    //std::cerr << "done." << std::endl;
}

void Transition_prior::update_auxiliary_data()
{
    PM(verbose_ > 0, "Updating transition auxiliary data...\n");
    //std::cerr << "Updating transition auxiliary data..." << std::endl;
}

const State_sequence_list& Transition_prior::z() const
{
    /*
     access the hidden state sequence for train file i in dynamic model
     a vector NF by 1 with each element T(i)+1 by 1
     */
    return parent->z();
}

const State_sequence& Transition_prior::z(const size_t& i) const
{
    /*
     access the hidden state sequence for train file i in dynamic model
     a vector T(i)+1 by 1
     */
    return parent->z(i);
}

const State_indicator& Transition_prior::z(const size_t& i, const size_t& t) const
{
    /*
     access the hidden state at time t for train file i in dynamic model
     (markov transition vs. semi-markov trainsition)
     */
    return parent->z(i,t);
}

/*
const State_sequence& Transition_prior::z() const {return parent->z();}
const State_indicator& Transition_prior::z(const size_t& t) const {return parent->z(t);}
 */

void Transition_prior::set_up_results_log() const
{
    //create_directory_if_nonexistent(write_path + "pi");
    create_directory_if_nonexistent(write_path + "pi0");
    create_directory_if_nonexistent(write_path + "A");
    create_directory_if_nonexistent(write_path + "N");
    create_directory_if_nonexistent(write_path + "Q");
    create_directory_if_nonexistent(write_path + "pi");
    std::ofstream ofs;
    ofs.open(write_path + "n_dot.txt", std::ofstream::out);
    ofs << "iteration value" << std::endl;
    ofs.close();
}

void Transition_prior::write_state_to_file(const std::string& name) const
{
    std::ofstream ofs;
    ofs.open(write_path + "N/" + name + ".txt", std::ofstream::out);
    ofs << N_;
    ofs.close();
    ofs.open(write_path + "Q/" + name + ".txt", std::ofstream::out);
    ofs << Q_;
    ofs.close();
    ofs.open(write_path + "pi/" + name + ".txt", std::ofstream::out);
    ofs << pi_;
    ofs.close();
    ofs.open(write_path + "n_dot.txt", std::ofstream::out | std::ofstream::app);
    ofs << name << " " << n_dot_ << std::endl;
    ofs.close();
    ofs.open(write_path + "pi0/" + name + ".txt", std::ofstream::out);
    ofs << kjb::log_normalize_and_exponentiate(pi0()) << std::endl;
    ofs.close();
    ofs.open(write_path + "A/" + name + ".txt", std::ofstream::out);
    ofs << kjb::ew_exponentiate(A()) << std::endl;
    ofs.close();
}

void Transition_prior::set_parent(HDP_HMM_LT* const p)
{
    /*
     Set HDP_HMM_LT as parent so have access to informtion in other module for updating
     transition variable and corresponding auxilary variables
     */
    parent = p;
    write_path = parent->write_path;
}

const Prob_matrix& Transition_prior::A() const
{
    /*
     If not using LT model, then A_ is equivalent to pi_ where pi_ is generate from the
     normalized Gamma Process representation of the HDP-HMM, more detail in section 2.1
     in the paper
     */
    if(parent != 0)
    {
        return parent->A();
    } else {
        return pi();
    }
}

const Prob_matrix& Transition_prior::Phi() const
{
    /*
     access Phi() from the similarity model, Phi_{j,j'} is the similarity measure between
     latent state j and j', the success probability of going from state j to j'
     Section 2.2 and 2.3 in the paper for more detail
     */
    return parent->Phi();
}

Prob Transition_prior::Phi(const size_t& j, const size_t& jp) const
{
    /*
     If not using LT, Phi_{j,j'}=1 for all j,j', then log(1) = 0 so return 0
     everything is in log until the necessary step to avoid numerical underflow
     */
    if(parent != 0)
    {
        return parent->Phi(j,jp);
    } else {
        return 0.0;
    }
}

void Transition_prior::sync_transition_counts(const State_sequence_list& labels)
{
    /*
     Synchronize the count for 
     N_: n_{j,j'} number of transition between j to j'
     n_dot: n_{j}, number of times state j is visited
     by accessing the hidden state sequence in dynamic model
     This will be used in updating auxiliary variables u, m, pi...
     (section 2 and 3 in the paper)
     */
    PM(verbose_ > 0, "Synchronizing transition counts...");
    //std::cerr << "Synchronizing transition counts...";
    std::fill(n_dot_.begin(), n_dot_.end(), 0);
    N_.zero_out();
    for (size_t i = 0; i < labels.size(); i++)
    {
        for (int t = 1; t < labels[i].size(); ++t)
        {
            N(labels[i][t-1], labels[i][t])++;
            n_dot(labels[i][t])++;
        }
    }
    PM(verbose_ > 0, "done.\n");
    //std::cerr << "done." << std::endl;
    //std::cerr << "N_ = " << N_ << std::endl;
    //std::cerr << "n_dot_ = " << n_dot_ << std::endl;
};

/*
void Transition_prior::sync_transition_counts(const State_sequence& labels)
{
    std::cerr << "    Synchronizing transition counts...";
    std::fill(n_dot_.begin(), n_dot_.end(), 0);
    N_.zero_out();
    for(int t = 1; t < labels.size(); ++t)
    {
        N(labels[t-1], labels[t])++;
        n_dot(labels[t])++;
    }
    std::cerr << "done." << std::endl;
};
 */
