/* $Id: dynamics_model.cpp 21443 2017-06-28 18:13:50Z chuang $ */

/*!
 * @file dynamics_model.cpp
 *
 * @author Colin Dawson 
 */

#include "dynamics_model.h"
#include "transition_prior.h"
#include "hdp_hmm_lt.h"

#include "util.h"
#include <boost/format.hpp>

void Dynamics_model::initialize_resources()
{
    /*
     NF(): call number of train files (NF) from main
     T(i): call time dimension for train file i
     z_: vector with NF elements, each element is a integer vector with T(i)+1 elements (+1 for initial state)
     Here, allocating memory for z_ by having all states to be 0
     */
    PM(verbose_ > 0, "Allocating resources for dynamics model...\n");
    //std::cerr << "Allocating resources for dynamics model..." << std::endl;
    PMWP(verbose_ > 0, "    NF = %d\n", (NF()));
    //std::cerr << "   NF = " << NF() << std::endl;
    z_ = State_sequence_list(NF());
    for (int i = 0; i < NF(); ++i)
    {
        PMWP(verbose_ > 0, "    Allocating z[%d] with T = %d\n", (i)(T(i)));
        //std::cerr << "    Allocating z [" << i << "] with T = " << T(i) << std::endl;
        z_[i] = State_sequence(T(i) + 1, 0);
    }
}

/*
void Dynamics_model::initialize_resources()
{
    std::cerr << "Allocating resources for dynamics model..." << std::endl;
    std::cerr << "    Allocating z for T = " << T() << std::endl;;
    z_ = State_sequence(T() + 1, 0);
};
 */

void Dynamics_model::initialize_params()
{
    PM(verbose_ > 0, "Initializing dynamics and state sequence...\n");
    //std::cerr << "Initializing dynamics and state sequence..." << std::endl;
};

void Dynamics_model::update_params(const size_t& i , const Likelihood_matrix&)
{
    PMWP(verbose_ > 0, "Updating state asequence and dynamics for dataset %d\n", (i));
    //std::cerr << "Updating state sequence and dynamics for dataset "
    //          << i << std::endl;
};

/*
void Dynamics_model::update_params(const Likelihood_matrix&)
{
    std::cerr << "Updating state sequence and dynamics..." << std::endl;
};
 */

void Dynamics_model::set_parent(HDP_HMM_LT* const p)
{
    /*
     function is called at hdp_hmm_lt to set hdp_hmm_lt to be the parent,
     so dynamic model can access functions in hdp_hmm_lt
     */
    parent = p;
    write_path = parent->write_path;
}

void Dynamics_model::set_up_results_log() const
{
    /*
     set up result directory and file
     if multiple train files, create z/ directory, and store each train state (for all iteration) 
     as individual files
     */
    if (NF() == 1)
    {
        std::ofstream ofs;
        ofs.open(write_path + "z.txt", std::ofstream::out);
        ofs << "iteration value" << std::endl;
        ofs.close();
    }
    else
    {
        create_directory_if_nonexistent(write_path + "z");
        boost::format fmt("%03i");
        for (int i = 0; i < NF(); i++)
        {
            std::ofstream ofs;
            ofs.open(write_path + "z/" + (fmt % (i+1)).str() + ".txt", std::ofstream::out);
            ofs << "iteration value" << std::endl;
            ofs.close();
        }
    }
}

/*
void Dynamics_model::set_up_results_log() const
{
    std::ofstream ofs;
    ofs.open(write_path + "z.txt", std::ofstream::out);
    ofs << "iteration value" << std::endl;
    ofs.close();
}
 */

void Dynamics_model::input_previous_results(const std::string& input_path, const std::string& name)
{
    /*
     Input state information from previous iteration, so we can continue the sampling from unfinished 
     experiement, or change parameter in between experiment to get more efficient convergent
     */
    PMWP(verbose_ > 0, "Inputting information from iteration %s for dynmamic model...\n", (name.c_str()));
    //std::cerr << "Inputting information from iteration " << name << " for dynamic model..." << std::endl;
    PM(verbose_ > 0, "Inputting information for state list z_...\n");
    //std::cerr << "  Inputting information for state list z_..." << std::endl;
    if (NF() == 1)
    {
        PM(verbose_ > 0, "    Inputting information for z0...\n");
        //std::cerr << "  Inputting information for z0..." << std::endl;
        z_[0] = State_sequence(input_to_vector<int>(input_path, "z.txt", name));
        //std::cerr << "z0_ = " << z_[0] << std::endl;
    }
    else
    {
        boost::format fmt("%03i");
        for (size_t i = 0; i < NF(); i++)
        {
            PMWP(verbose_ > 0, "    Inputting information for z%d\n", (i));
            //std::cerr << "  Inputting information for z" << i << "..." << std::endl;
            z_[i] = State_sequence(input_to_vector<int>(input_path + "z/", (fmt % i).str() + ".txt", name));
            //std::cerr << "z" << i << " = " << z_[i] << std::endl;
        }
    }
    PM(verbose_ > 0, "done.\n");
    //std::cerr << "done." << std::endl;
}

void Dynamics_model::write_state_to_file(const std::string& name) const
{
    const std::string filestem = write_path + name;
    if (NF() == 1)
    {
        std::ofstream ofs;
        ofs.open(write_path + "z.txt", std::ofstream::out | std::ofstream::app);
        ofs << name << " " << z(0) << std::endl;
        ofs.close();
    }
    else
    {
        boost::format fmt("%03i");
        for (int i = 0; i < NF(); i++)
        {
            std::ofstream ofs;
            ofs.open(write_path + "z/" + (fmt % (i+1)).str() + ".txt", std::ofstream::out | std::ofstream::app);
            ofs << name << " " << z(i) << std::endl;
            ofs.close();
        }
    }
}

/*
void Dynamics_model::write_state_to_file(const std::string& name) const
{
    const std::string filestem = write_path + name;
    std::ofstream ofs;
    ofs.open(write_path + "z.txt", std::ofstream::out | std::ofstream::app);
    ofs << name << " " << z_ << std::endl;
    ofs.close();
}
 */

const size_t& Dynamics_model::NF() const
{
    /*
     accessing NF() function in parent (see function "set_parent" above) to get NF_ (number of train files)
     */
    return parent->NF();
}

size_t Dynamics_model::T(const size_t& i) const
{
    /*
     accessing T(i) function in parent to get T_(i), time dimension for train file i
     */
    return parent->T(i);
}

T_list Dynamics_model::T() const
{
    /*
     accessing T() function in parent to get T_, vector of NF elements with each element (int) being
     the time dimensions for train file i
     */
    return parent->T();
}

//size_t Dynamics_model::T() const {return parent->T();}

size_t Dynamics_model::J() const
{
    /*
     accessing J() function in parent to get J_, the number of possible states
     so trainsition matrix is J_ by J_
     */
    return parent->J();
}

//size_t Dynamics_model::test_T() const {return parent->test_T();}

size_t Dynamics_model::test_T(const size_t& i) const
{
    /*
     accessing test_T(i) function in parent to get test_T_(i), time dimesion for test file i
     */
    return parent->test_T(i);
}

T_list Dynamics_model::test_T() const
{
    /*
     accessing test_T() function in parent to get test_T_, the vector with test_NF_ elements 
     with each element (int) being the time dimension for test file i
     */
    return parent->test_T();
}

const Prob_vector& Dynamics_model::pi0() const
{
    /*
     accessing pi0() function in parent to get pi0_, the vector of probability for the initial state (S0),
     from transition_prior (transition model)
     */
    return parent->pi0();
}

const Prob& Dynamics_model::pi0(const size_t& j) const
{
    /* 
     accessing pi0(j) function in parent to get pi0_(j), the probability of having state j as the initial 
     state (s0), from the trainsition_prior (transition model)
     */
    return parent->pi0(j);
}

const Prob_matrix& Dynamics_model::A() const
{
    /*
     accessing A() function in parent to get A_, the transition matrix, J_ by J_, from transition_prior
     */
    return parent->A();
}

const Prob& Dynamics_model::A(const size_t& j, const size_t& jp) const
{
    /*
     accessing A() function in parent to get A_ from transition_prior,
     here, return the (j,jp) entry of A_, the probability of going from state j to jp
     */
    return A()(j,jp);
}

const Count& Dynamics_model::n_dot(const size_t& j) const
{
    /*
     accessing n_dot(j) function in parent to get n_dot_(j) from transiition_prior,
     number of occurence of state j in the sample state in the current iteration
     */
    return parent->n_dot(j);
}

Prob& Dynamics_model::A(const size_t& j, const size_t& jp)
{
    /*
     accessing A(j,jp) function in parent to get probability of transitioning from state j to jp
     */
    return parent->A(j,jp);
}

Likelihood_matrix Dynamics_model::log_likelihood_matrix(
    const size_t&           i,
    const kjb::Index_range& states,
    const kjb::Index_range& times
    ) const
{
    /*
     accessing log_likelihood_matrix function in parent to get log liklihood matrix for train file
     i, states range and time range are for beam sampling. 
     */
    return parent->log_likelihood_matrix(i, states, times);
}

/*
Likelihood_matrix Dynamics_model::log_likelihood_matrix(
    const kjb::Index_range& states,
    const kjb::Index_range& times
    ) const
{
    return parent->log_likelihood_matrix(states, times);
}
 */

