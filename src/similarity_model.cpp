/* $Id: similarity_model.cpp 21479 2017-07-17 14:08:40Z chuang $ */

/*!
 * @file similarity_model.cpp
 *
 * @author Colin Dawson 
 */

#include "util.h"
#include "state_model.h"
#include "similarity_model.h"
#include "transition_prior.h"
#include "hdp_hmm_lt.h"

void Similarity_model::set_parent(const HDP_HMM_LT* const p)
{
    parent = p;
    write_path = parent->write_path;
}

void Similarity_model::set_up_results_log() const
{
    create_directory_if_nonexistent(write_path + "phi");
}

void Similarity_model::write_state_to_file(const std::string& name) const
{
    std::ofstream ofs;
    ofs.open(write_path + "phi/" + name + ".txt", std::ofstream::out);
    ofs << Phi();
    ofs.close();
}

Coordinate Similarity_model::get_theta(const size_t& j, const size_t& d) const
{
    /*
     get dimension d of the location for latent state j
     */
    return parent->theta(j,d);
}

const State_matrix& Similarity_model::theta() const {return parent->theta();}

State_type Similarity_model::theta_prime(const size_t& j) const {return parent->theta_prime(j);}

const size_t& Similarity_model::J() const {return parent->J_;}
const size_t& Similarity_model::D() const {return parent->D_;}
const Count_matrix& Similarity_model::N() const {return parent->N();}
const Count_matrix& Similarity_model::Q() const {return parent->Q();}
Count Similarity_model::N(const size_t& j, const size_t& jp) const {return parent->N(j,jp);}
Count Similarity_model::Q(const size_t& j, const size_t& jp) const {return parent->Q(j,jp);}

const size_t& Similarity_model::D_prime() const {return parent->D_prime();};

Prob_vector Similarity_model::log_likelihood_for_state_range(
    const size_t& j,
    const size_t& d,
    const size_t& num_states
    ) const
{
    Prob_vector result((int) num_states, 0.0);
    for(size_t s = 0; s < num_states; ++s)
    {
        result[s] = this->log_likelihood_for_state(j,d,s);
    }
    return result;
}

/*
 Below is the HMC interface, refer to section 4.3 "Modifications to Model and Inference" in the paper
 gradient for the posteiror log likelihood in the derived class
 */

Similarity_model::HMC_interface::HMC_interface(const Similarity_model* const sim_m)
    : sim_m(sim_m),
      J(sim_m->J()),
      D(sim_m->D())
{}

void Similarity_model::HMC_interface::sync_Delta()
{
    // std::cerr << "    Synchronizing Delta...";
    Delta = sim_m->get_Delta(theta);
    // std::cerr << Delta << std::endl;
    // std::cerr << "done." << std::endl;
}

void Similarity_model::HMC_interface::sync_Phi()
{
    // std::cerr << "    Synchronizing Phi...";
    Phi = sim_m->get_Phi(sim_m->lambda(), Delta);
    // std::cerr << Phi << std::endl;
    // std::cerr << "done." << std::endl;
}

void Similarity_model::HMC_interface::sync_anti_Phi()
{
    // std::cerr << "    Synchronizing anti-Phi...";
    anti_Phi = sim_m->get_anti_Phi(sim_m->lambda(), Delta);
    // std::cerr << anti_Phi << std::endl;
    // std::cerr << "done." << std::endl;
}

Coordinate Similarity_model::HMC_interface::get(size_t pos) const
{
    size_t j = pos / D;
    size_t d = pos % D;
    assert(j*D + d == pos);
    return theta(j,d);
}

void Similarity_model::HMC_interface::set(size_t pos, Coordinate val)
{
    size_t j = pos / D;
    size_t d = pos % D;
    assert(j*D + d == pos);
    // std::cerr << "old = " << theta(j,d) << "; new = " << val << std::endl;
    theta(j,d) = val;
    if(pos == size() - 1) initialize_params(theta);
    // sync_after_theta_update(j, d, theta(j,d), val);
}

size_t Similarity_model::HMC_interface::size() const
{
    return J * D;
}

void Similarity_model::HMC_interface::initialize_params(const State_matrix& new_theta)
{
    theta = new_theta;
    // std::cerr << "    Before:" << std::endl;
    // std::cerr << "Delta = " << std::endl;
    // std::cerr << Delta << std::endl;
    // std::cerr << "Phi = " << std::endl;
    // std::cerr << Phi << std::endl;
    // std::cerr << "anti-Phi = " << std::endl;
    // std::cerr << anti_Phi << std::endl;
    if(sim_m != NULL)
    {
        sync_Delta();
        sync_Phi();
        sync_anti_Phi();
    }
    // std::cerr << "    After:" << std::endl;
    // std::cerr << "Delta = " << std::endl;
    // std::cerr << Delta << std::endl;
    // std::cerr << "Phi = " << std::endl;
    // std::cerr << Phi << std::endl;
    // std::cerr << "anti-Phi = " << std::endl;
    // std::cerr << anti_Phi << std::endl;
}

Prob Similarity_model::HMC_interface::log_likelihood() const
{
    Prob log_likelihood = 0.0;
    // std::cerr << "Computing log_likelihood for theta given N,Q:" << std::endl;
    // std::cerr << "N = " << std::endl;
    // std::cerr << sim_m->N() << std::endl;
    // std::cerr << "Q = " << std::endl;
    // std::cerr << sim_m->Q() << std::endl;
    // std::cerr << "Delta = " << std::endl;
    // std::cerr << Delta << std::endl;
    // std::cerr << "Phi = " << std::endl;
    // std::cerr << Phi << std::endl;
    // std::cerr << "anti-Phi = " << std::endl;
    // std::cerr << anti_Phi << std::endl;
    for(size_t j = 0; j < J; ++j)
    {
        for(size_t jp = 0; jp < J; ++jp)
        {
            // std::cerr << "Log likelihood increment is " << N(j,jp) << "*" << Phi(j,jp)
            //           << " + " << Q(j,jp) << "*" << anti_Phi(j,jp) << std::endl;
            // std::cerr << "Yields " << N(j,jp) * Phi(j,jp) + Q(j,jp) * anti_Phi(j,jp)
            //           << std::endl;
            log_likelihood += N(j,jp) * Phi(j,jp);
            if(Q(j,jp) > 0) log_likelihood += Q(j,jp) * anti_Phi(j,jp);
        }
    }
    return log_likelihood;
}

/*
Prob Similarity_model::HMC_interface::log_likelihood_gradient(
    const size_t& j,
    const size_t& d
    ) const
{
    Prob result = 0;
    for(size_t jp = 0; jp < J; ++jp)
    {
        result +=
            sim_m->log_likelihood_gradient_term(
                theta(j,d), theta(jp,d), N(j,jp), Q(j,jp),
                Phi(j,jp), anti_Phi(j,jp));
    }
    return result;
}
 */

Prob Similarity_model::HMC_interface::log_likelihood_gradient(
    const size_t& j,
    const size_t& d
    ) const
{
    /*
     access the log likelihood gradient in the Derived Class
     */
    Prob result = 0;
    for(size_t jp = 0; jp < J; ++jp)
    {
        if (j == jp) continue;
        result +=
        sim_m->log_likelihood_gradient_term(
            theta(j,d), theta(jp,d), Delta(j,jp), N(j,jp), Q(j,jp),
            Phi(j,jp), anti_Phi(j,jp));
    }
    return result;
}

void Similarity_model::HMC_interface::sync_Delta_jd(
    const size_t&     j,
    const size_t&     d,
    const Coordinate& theta_old,
    const Coordinate& theta_new
    )
{
    for (size_t jp = 0; jp < J; ++jp)
    {
        if(jp == j) continue;
        double increment =
            sim_m->get_change_to_Delta(jp, d, theta_old, theta_new, theta);
        // std::cerr << "        increment = " << increment << std::endl;
        Delta(jp,j) = Delta(j,jp) += increment;
    }
}

void Similarity_model::HMC_interface::sync_after_theta_update(
    const size_t&     j,
    const size_t&     d,
    const Coordinate& theta_old,
    const Coordinate& theta_new
    )
{
    // std::cerr << "    Updating coordinate (j,d) = (" << j << "," << d << ")"
    //           << std::endl;
    assert(theta(j,d) == theta_old);
    sync_Delta_jd(j, d, theta_old, theta_new);
    for(size_t jp = 0; jp < J; ++jp)
    {
        Phi(j,jp) = Phi(jp,j) = sim_m->get_Phi_j_jp(sim_m->lambda(), Delta(j,jp));
    }
    theta(j,d) = theta_new;
    assert(sim_m->get_theta(j,d) == theta_new);
    // std::cerr << "    Synchronizing Phi = " << std::endl;
    // std::cerr << Phi << std::endl;
}

const Prob& Similarity_model::HMC_interface::get_Phi(const size_t& j, const size_t& jp) const
{
    return Phi(j,jp);
}

Count Similarity_model::HMC_interface::N(const size_t& j, const size_t& jp) const
{
    return sim_m->N(j,jp);
}

Count Similarity_model::HMC_interface::Q(const size_t& j, const size_t& jp) const
{
    return sim_m->Q(j,jp);
}
