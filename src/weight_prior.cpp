/* $Id: weight_prior.cpp 21443 2017-06-28 18:13:50Z chuang $ */

/*!
 * @file weight_prior.cpp
 *
 * @author Colin Dawson 
 */

#include "util.h"
#include "weight_prior.h"
#include "linear_emission_model.h"

void Weight_prior::initialize_resources()
{
    /*
     Allocating memory for the weight matrix W, 
     refer to section 4.1 "The Model" in the paper
     */
    PM(verbose_ > 0, "    Allocating resources for weight prior...\n");
    //std::cerr << "    Allocating resources for weight prior..." << std::endl;
    PMWP(verbose_ > 0, "        Allocating W(%d x %d)...", (D_prime())(K()));
    //std::cerr << "        Allocating W (" << D_prime()
    //          << " x " << K() << ")...";
    W_ = kjb::Matrix(D_prime(), K(), 0.0);
    PM(verbose_ > 0, "done.\n");
    //std::cerr << "done." << std::endl;
    if(include_bias_)
    {
        PMWP(verbose_ > 0, "      Allocating b(%d)...", (K()));
        //std::cerr << "        Allocating b (" << K() << ")...";
        b_ = kjb::Vector((int) K(), 0.0);
        PM(verbose_ > 0, "done.\n");
        //std::cerr << "done." << std::endl;
    }
}

/*
void Weight_prior::input_previous_results(const std::string& name)
{
    std::cerr << "Inputting information from iteration " << name << " for weight prior..." << std::endl;
    
}
 */

void Weight_prior::set_parent(Linear_emission_model* const p)
{
    parent = p;
    write_path = parent->write_path;
}

size_t Weight_prior::NF() const
{
    return parent->NF();
}

size_t Weight_prior::D() const
{
    return parent->D();
}

size_t Weight_prior::D_prime() const
{
    return parent->D_prime();
}

size_t  Weight_prior::K() const
{
    return parent->K();
}

Mean_matrix Weight_prior::get_mean_matrix(const State_matrix& theta_star)
{
    /*
     Compute mean matrix for the observed data y
     refer to section 4.1 "The Model" for more details
     */
    // std::cerr << "W = " << std::endl;
    // std::cerr << W_ << std::endl;
    Mean_matrix result = theta_star * W_;
    if(include_bias_)
    {
        result.shift_rows_by(b_);
    }
    return result;
}

Mean_vector Weight_prior::get_mean_vector(const State_type& theta_j)
{
    Mean_matrix result = kjb::create_row_matrix(theta_j) * W_;
    if(include_bias_)
    {
        result.shift_rows_by(b_);
    }
    return State_type(result);
}

void Weight_prior::insert_latent_dimension(const size_t& d, const size_t& new_pos)
{
    // std::cerr << "Creating new weight vector in position " << new_pos << std::endl;
    // std::cerr << "Current weight matrix is " << std::endl;
    // std::cerr << W_ << std::endl;
    // std::cerr << "New weight vector is " << std::endl;
    // std::cerr << w_new_ << std::endl;
    Weight_matrix W_before = W_.submatrix(0, 0, new_pos, K());
    // std::cerr << "W_before = " << std::endl;
    // std::cerr << W_before << std::endl;
    Weight_vector w_new = propose_weight_vector(d);
    if((int) new_pos == W_.get_num_rows())
    {
        W_.vertcat(kjb::create_row_matrix(w_new));
    } else
    {
    Weight_matrix W_after = W_.submatrix(new_pos, 0, W_.get_num_rows() - new_pos, K());
    // std::cerr << "W_after = " << std::endl;
    // std::cerr << W_after << std::endl;
    W_.resize(W_.get_num_rows() + 1, K(), 0.0);
    // std::cerr << "Resized weight matrix is " << std::endl;
    // std::cerr << W_ << std::endl;
    W_.insert_zero_row(new_pos);
    // std::cerr << "Zero-inserted weight matrix is " << std::endl;
    // std::cerr << W_ << std::endl;
    W_.set_row(new_pos, w_new);
    // std::cerr << "New weight matrix is " << std::endl;
    // std::cerr << W_ << std::endl;
    // std::cerr << "New W_before is " << std::endl;
    // std::cerr << W_.submatrix(0, 0, new_pos, K()) << std::endl;
    // assert(
    //     max_abs_difference(
    //         W_.submatrix(0, 0, new_pos, K()),
    //         W_before) < FLT_EPSILON);
    // std::cerr << "New W_after is " << std::endl;
    // std::cerr << W_.submatrix(new_pos + 1, 0, W_.get_num_rows() - new_pos - 1, K()) << std::endl;
    // assert(
    //     max_abs_difference(
    //         W_.submatrix(new_pos + 1, 0, W_.get_num_rows() - new_pos - 1, K()),
    //         W_after) < FLT_EPSILON);
    }
    // std::cerr << "Done inserting weight vector." << std::endl;
}

void Weight_prior::remove_latent_dimension(const size_t& old_pos)
{
    // std::cerr << "Removing weight vector in position " << old_pos << std::endl;
    W_.remove_row(old_pos);
};

void Weight_prior::replace_latent_dimension(const size_t& d, const size_t& pos)
{
    // std::cerr << "Replacing weight vector in position " << pos << std::endl;
    W_.set_row(pos, propose_weight_vector(d));
};

void Weight_prior::set_up_results_log() const
{
    create_directory_if_nonexistent(write_path + "W");
}

Weight_prior::Noise_parameters Weight_prior::noise_parameters() const
{
    return parent->noise_parameters();
}
