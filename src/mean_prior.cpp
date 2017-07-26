/* $Id: mean_prior.cpp 21443 2017-06-28 18:13:50Z chuang $ */

/*!
 * @file mean_prior.cpp
 *
 * @author Colin Dawson 
 */

#include "util.h"
#include "mean_prior.h"
#include "mean_emission_model.h"

void Mean_prior::initialize_resources()
{
    PM(verbose_ > 0, "    Allocating X...");
    //std::cerr << "    Allocating X...";
    X_ = Mean_matrix(J(), K(), 0.0);
}

void Mean_prior::set_parent(Mean_emission_model * const p)
{
    parent = p;
    write_path = parent->write_path;
}

size_t Mean_prior::J() const
{
    return parent->J();
}

size_t Mean_prior::K() const
{
    return parent->K();
}

size_t Mean_prior::T(const size_t& i) const
{
    return parent->T(i);
}

size_t Mean_prior::NF() const
{
    return parent->NF();
}

/*
size_t Mean_prior::T() const
{
    return parent->K();
}
 */

const Time_set& Mean_prior::partition_map(const size_t i, const State_indicator& j) const
{
    return parent->partition_map(i,j);
}

/*
Mean_prior::Noisy_data_matrix Mean_prior::noisy_data()
{
    return parent->noisy_data();
}
 */

Mean_prior::Noisy_data_matrix_list Mean_prior::noisy_data()
{
    return parent->noisy_data();
}

Mean_prior::Noisy_data_matrix Mean_prior::noisy_data(const size_t& i)
{
    return parent->noisy_data(i);
}

Mean_prior::Noise_parameters Mean_prior::noise_parameters()
{
    return parent->noise_parameters();
}

