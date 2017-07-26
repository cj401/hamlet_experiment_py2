/* $Id: util.cpp 21042 2017-01-11 16:00:00Z chuang $ */

/*!
 * @file util.cpp
 *
 * @author Colin Dawson 
 */

#include "util.h"
#include <fstream>
//need to include boost algorithm string
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

/*
void score_binary_states(
    const std::string& results_path,
    const std::string& ground_truth_path,
    const std::string& label,
    const State_matrix& state_matrix
    )
{
    kjb::Matrix gt((ground_truth_path + "states.txt").c_str());
    int T = state_matrix.get_num_rows();
    int D = state_matrix.get_num_cols();
    assert(T == gt.get_num_rows());
    assert(D == gt.get_num_cols());
    if(T == 0) return;
    int TD = T * D;
    double available_to_recall = kjb::sum_squared_elements(gt);
    double total_recalled = kjb::sum_squared_elements(state_matrix);
    double correctly_recalled =
        kjb::sum_squared_elements(kjb::ew_multiply(gt, state_matrix));
    double recall = correctly_recalled / available_to_recall;
    double precision = correctly_recalled / total_recalled;
    double accuracy = 1 - kjb::sum_squared_elements(state_matrix - gt) / TD;
    double F1 = 2 * precision * recall / (precision + recall);
    std::ofstream ofs;
    ofs.open(results_path + "precision.txt", std::ofstream::app);
    ofs << label << " " << precision << std::endl;
    ofs.close();
    ofs.open(results_path + "recall.txt", std::ofstream::app);
    ofs << label << " " << recall << std::endl;
    ofs.close();
    ofs.open(results_path + "accuracy.txt", std::ofstream::app);
    ofs << label << " " << accuracy << std::endl;
    ofs.close();
    ofs.open(results_path + "F1_score.txt", std::ofstream::app);
    ofs << label << " " << F1 << std::endl;
    ofs.close();
}
 */

void score_binary_states(
    const std::string& results_path,
    const std::string& ground_truth_path,
    const std::string& label,
    const State_matrix_list& state_matrix_list
    )
{
    size_t NF = state_matrix_list.size();
    double available_to_recall = 0.0;
    double total_recalled = 0.0;
    double correctly_recalled = 0.0;
    int TD = 0;
    double Error = 0.0;
    boost::format fmt("%03i");
    for (size_t i = 0; i < NF; i++)
    {
        kjb::Matrix gt = kjb::Matrix();
        State_matrix state_matrix = state_matrix_list[i];
        if (NF == 1)
            gt = kjb::Matrix((ground_truth_path + "states.txt").c_str());
        else
            gt = kjb::Matrix((ground_truth_path + "states/" + (fmt % (i+1)).str()).c_str());
        int T = state_matrix.get_num_rows();
        int D = state_matrix.get_num_cols();
        // std::cerr << "T = " << T << ", D = " << D << std::endl;
        // std::cerr << "gt is " << gt.get_num_rows() << " x "
        //           << gt.get_num_cols() << std::endl;
        assert(T == gt.get_num_rows());
        assert(D == gt.get_num_cols());
        if (T == 0 && NF == 1) return;
        TD += T * D;
        available_to_recall += kjb::sum_squared_elements(gt);
        total_recalled += kjb::sum_squared_elements(state_matrix);
        correctly_recalled += kjb::sum_squared_elements(kjb::ew_multiply(gt, state_matrix));
        Error += kjb::sum_squared_elements(state_matrix - gt);
    }
    double recall = correctly_recalled / available_to_recall;
    double precision = correctly_recalled / total_recalled;
    double accuracy = 1 - Error / TD;
    double F1 = 2 * precision * recall / (precision + recall);
    std::ofstream ofs;
    ofs.open(results_path + "precision.txt", std::ofstream::app);
    ofs << label << " " << precision << std::endl;
    ofs.close();
    ofs.open(results_path + "recall.txt", std::ofstream::app);
    ofs << label << " " << recall << std::endl;
    ofs.close();
    ofs.open(results_path + "accuracy.txt", std::ofstream::app);
    ofs << label << " " << accuracy << std::endl;
    ofs.close();
    ofs.open(results_path + "F1_score.txt", std::ofstream::app);
    ofs << label << " " << F1 << std::endl;
    ofs.close();
}

void add_binary_gt_eval_header(const std::string& results_path)
{
    std::ofstream ofs;
    ofs.open(results_path + "precision.txt", std::ofstream::out);
    ofs << "iteration value" << std::endl;
    ofs.close();
    ofs.open(results_path + "recall.txt", std::ofstream::out);
    ofs << "iteration value" << std::endl;
    ofs.close();
    ofs.open(results_path + "accuracy.txt", std::ofstream::out);
    ofs << "iteration value" << std::endl;
    ofs.close();
    ofs.open(results_path + "F1_score.txt", std::ofstream::out);
    ofs << "iteration value" << std::endl;
    ofs.close();
}

void sample_mv_normal_vectors_from_row_means(
    kjb::Matrix&        result,
    const kjb::Matrix&  means,
    const kjb::Matrix&  cov
    )
{
    int T = means.get_num_rows();
    int K = means.get_num_cols();
    assert(result.get_num_rows() == T);
    assert(result.get_num_cols() == K);
    assert(cov.get_num_rows() == K);
    assert(cov.get_num_cols() == K);
    for(int t = 0; t < T; ++t)
    {
        const kjb::Vector mean = means.get_row(t);
        kjb::MV_gaussian_distribution r_y(mean, cov);
        result.set_row(t, kjb::sample(r_y));
    }
}

void threshold_a_double(
    double&            x,
    const std::string& write_path,
    const std::string& message
    )
{
    if(log(x) < -30)
    {
        x = exp(-30);
        std::ofstream ofs;
        ofs.open(write_path + "UNDERFLOW", std::ofstream::out | std::ofstream::app);
        ofs << "Underflow sampling " << message << std::endl;
        ofs.close();
    }
}

Binary_matrix generate_matrix_of_binary_range(const size_t& num_cols)
{
    size_t num_rows = pow(2,num_cols);
    Binary_matrix result(num_rows, num_cols);
    Binary_vector curr_row(num_cols);
    for(size_t i = 0; i < num_rows; ++i)
    {
        int remainder = i;
        int modulus = pow(2, num_cols - 1);
        for(size_t j = 0; j < num_cols; ++j)
        {
            curr_row[j] = remainder / modulus;
            remainder = remainder % modulus;
            modulus /= 2;
        }
        result.set_row(i, curr_row);
    }
    return result;
}

void initialize_gnu_rng(const unsigned long int& seed)
{
    gsl_rng_env_setup();
    const gsl_rng_type* type = gsl_rng_default;
    gnu_rng = gsl_rng_alloc(type);
    gsl_rng_set(gnu_rng, seed);
}

double sample_left_truncated_normal(double mean)
{
    return(rtnorm(gnu_rng, 0, std::numeric_limits<double>::infinity(), mean, 1).first);
}

double sample_right_truncated_normal(double mean)
{
    return(rtnorm(gnu_rng, -std::numeric_limits<double>::infinity(), 0, mean, 1).first);
}

int create_directory_if_nonexistent(const std::string& dir_name)
{
    if(kjb_c::is_directory(dir_name.c_str()) != true)
    {
        KJB(ETX(kjb_c::kjb_mkdir(dir_name.c_str())));
    }
    if(kjb_c::is_directory(dir_name.c_str()) != true)
    {
        KJB_THROW_3(kjb::IO_error, "Directory %s could not be created", (dir_name.c_str()));
    }
    return EXIT_SUCCESS;
}
