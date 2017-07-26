
#include "parameters.h"

namespace
{
    const size_t SUCCESS = 0;
    const size_t ERROR_IN_COMMAND_LINE = 1;
    const size_t STOP_ON_HELP = 2;
}

std::string get_timestamp_string(size_t verbose = 0);

int ensure_directory_exists(const std::string directory_path, const size_t verbose);

int process_program_options_and_read_parameters
    (int argc, char *argv[],
    Parameters& params,
    bool& generate_sequence,
    //bool& verbose,
    size_t& verbose,
    bool& resume);
