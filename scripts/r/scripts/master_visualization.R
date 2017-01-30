#need to install optparse to use option parser
library("optparse")
source("define_functions.R")

option_list = list(
  make_option(c("-q","--query_file"), type="character", 
              default=NULL, 
              help="name of the query file under queries/", metavar="character"),
  make_option(c("-d", "--data_set"), type="character",
              default=NULL,
              help="name of root directory after results_root", metavar="character"),
  make_option(c("-b", "--burnin_samples"), type="character",
              default="10",
              help="number of logged iteration to discard as burnin", metavar="character"),
  make_option(c("-p", "--path_glob"), type="character",
              default = "*",
              help="a glob expression indicating which datasets within data_set to use", 
              metavar="character"),
  make_option(c("-s","--smoothing_window_size"), type="character",
              default="1",
              help="iteration to average over in a rolling window when plotting",
              metavar="character"),
  make_option(c("-v","--vars"), type="character",
              default = "base",
              help = "several options:\n 
                      1. base include F1_score, precision, recall, accuracy and A\n
                      2. variables you want to include seperate by comma",
              metavar="character"),
  make_option(c("-r","--project_root"), type="character",
              default = "../../../../data",
              help = "project root directory relative to where the this script lives",
              metavar="character"),
  make_option(c("-t","--threshold"), type="character", 
              default="2",
              help = "threshold for constructing the block diagonal matrix, default is 2",
              metavar="character"),
  make_option(c("-c", "--block_matrix_code"), type="character",
              default="../../python/experiments/latent_continue_syn/postprocessing/block_diag_analysis.py",
              help = "block diagonal matrix construction code path relative to this script",
              metavar="character"),
  make_option(c("--binary"), type="character",
              default="0",
              help="whether to plot binary matrix", metavar="character")
);

opt_parser = OptionParser(option_list = option_list)
opt = parse_args(opt_parser)
if (is.null(opt$query_file)||is.null(opt$data_set))
{
  print_help(opt_parser)
  stop("Arugment for query file and data_set must be provided.\n", call.=FALSE)
}
plot.vars <- strsplit(opt$vars, ",")[[1]]
if ("base" %in% plot.vars)
{
  base.vars = c("F1_score", "precision", "recall", "accuracy", "A")
  plot.vars <- plot.vars[plot.vars!="base"]
  plot.vars <- append(base.vars, plot.vars)
}
print(plot.vars)
make_plots(opt$query_file,
          opt$data_set,
          as.numeric(opt$burnin_samples),
          opt$path_glob,
          as.numeric(opt$smoothing_window_size),
          plot.vars,
          opt$project_root,
          as.numeric(opt$threshold),
          opt$block_matrix_code,
          as.numeric(opt$binary))
