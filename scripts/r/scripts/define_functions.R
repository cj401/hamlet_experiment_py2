require(abind)

get_directories <- function(project_root)
{
    results_root = paste(project_root, "results/", sep = "")
    data_root = paste(project_root, "data/", sep = "")
    visualization_root = paste(project_root, "fig/", sep = "")
    return(list(results_root = results_root, data = data_root, fig_root= visualization_root))
}


## import the data frame containing data locations
get_specs <- function(query_file, results_dir, data_set, comparison_name)
{
    specs <-
        read.table(
            paste("../queries/", query_file, sep = ""),
            header = TRUE
        )
    specs <- subset(specs, comparison == comparison_name)
    return(
        list(models = as.character(specs$model),
             results = results_dir,
             dataset = data_set,
             comparison = comparison_name))
}

## create a list of data frames collected from the directories
## listed in specs$specs_var_name with filenames `output_type`

get_scalar_or_vector_data <- function(specs, output_type, paths)
{
    models <- specs$models
    data_list <- rep(list(list()), length(models))
    names(data_list) <- unique(models)
    for(m in models)
    {
        cur_path <- getwd()
        model_dir <-
            paste(
                paths$results, "/",
                specs$results, "/",
                specs$dataset, "/",
                m, "/", sep = "")
        ## print(model_dir)
        setwd(model_dir)
        items <- Sys.glob("*")
        setwd(cur_path)
        for(i in items)
        {
            next_data <-
                read.table(
                    paste(
                        model_dir, "/",
                        i, "/",
                        output_type, ".txt",
                        sep = ""),
                    header = FALSE,
                    skip = 1
                )
            data_list[[m]] <-
                append(data_list[[m]], list(next_data))
        }
    }
    return(data_list)
}

get_matrix_data <- function(specs, output_type, paths)
{
    groups <- specs$models
    data_list <- rep(list(list()), length(groups))
    names(data_list) <- unique(groups)
    for(i in 1:length(specs$items))
    {
        data_path <-
            paste(
                paths$data, "/", specs$results, "/",
                specs$items[i], "/",
                output_type, "/", sep = ""
                )
        file_names <- dir(data_path)
        first_matrix <-
            as.matrix(
                read.table(
                    paste(data_path, file_names[1], sep = ""),
                    header = FALSE))
        r <- nrow(first_matrix)
        c <- ncol(first_matrix)
        matrix_stack <- array(0, dim = c(r,c,length(file_names)))
        for(t in 1:length(file_names))
        {
            next_matrix <-
                as.matrix(
                    read.table(
                        paste(data_path, file_names[t], sep = ""),
                        header = FALSE))
            matrix_stack[,,t] <- next_matrix
        }
        data_list[[groups[i]]] <-
            append(
                data_list[[groups[i]]],
                list(matrix_stack)
                )
    }
    return(data_list)
}

collect_data_as_scalar <- function(data_list, summary_function = I)
{
    result = list()
    for(l in data_list)
    {
        vals <- numeric(0)
        iterations <- l[[1]][,1]
        for(df in l)
        {
            newdata <- summary_function(df[,-1])
            length(newdata) <- length(iterations)
            vals <- cbind(vals, newdata)
        }
        l[[1]] <- l[[1]][order(l[[1]][,1]),]
        result <- append(result, list(vals))
    }
    names(result) <- names(data_list)
    return(list(iterations = iterations, values = result))
}

summarize_scalar_data_across_runs <- function(collapsed_data, smoothing_window_size)
{
    print(paste("Summarizing data with smoothing window", smoothing_window_size))
    result <- list()
    center_iteration <-
        floor(collapsed_data$iterations / smoothing_window_size) * smoothing_window_size +
            0.5 * smoothing_window_size
    for(d in collapsed_data$values)
    {
        dd <- apply(d, 2, function(x){tapply(x, center_iteration, mean)})
        result <-
            append(
                result,
                list(data.frame(
                    mean = apply(dd, 1, mean, na.rm = TRUE),
                    se_upper = apply(dd, 1, function(x) {mean(x, na.rm = TRUE) + sd(x, na.rm = TRUE) / sqrt(length(x))}),
                    se_lower = apply(dd, 1, function(x) {mean(x, na.rm = TRUE) - sd(x, na.rm = TRUE) / sqrt(length(x))}),
                    cint_upper =
                        apply(dd, 1, function(x)
                            {mean(x, na.rm = TRUE) + sqrt(var(x, na.rm = TRUE) / length(x)) * 2 * qt(0.995, length(x) - 1)}),
                    cint_lower =
                        apply(dd, 1, function(x)
                            {mean(x, na.rm = TRUE) - sqrt(var(x, na.rm = TRUE) / length(x)) * 2 * qt(0.995, length(x) - 1)}),
                    quantile_upper =
                        apply(dd, 1, function(x) {quantile(x, 0.9)}),
                    quantile_lower =
                        apply(dd, 1, function(x) {quantile(x, 0.1)}),
                    median =
                        apply(dd, 1, median))))

    }
    names(result) <- names(collapsed_data$values)
    return(list(iterations = unique(center_iteration), values = result))
}

summarize_matrix_data_across_iterations <-
    function(
        data_list,
        summary_function = mean,
        burnin_samples = 1
        )
{
    result <- list()
    for(l in data_list)
    {
        group_result <- list()
        for(m in l)
        {
            group_result <-
                append(
                    group_result,
                    list(apply(m[,,-c(1,burnin_samples)],
                               MARGIN = c(1,2),
                               FUN = summary_function))
                    )
        }
        matrix_result <- do.call("abind", list(group_result, along = 3))
        result <- append(result, list(matrix_result))
    }
    names(result) <- names(data_list)
    return(result)
}

summarize_matrix_data_across_iterations_and_runs <-
    function(
        data_list,
        iteration_summary_function = mean,
        run_summary_function = mean,
        burnin_samples = 1
        )
{
    result <- list()
    data_by_run <-
        summarize_matrix_data_across_iterations(
            data_list, iteration_summary_function, burnin_samples)
    for(l in data_by_run)
    {
        result <-
            append(
                result,
                list(apply(l, MARGIN = c(1,2), FUN = run_summary_function)))
    }
    names(result) <- names(data_list)
    return(result)
}


plot_scalar_by_iteration <-
    function(
        specs,
        output_type,
        paths,
        smoothing_window_size,
        summary_function = I,
        error_var = "cint",
        yrange = c(-Inf, Inf),
        burnin_samples = 10
        )
{
    results_list <- get_scalar_or_vector_data(specs, output_type, paths)
    collected_data <-
        collect_data_as_scalar(
            results_list, summary_function = summary_function
            )
    summarized_data <- summarize_scalar_data_across_runs(collected_data, smoothing_window_size)
    t <- summarized_data$iterations
    index_subset = t > burnin_samples
    ## calculate a suitable range to plot
    lowest_val <- Inf
    highest_val <- -Inf
    n_iterations = max(t, na.rm = TRUE)
    for(l in summarized_data$values)
    {
        lowest_val = max(min(lowest_val, min(l[[paste(error_var,"_lower", sep = "")]][index_subset], na.rm = TRUE)), yrange[1])
        highest_val = min(max(highest_val, max(l[[paste(error_var,"_upper", sep = "")]][index_subset], na.rm = TRUE)), yrange[2])
    }
    output_path <- paste(paths$fig_root, specs$results, "/", specs$comparison, "/", specs$dataset, "/", sep = "")
    if(!dir.exists(output_path)) dir.create(output_path, recursive = TRUE)
    pdf(paste(output_path, "/", output_type, ".pdf", sep = ""))
    plot(
        NULL, xlim = c(0, n_iterations), ylim = c(lowest_val, highest_val),
        xlab = "Iteration", ylab = output_type)
    groups <- specs$models
    plot_vars <- 1:length(unique(groups))
    names(plot_vars) <- unique(groups)
    for(g in unique(groups))
    {
        m <- summarized_data$values[[g]]$mean
        lwr <- summarized_data$values[[g]][[paste(error_var,"_lower",sep = "")]]
        upr <- summarized_data$values[[g]][[paste(error_var,"_upper",sep = "")]]
        lines(t[index_subset], m[index_subset], lty = plot_vars[g])
        lines(t[index_subset], lwr[index_subset], lty = plot_vars[g], lwd = 0.25)
        lines(t[index_subset], upr[index_subset], lty = plot_vars[g], lwd = 0.25)
        ## arrows(x0 = t[index_subset],
        ##        y0 = lwr[index_subset],
        ##        y1 = upr[index_subset],
        ##        angle = 90, code = 3,
        ##        length = 0.1, lty = plot_vars[g])
    }
    legend("bottomright", lty = plot_vars, legend = unique(groups))
    dev.off()
}

format_data_for_binary_matrix_plot <-
    function(
        specs, output_type, paths,
        iteration_summary_function = mean,
        run_summary_function = mean,
        burnin_samples = 1
        )
{
    ground_truth_data <-
        as.matrix(
            read.table(
                paste(paths$data, "/", "states.txt", sep = "")))
    data_list <- get_matrix_data(specs, output_type, paths)
    result_matrices <-
        summarize_matrix_data_across_iterations_and_runs(
            data_list,
            iteration_summary_function, run_summary_function,
            burnin_samples)
    return(list(gt = ground_truth_data, results = result_matrices))
}

plot_binary_matrices <- function(specs, data, paths)
{
    T <- nrow(data$gt)
    D <- ncol(data$gt)
    groups <- names(data$results)
    G <- length(groups)
    pdf(
        paste(
            paths$vis, specs$results, "/grids.pdf", sep = ""))
    par(mfrow = c(G + 1, 1), mar = c(1,3,1,1), mgp = c(1,1,0))
    image(data$gt, x = seq(0.5, T + 0.5), y = seq(0.5, D + 0.5),
          col = gray.colors(100), xlab = "", ylab = "Ground Truth",
          xaxt = "n", yaxt = "n")
    for(g in groups)
    {
        m <- data$results[[g]]
        ## image(data$gt, x = seq(0.5, T + 0.5), y = seq(0.5, D + 0.5),
        ##       col = gray.colors(100), xlab = "", ylab = g,
        ##       xaxt = "n", yaxt = "n")
        ## image(0.5 + sign(data$gt - m) * (data$gt - m)^2 / 2, x = seq(0.5, T + 0.5), y = seq(0.5, D + 0.5),
        ##       col = gray.colors(100), xlab = "", ylab = g,
        ##       xaxt = "n", yaxt = "n")
        image(m, x = seq(0.5, T + 0.5), y = seq(0.5, D + 0.5),
              col = gray.colors(100), xlab = "", ylab = g,
              xaxt = "n", yaxt = "n")
    }
    dev.off()
}


count_nonzero_entries_per_row <- function(data_matrix)
{
    return(apply(data_matrix, 1, function(x){return(sum(x != 0))}))
}

states_to_reach_one_minus_epsilon <- function(weight_vector_array, tol = 0.001)
{
    apply(
        weight_vector_array, 1,
        function(weight_vector)
        {
            sum(cumsum(sort(weight_vector, decreasing = TRUE)) < 1 - tol)
        }
        )
}

make_key_scalar_plots <-
    function(
        query_file,
        results_dir,
        data_set,
        burnin_samples,
        paths,
        comparison_name,
        smoothing_window_size,
        plot.vars = c("F1_score", "precision", "recall", "accuracy")
        )
{
    specs <- get_specs(query_file, results_dir, data_set, comparison_name)
    for(v in plot.vars)
    {
        plot_scalar_by_iteration(
            specs, v, burnin_samples = burnin_samples, paths = paths,
            summary_function = I,
            smoothing_window_size)
    }
    if("n_dot" %in% plot.vars)
    {
        plot_scalar_by_iteration(
            specs, "n_dot", burnin_samples = burnin_samples,
            summary_function = count_nonzero_entries_per_row,
            paths = paths,
            smoothing_window_size)
    }
    ## if(binary)
    ## {
    ##     binary_matrices <-
    ##         format_data_for_binary_matrix_plot(
    ##             specs, "thetastar",
    ##             burnin_samples = burnin_samples
    ##             )
    ##     plot_binary_matrices(specs, binary_matrices)
    ## }
}

make_scalar_plots_batch <-
    function(
        query_file,      #name of a text file w/ list of
                         #leaf subdirectories
        data_set,        #name of root directory after results_root
        burnin_samples,  #number of logged iteration to discard as burnin
        path_glob,       #a glob expression indicating which datasets within data_set to use
        smoothing_window_size,
        extra.plot.vars = c(),
        base.plot.vars = c("F1_score", "precision", "recall",
               "accuracy"),
        project_root = "../../../"
        )
{
    specs <-
        read.table(
            paste("../queries/", query_file, sep = ""),
            header = TRUE
        )
    comparisons <- specs$comparison
    root <- get_directories(project_root = project_root)
    cur_path <- getwd()
    results_dir <- paste(root$results, "/", data_set, sep = "")
    ## print(results_dir)
    setwd(results_dir)
    paths <- Sys.glob(path_glob)
    setwd(cur_path)
    for(p in paths)
    {
        print(p)
        for(comp in unique(comparisons))
        {
            print(paste("    ", comp, sep = ""))
            make_key_scalar_plots(
                query_file = query_file,
                results_dir = data_set,
                data_set = p,
                burnin_samples = burnin_samples,
                paths = root,
                comparison_name = comp,
                plot.vars = c(base.plot.vars, extra.plot.vars),
                smoothing_window_size
            )
            print("........done.")
        }
    }
}


collect.iterations <- function(path)
{
    result <- as.matrix(read.table(paste(path, "/", "thetastar/00000.txt", sep = "")))
    result <- array(the.array, c(dim(the.array), 1, 1))
    for(dd in dir(paste(path, "/thetastar", sep = ""))[-1])
    {
        new.array <- as.matrix(read.table(paste(d, "/thetastar/", dd, sep = "")))
        new.array <- array(new.array, c(dim(new.array), 1, 1))
        result <- abind(result, new.array, along = 3)
    }
    return(result)
}

create.thetastar.array <- function(root, exclusions)
{
    d <- dir(root)[1]
    subdir <- paste(root, "/", d, sep = "")
    the.array <- collect.iterations(subdir)
    ## the.array <- as.matrix(read.table(paste(d, "/", "thetastar/00000.txt", sep = "")))
    ## the.array <- array(the.array, c(dim(the.array), 1, 1))
    ## for(dd in dir(paste(d, "/thetastar", sep = ""))[-1])
    ## {
    ##     new.array <- as.matrix(read.table(paste(d, "/thetastar/", dd, sep = "")))
    ##     new.array <- array(new.array, c(dim(new.array), 1, 1))
    ##     the.array <- abind(the.array, new.array, along = 3)
    ## }
    for(d in dir(root)[-c(1,exclusions)])
    {
        ## slice.array <- as.matrix(read.table(paste(d, "/thetastar/00000.txt", sep = "")))
        ## slice.array <- array(slice.array, c(dim(slice.array), 1, 1))
        ## for(dd in dir(paste(d, "/thetastar", sep = ""))[-1])
        ## {
        ##     new.array <- as.matrix(read.table(paste(d, "/thetastar/", dd, sep = "")))
        ##     new.array <- array(new.array, c(dim(new.array), 1, 1))
        ##     slice.array <- abind(slice.array, new.array, along = 3)
        ## }
        ## the.array <- abind(the.array, slice.array, along = 4)
        subdir <- paste(root, "/", d, sep = "")
        the.array <- abind(the.array, collect.iterations(subdir), along = 4)
    }
}
