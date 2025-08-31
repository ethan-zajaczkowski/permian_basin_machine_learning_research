# Packages
library(doParallel)

hdbscan_analysis = function(mode = 0, points_final = NULL, minPts_range = 2:8, n_cores = 8) {
  # This function will iterate through the minPts_range and perform the HDBSCAN algorithm on the points_final dataset (presumably the raw_viirs_dataset)
  
  # Parameters:
  # - Mode = 0 → run the full analysis with the code written. However, the code has already been pre-ran and saved in "permian_basin_hdbscan_results.csv".
  # - Mode = 1 → will load the pre-ran csv.
  # - points_final = raw_viirs_dataframe with coordinate data (required for mode 0)
  # - minPts_range = range of minPts values to test (e.g., 3 or 4:8)
  # - n_cores = number of cores to register for parallel processing
  
  csv_path = "datasets/hdbscan_datasets/raw_hdbscan_results.csv"
  
  if (mode == 0) {
    
    if (is.null(points_final)) {
      stop("Please upload a dataset to points_final parameter.")
    }
    
    registerDoParallel(cores = n_cores)
    
    # Part A: Convert raw data into a numeric matrix of coordinates. Handles both sptial and regular dataframes
    if ("sf" %in% class(points_final)) {
      points_final = st_drop_geometry(points_final)
    }
    
    points_matrix = points_final %>%
      select(lon_km_utm13, lat_km_utm13) %>%
      mutate(across(everything(), as.numeric)) %>%
      as.matrix()
    
    # Part B: Check if there are any missing values that could break HDBSCAN. Remove them.
    if (any(is.na(points_matrix))) {
      na_count = sum(is.na(points_matrix))
      cat(sprintf("Warning: Found %d NA values. Removing rows with NAs...\n", na_count))
      points_matrix = na.omit(points_matrix)
      cat(sprintf("After cleaning: %d rows remaining\n", nrow(points_matrix)))
    }
    
    # Part C: Initialize results dataframe
    results = data.frame(
      minPts = numeric(),
      clustered_obs = numeric(),
      noise_obs = numeric(),
      noise_percentage = numeric(),
      clusters = numeric(),
      ratio_flares_per_cluster = numeric()
    )
    
    # Part D: HDBSCAN Analysis Start
    total_iterations = length(minPts_range)
    cat(sprintf("Starting HDBSCAN analysis for %d minPts values: %s\n", total_iterations, paste(minPts_range, collapse = ", ")))
    
    for (i in seq_along(minPts_range)) {
      
      minPts = minPts_range[i]
      cat(sprintf("Processing minPts = %d (%d/%d)...\n", minPts, i, total_iterations))
      raw
      # HDBSCAN
      hdbscan_result = hdbscan(points_matrix, minPts = minPts)
      
      # Points In Cluster
      clustered_obs = sum(hdbscan_result$cluster != 0)
      
      # Points Classified As Noise
      noise_obs = sum(hdbscan_result$cluster == 0)
      
      # Noise Percentage
      noise_percentage = (noise_obs / nrow(points_matrix)) * 100
      
      # Exclude noise cluster 0
      clusters = length(unique(hdbscan_result$cluster)) - 1
      
      # Ratio of flares per cluster
      ratio_flares_per_cluster = clustered_obs / clusters
      
      # Update results DataFrame
      results = rbind(results, data.frame(
        minPts = minPts,
        clustered_obs = clustered_obs,
        noise_obs = noise_obs,
        noise_percentage = noise_percentage,
        clusters = clusters,
        ratio_flares_per_cluster = ratio_flares_per_cluster
      ))
    }
    
    # Part E: Results And Saving
    cat("\nResults Summary:\n")
    print(kable(results, caption = "Summary Statistics for HDBSCAN Clustering"))
    write.csv(results, csv_path, row.names = FALSE)
    cat(sprintf("\nResults saved to: %s\n", csv_path))
    
    return(results)
    
  } else if (mode == 1) {
    results = read.csv(csv_path, stringsAsFactors = FALSE)
    print(kable(results, caption = "Summary Statistics for HDBSCAN Clustering"))
    
    return(results)
    
  } else {
    stop("`mode` must be 0 (run analysis) or 1 (load results).")
  }
}


hdbscan_dataset_merge = function(mode = 0, points_final = NULL, minPts = 6) {
  # This function will merge the points_final dataset (presumably the raw_viirs_dataset) and the selected minPts HDBSCAN dataset.
  # 
  # Mode = 0 → run HDBSCAN on minPts
  # Mode = 1 → load existing HDBSCAN dataset with minPts
  # points_final = raw_viirs_dataframe with coordinate data (required for mode 0)
  # minPts = minimum points parameter for HDBSCAN (required for mode 0)
  
  csv_path = "datasets/hdbscan_datasets/raw_hdbscan_merged_dataset.csv"
  
  if (mode == 0) {
    
    if (is.null(points_final)) {
      stop("Please upload a dataset to points_final parameter.")
    }
    
    # Part A: Convert raw data into a numeric matrix of coordinates
    coords_for_hdbscan = points_final %>%
      st_drop_geometry() %>%
      select(lon_km_utm13, lat_km_utm13) %>%
      mutate(across(everything(), as.numeric)) %>%
      filter(!is.na(lon_km_utm13) & !is.na(lat_km_utm13))
    points_matrix = as.matrix(coords_for_hdbscan)
    
    # Part B: Perform HDBSCAN
    hdbscan_result = hdbscan(points_matrix, minPts = minPts)
    
    # Part C: Load results of the HDBSCAN into a df (to prepare for merging)
    hdbscan_df = data.frame(
      lon_km_utm13 = coords_for_hdbscan$lon_km_utm13,
      lat_km_utm13 = coords_for_hdbscan$lat_km_utm13,
      cluster = hdbscan_result$cluster,
      cluster_prob = hdbscan_result$membership_prob
    )
    
    # Part D: Merge the HDBSCAN results onto the raw VIIRS dataframe
    merged_data = points_final %>% left_join(hdbscan_df, by = c("lon_km_utm13", "lat_km_utm13"))
    write.csv(merged_data, csv_path, row.names = FALSE)
    
    return(merged_data)
    
  } else if (mode == 1) {
    hdbscan_dataset = read.csv(csv_path, stringsAsFactors = FALSE)
    return(hdbscan_dataset)
    
  } else {
    stop("`mode` must be 0 (run HDBSCAN) or 1 (load dataset).")
  }
}


hdbscan_noise_distance_calculation = function(mode = 0, noise_with_methane = NULL, hdbscan_dataset_filtered = NULL) {
  # This function will first, classified each cluster with a Convex Hull, then calculate the distance (m) between the noise point and the nearest edge.
  # 
  # Mode = 0 → calculate distances and save to CSV
  # Mode = 1 → load existing CSV (pre-trained)
  
  csv_path = "datasets/hdbscan_datasets/non_na_methane_noise_points.csv"
  
  if (mode == 0) {
    calculate_nearest_cluster_distance_hull = function(point_lon, point_lat, filtered_data) {
      # Use convex hull to approximate the closest cluster edge to the point
      
      clusters = unique(filtered_data$cluster)
      min_distance = Inf
      nearest_cluster_id = NA
      
      for(cluster_id in clusters) {
        cluster_points = filtered_data %>% filter(cluster == cluster_id)
        
        hull_indices = chull(cluster_points$lon_km_utm13, cluster_points$lat_km_utm13)
        hull_points = cluster_points[hull_indices, ]
        
        distances_to_cluster = sqrt((hull_points$lon_km_utm13 - point_lon)^2 + (hull_points$lat_km_utm13 - point_lat)^2)
        min_distance_to_cluster = min(distances_to_cluster, na.rm = TRUE)
        
        if(min_distance_to_cluster < min_distance) {
          min_distance = min_distance_to_cluster
          nearest_cluster_id = cluster_id
        }
      }
      
      return(list(distance = min_distance, cluster = nearest_cluster_id))
    }
    
    noise_distances = noise_with_methane %>%
      rowwise() %>%
      mutate(
        nearest_info = list(calculate_nearest_cluster_distance_hull(lon_km_utm13, lat_km_utm13, hdbscan_dataset_filtered)),
        nearest_cluster_distance = nearest_info$distance,
        nearest_cluster = nearest_info$cluster
      ) %>%
      select(-nearest_info) %>%
      ungroup()
    
    write.csv(noise_distances, csv_path, row.names = FALSE)
    return(noise_distances)
    
  } else if (mode == 1) {
    noise_distances = read.csv(csv_path, stringsAsFactors = FALSE)
    return(noise_distances)
    
  } else {
    stop("`mode` must be 0 (calculate and save) or 1 (load existing).")
  }
}