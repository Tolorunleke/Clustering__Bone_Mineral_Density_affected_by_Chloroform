Project Summary

Title: Clustering Analysis for Chloroform Exposure and Bone Mineral Density

Overview

This project explores the application of clustering techniques to analyze the relationship between chloroform exposure and Bone Mineral Density (BMD). The goal is to uncover patterns in unlabeled datasets, providing insights into the impact of chloroform on bone health. K-means and Hierarchical clustering are employed for the study.

Methodology

Background
Clustering has proven effective in various fields, but its potential in understanding the relationship between environmental factors and BMD is untapped.
Harvard Dataverse provided chloroform exposure and BMD data from National Health and Nutrition Examination Surveys (NHANES).
Data Exploration
Utilizes pandas to import and explore the dataset, including information on age, sex, health status, blood profile, biochemistry, bone X-rays, chloroform concentrations, and more.
Focuses on key BMD data points, such as TotalFemurBMD, FemoralNeckBMD, TrochanterBMD, IntertrochanterBMD, WardsTriangleBMD, and TotalSpineBMD.
Data Preprocessing
Addresses missing values using SimpleImputer for mean and mode imputation.
Utilizes standardization and z-score-based outlier elimination.
Applies variance threshold method for feature selection.
Clustering Techniques
K-means Clustering:
Focuses on clustering chloroform and TotalFemur, the most important bone in the human body.
Evaluates optimal k using the elbow curve, determining three clusters as the optimal choice.
Applies k-means clustering and visualizes the clusters.
Hierarchical Clustering:
Utilizes hierarchical clustering to detect hidden patterns and manage noisy data.
Determines the number of clusters by inspecting the dendrogram, leading to three clusters.
Applies agglomerative clustering and visualizes the clusters.
Multiple Class Clustering:
Extends clustering to all sub-selected features using k-means.
Determines the optimal k using silhouette score.
Applies k-means clustering and visualizes the clusters.
Results

Evaluation Metrics
K-means:
Silhouette Score: 0.40 (Outperforms other clustering methods)
Hierarchical Clustering:
Silhouette Score: 0.375
Cluster Analysis
K-means:
Provides better performance with a silhouette score of 0.40.
Three clusters are identified, visually identifiable and presentable.
Hierarchical Clustering:
Shows positive results with a silhouette score of 0.375.
Patient clusters around the three groups are visually identifiable.
Multiple Class Clustering:
Silhouette method used to determine k, providing insights into the average characteristics of each cluster.
Conclusion
K-means performs better:
Judged by the silhouette score, k-means outperforms other clustering methods.
Patients in Cluster 0 exhibit higher chloroform presence, leading to lower bone density.
Patients in Cluster 1 have lower chloroform presence and higher bone density.
Implications for treatment and environmental precautions based on cluster identification.
Future Directions

Further analysis and collaboration with medical professionals to validate findings.
Integration of additional environmental factors for a comprehensive study.
Continuous refinement of clustering models for enhanced predictive capabilities.
