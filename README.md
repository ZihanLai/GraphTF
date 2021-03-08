# GraphTF
Graph Attention Mechanism-based Deep Tensor Factorization for Predicting disease-associated miRNA-miRNA pairs.
# Requirements
pytorch(tested on version 1.6.0)  numpy(tested on version 1.20.1)  sklearn(tested on version 0.24.1)  tensorly(tested on version 0.5.1), 

# Quick start To reproduce our results
Run main_cv.py to RUN GraphTF.

# Data description:
* 283_dis_semantic_sim_matrix.csv:disease semantic similarity matrix.   
* 325_mir_functional_sim_matrix.csv: miRNA functional similarity matrix.   
* 325_mir_seq_sim_matrix.csv: miRNA sequence similarity matrix.  
* 283_disease_name.csv: list of disease names.   
* 325_miRNA_name.csv: list of miRNA names.   
* Index_mir_pair_dis_link.csv: miRNA-miRNA-disease association matrix.  
* negative_sample_15484_xxx.csv: ten randomly sampled negative miRNA-miRNA-disease association.
