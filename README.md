# DeepPSP

DeepPSP provides the prediction of general and kinase-specific phosphorylation site using deep leraning. Developer is Lei Guo from Laboratory of Biomedical Network, Department of Electronic Science, Xiamen University of China.

# Requirement

    python == 3.5, 3.6 or 3.7
    
    keras == 2.1.2
    
    tensorflow == 1.14.0

    numpy >= 1.8.0

    backend == tensorflow

# Predict For Your Test Data

If you want to predict general site, run:

    python predict.py -input ...\DATASET\test_general_ST.fasta -predict-type general -output ...\result_general_model_ST -residue ST
    
If you want to predict kinase-specific site, taking MAPK as a example, run:

    python predict.py -input ...\DATASET\test_MAPK.fasta -predict-type kinase -kinase MAPK -output ...\result_kinase_model_MAPK -residue ST
 
 Output file including three columns, position, residue type and score. The larger the score, the more likely this site is to be phosphorylated.

# Train For Your own Data

“#” should be first added after each phosphorylation site

if you want to train for general site, then run:

    python train.py -input ...\DATASET\train_general_ST###.fasta -train-type general -residue ST   

if you want to train for kinase-specific site, taking MAPK as a example, then run:

    python train.py -input ...\DATASET\train_MAPK###.fasta -train-type kinase -kinase MAPK -residue ST   

Note that pre-training model would been loaded before kinase-specific site training. If you don't need it, you can train model on general site. 
    
# Contact

Please contact me if you have any help: gl5121405@gmail.com
