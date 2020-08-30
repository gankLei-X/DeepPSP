import sys
import pandas as pd
import numpy as np
import argparse
import csv
import numpy as np
from Method import get_Matrix_Label, get_Matrix_Label_2,write_result
from sklearn.metrics import roc_curve, auc
from model_DeepPSP import Deep_PSP_model_testing

def main():
    parser = argparse.ArgumentParser(
        description='DeepPSP: a prediction tool for general, kinase-specific phosphorylation prediction')
    parser.add_argument('-input', dest='inputfile', type=str, help='Data format of prediction should be .fasta',
                        required=True)
    parser.add_argument('-predict-type',
                        dest='predicttype',
                        type=str,
                        help='predict types. \'general\' for general human phosphorylation site prediction by models pre-trained in DeepPSP. \n \
                        \'kinase\' for kinase-specific human phosphorylation site prediction by models pre-trained in DeepPSP',
                        required=True)
    parser.add_argument('-output', dest='outputfile', type=str, help='prefix of the prediction results.', required=True)
    parser.add_argument('-kinase', dest='kinase', type=str,
                        help='if -predict-type is \'kinase\', -kinase indicates the specific kinase, currently we accept \'CDK\' or \'PKA\' or \'CK2\' or \'MAPK\' or \'PKC\' or \'CAMKL\' or \'PKG\' or \'GRK\'.',
                        required=False, default=None)
    parser.add_argument('-residue', dest='residues', type=str,
                        help='Residue types that to be predicted, only accept \'ST\' or \'Y\'',
                        required=True)

    args = parser.parse_args()

    kinaselist = ["CDK", "PKA", "CK2", "MAPK", "PKC", "AGC", "CAMK", "CMGC"];

    inputfile = args.inputfile;
    outputfile = args.outputfile;
    predicttype = args.predicttype;
    residues = args.residues
    kinase = args.kinase;

    m = 25
    n = 25

    if predicttype == 'general':
        modelname = "general_model_{:s}".format(residues)

        training_set = 'DATASET/train_general_%s###.fasta' % (residues)

        _, _, _, _, _, X_val1, X_val2, Y_val = get_Matrix_Label(training_set, residues, m, n)

        X_test, X_test_2, indexs, name, site_types = get_Matrix_Label_2(inputfile, residues, m, n)

    if predicttype == 'kinase':

        if kinase not in kinaselist:
            print(
                "wrong parameter for -kinase! Must be one of \'CDK\' or \'PKA\' or \'CK2\' or \'MAPK\' or \'PKC\'or \'AGC\'or \'CMGC\'or \'CAMK\' !\n");
            exit()

        modelname = "kinase_model_{:s}".format(kinase)

        training_set = 'DATASET/train_%s###.fasta' % (kinase)

        _, _, _, _, _, X_val1, X_val2, Y_val = get_Matrix_Label(training_set, residues, m, n)

        X_test, X_test_2, indexs, name, site_types = get_Matrix_Label_2(inputfile, residues, m, n)

    result, result_probe = Deep_PSP_model_testing(X_test, X_test_2, modelname, X_val1, X_val2, Y_val)

    write_result(modelname, indexs, name, site_types, result_probe,outputfile)

if __name__ == "__main__":
    main()

