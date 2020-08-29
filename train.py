import matplotlib
matplotlib.use('Agg')
from Method import get_Matrix_Label,get_Matrix_Label_3
from model_DeepPSP import Deep_PSP_model_training
import argparse

def main():

    parser = argparse.ArgumentParser(
        description='DeepPSP: a prediction tool for general, kinase-specific phosphorylation prediction')
    parser.add_argument('-input', dest='inputfile', type=str, help='Data format of prediction should be .fasta',
                        required=True)
    parser.add_argument('-train-type',
                        dest='traintype',
                        type=str,
                        help='train types. \'general\' for general human phosphorylation prediction trained by DeepPSP. \n \
                            \'kinase\' for kinase-specific human phosphorylation prediction trained by DeepPSP.',
                        required=True)
    parser.add_argument('-kinase', dest='kinase', type=str,
                        help='if -train-type is \'kinase\', -kinase indicates the specific kinase.',
                        required=False, default=None)
    parser.add_argument('-residue', dest='residues', type=str,
                        help='Residue types that to be trained, only accept \'ST\' or \'Y\'',
                        required=True)

    args = parser.parse_args()

    inputfile = args.inputfile;
    traintype = args.traintype;
    residues = args.residues
    kinase = args.kinase;

    m = 25
    n = 25

    X_train_positive, X_train_negative, global_train_positive, global_train_negative, Y, X_val1, X_val2,Y_val = get_Matrix_Label_3(inputfile,residues, m, n)

    if traintype == 'general':
        modelname = "general_model_{:s}".format(residues)
        Deep_PSP_model_training(inputfile, X_train_positive, X_train_negative, global_train_positive, global_train_negative, Y, X_val1, X_val2,Y_val,modelname)

    if traintype == 'kinase':
        modelname = "kinase_model_{:s}".format(kinase)
        Deep_PSP_model_training(inputfile, X_train_positive, X_train_negative, global_train_positive, global_train_negative, Y, X_val1, X_val2,Y_val,modelname,pretraining_model = True)

if __name__ == "__main__":

    main()

