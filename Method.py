import numpy as np
import random
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

def file2str(filename):
    fr = open(filename)  # 打开文件
    numline = fr.readlines()  # 读取文件的行向量

    index = -1
    A = []
    F = []
    for eachline in numline:
        index += 1
        if '>' in eachline:
            A.append(index)
    A.append(index+1)

    B = []
    for eachline in numline:
        line = eachline.strip()
        listfoemline = line.split()
        B.append(listfoemline)

    name = []
    for i in range(len(A) - 1):
        K = A[i]
        input_sequence = str(B[K])
        input_sequence = input_sequence[3:-2]
        name.append(input_sequence)

    for i in range(len(A)-1):
        K = A[i]
        input_sequence = B[K + 1]
        input_sequence = str(input_sequence)
        input_sequence = input_sequence[1:-1]

        for j in range(A[i + 1] - A[i]):
            if K < A[i + 1] - 2:
                C = str(B[K + 2])
                input_sequence = input_sequence + C[1:-1]
                K += 1
        input_sequence = input_sequence.replace('\'', '')
        F.append(input_sequence)

    return name,F


def separt_positive(sequence, m, n):
    indexs = []
    for k in range(len(sequence)):
        sequence[k] = '****************************************************************************************************************************************************************************************************************************' + sequence[k] + '**********************************************************************************************************************************************************************************************************************************************************************************************'
        index = []
        kk = 0
        for i in range(len(sequence[k])):
            if sequence[k][i] == '#':
                index.append(i - kk - 1)
                kk += 1
        indexs.append(index)
    sub_sequences = []
    globalsa = []
    for i in range(len(sequence)):
        sequence[i] = sequence[i].translate(str.maketrans('', '', '#'))
        for k in range(len(indexs[i])):
            sub_sequence = sequence[i][indexs[i][k] - m:indexs[i][k] + n + 1]
            sub_sequences.append(sub_sequence)
            globalsa.append(i)
    return sub_sequences,np.array(globalsa)


def separt_negative(sequence, m, n):

    sequences = []
    for i in range(len(sequence)):
        if '#' in sequence[i]:
            sequences.append(sequence[i])
    sequence = sequences

    indexs = []
    for k in range(len(sequence)):
        sequence[k] = '**************************************************************************************************************************************************************************************************************************************************' + sequence[k] + '**************************************************************************************************************************************************************************************************************************************************'
        index = []
        kk = 0
        for i in range(len(sequence[k])):
            if sequence[k][i] == '#':
                index.append(i - kk - 1)
                kk += 1
        indexs.append(index)

    indexs2 = []
    for k in range(len(sequence)):

        sequence[k] = sequence[k].translate(str.maketrans('', '', '#'))
        index = []
        for i in range(len(sequence[k])):
            if sequence[k][i] == 'S' or sequence[k][i] == 'T':
                index.append(i)
        indexs2.append(index)
    indexs3 = []
    for i in range(len(indexs)):
        c = [x for x in indexs[i] if x in indexs2[i]]
        d = [y for y in (indexs[i] + indexs2[i]) if y not in c]
        indexs3.append(d)
    sub_sequences = []
    globals = []
    for i in range(len(sequence)):
        for k in range(len(indexs3[i])):
            sub_sequence = sequence[i][indexs3[i][k] - m:indexs3[i][k] + n + 1]
            sub_sequences.append(sub_sequence)
            globals.append(i)
    return sub_sequences,np.array(globals)


def separt_positive_2(sequence, m, n):
    indexs = []
    for k in range(len(sequence)):
        sequence[k] = '****************************************************************************************************************************************************************************************************************************' + sequence[k] + '**********************************************************************************************************************************************************************************************************************************************************************************************'
        index = []
        kk = 0
        for i in range(len(sequence[k])):
            if sequence[k][i] == '#':
                index.append(i - kk - 1)
                kk += 1
        indexs.append(index)
    sub_sequences = []
    globalsa = []
    for i in range(len(sequence)):
        sequence[i] = sequence[i].translate(str.maketrans('', '', '#'))
        for k in range(len(indexs[i])):
            sub_sequence = sequence[i][indexs[i][k] - m:indexs[i][k] + n + 1]
            sub_sequences.append(sub_sequence)
            globalsa.append(i)
    return sub_sequences,np.array(globalsa)


def separt_negative_2(sequence, m, n):

    sequences = []
    for i in range(len(sequence)):
        if '#' in sequence[i]:
            sequences.append(sequence[i])
        else:
            print('??')
    sequence = sequences

    indexs = []
    for k in range(len(sequence)):
        sequence[k] = '**************************************************************************************************************************************************************************************************************************************************' + sequence[k] + '**************************************************************************************************************************************************************************************************************************************************'
        index = []
        kk = 0
        for i in range(len(sequence[k])):
            if sequence[k][i] == '#':
                index.append(i - kk - 1)
                kk += 1
        indexs.append(index)

    indexs2 = []
    for k in range(len(sequence)):

        sequence[k] = sequence[k].translate(str.maketrans('', '', '#'))
        index = []
        for i in range(len(sequence[k])):
            if sequence[k][i] == 'Y':
                index.append(i)
        indexs2.append(index)
    indexs3 = []
    for i in range(len(indexs)):
        c = [x for x in indexs[i] if x in indexs2[i]]
        d = [y for y in (indexs[i] + indexs2[i]) if y not in c]
        indexs3.append(d)
    sub_sequences = []
    globals = []
    for i in range(len(sequence)):
        for k in range(len(indexs3[i])):
            sub_sequence = sequence[i][indexs3[i][k] - m:indexs3[i][k] + n + 1]
            sub_sequences.append(sub_sequence)
            globals.append(i)
    return sub_sequences,np.array(globals)


def get_test_file_ST(sequence, m, n):

    id = []
    indexs2 = []
    site_types = []

    for k in range(len(sequence)):
        sequence[k] = '**************************************************************************************************************************************************************************************************************************************************' + \
                 sequence[k] + '**************************************************************************************************************************************************************************************************************************************************'

        index = []
        site_type = []
        for i in range(len(sequence[k])):
            if sequence[k][i] == 'S' or sequence[k][i] == 'T':
                index.append(i)
                site_type.append(sequence[k][i])
        indexs2.append(index)
        id.append(np.array(index) - 241)
        site_types.append(site_type)
    sub_sequences = []
    globals = []

    for i in range(len(sequence)):
        for k in range(len(indexs2[i])):
            sub_sequence = sequence[i][indexs2[i][k] - m:indexs2[i][k] + n + 1]
            sub_sequences.append(sub_sequence)
            globals.append(i)

    return sub_sequences, np.array(globals), id, site_types

def get_test_file_Y(sequence, m, n):
    id = []
    site_types = []
    indexs2 = []
    for k in range(len(sequence)):
        sequence[
            k] = '**************************************************************************************************************************************************************************************************************************************************' + \
                 sequence[
                     k] + '**************************************************************************************************************************************************************************************************************************************************'

        index = []
        site_type = []
        for i in range(len(sequence[k])):
            if sequence[k][i] == 'Y':
                index.append(i)
                site_type.append(sequence[k][i])
        indexs2.append(index)
        site_types.append(site_type)
        id.append(np.array(index) - 241)
    sub_sequences = []
    globals = []
    for i in range(len(sequence)):
        for k in range(len(indexs2[i])):
            sub_sequence = sequence[i][indexs2[i][k] - m:indexs2[i][k] + n + 1]
            sub_sequences.append(sub_sequence)
            globals.append(i)

    return sub_sequences,np.array(globals), id, site_types

def str2dic(input_sequence):
    char = sorted(
        ['G', 'A', 'V', 'L', 'I', 'P', 'F', 'Y', 'W', 'S', 'T', 'C', 'M', 'N', 'Q', 'D', 'E', 'K', 'R', 'H'])

    char_to_index = {}
    index = 1
    result_index = []
    for c in char:
        char_to_index[c] = index
        index = index + 1
    char.append('*')
    char.append('U')
    char.append('B')
    char_to_index['*'] = 0
    char_to_index['U'] = char_to_index['D']
    char_to_index['B'] = char_to_index['D']
    for word in input_sequence:
        result_index.append(char_to_index[word])
    return result_index


def vec_to_onehot(mat,pc,kk,mmm=51):
    m = len(mat)
    return_mat = np.zeros((m, mmm, kk))
    for i in range(len(mat)):
        metrix = np.zeros((mmm, kk))
        for j in range(len(mat[i])):
            metrix[j] = pc[mat[i][j]]
        return_mat[i,:,:] = metrix
    return return_mat

def get_Matrix_Label (filename,site_type, m, n):


    name, input_sequence = file2str(filename)
    input_sequence_2 = np.copy(input_sequence)
    input_sequence_3 = np.copy(input_sequence)

    if site_type == 'Y':
        sequence_positive, globals_positive = separt_positive_2(input_sequence, m, n)
        sequence_negative, globals_negative = separt_negative_2(input_sequence_2, m, n)

    else:

        sequence_positive,globals_positive = separt_positive(input_sequence, m, n)
        sequence_negative,globals_negative = separt_negative(input_sequence_2,m,n)

    num_positive = len(sequence_positive)
    num_negative = len(sequence_negative)

    num_val = int(num_positive/10)

    X_train_positive = []
    for i in range(len(sequence_positive)):
        result_index = str2dic(sequence_positive[i])
        X_train_positive.append(result_index)

    X_train_negative = []
    for i in range(len(sequence_negative)):
        result_index = str2dic(sequence_negative[i])
        X_train_negative.append(result_index)

    random.seed(0)
    X_train_positive = np.array(X_train_positive)
    X_train_negative = np.array(X_train_negative)

    ls = list((range(len(X_train_positive))))
    random.shuffle(ls)

    X_val_positive = X_train_positive[ls][:num_val]
    X_train_positive = X_train_positive[ls][num_val:]

    global_val_positive = globals_positive[ls][:num_val]
    global_train_positive = globals_positive[ls][num_val:]

    random.seed(1)
    ls2 = list((range(len(X_train_negative))))
    random.shuffle(ls2)
    X_val_negative = X_train_negative[ls2][:num_val]
    X_train_negative = X_train_negative[ls2][num_val:]

    global_val_negative = globals_negative[ls2][:num_val]
    global_train_negative = globals_negative[ls2][num_val:]
    X_val = np.vstack((X_val_positive, X_val_negative))
    for kk in range(len(input_sequence_3)):
        input_sequence[kk] = input_sequence_3[kk].translate(str.maketrans('', '', '#'))

    globel_input_sequence = []
    for kk in range(len(input_sequence)):
        result_index = str2dic(input_sequence[kk])
        globel_input_sequence.append(result_index)

    input_sequence = pad_sequences(globel_input_sequence,maxlen = 2000)
    X_val2 = np.vstack((input_sequence[global_val_positive],input_sequence[global_val_negative]))
    X_val2 = to_categorical(X_val2)
    X_val1 = to_categorical(X_val)

    Y = [0]*(num_positive-num_val)+[1]*(num_negative-num_val)
    Y_val = [0]*num_val+[1]*num_val

    Y_val = to_categorical(Y_val)
    return X_train_positive, X_train_negative, global_train_positive, global_train_negative, Y, X_val1, X_val2,Y_val


def get_Matrix_Label_2 (filename,site_type, m, n):

    name, input_sequence = file2str(filename)

    if site_type == 'Y':

        sequence,globals, indexs, site_types = get_test_file_Y(input_sequence, m, n)

    else:

        sequence,globals, indexs, site_types = get_test_file_ST(input_sequence, m, n)

    X_train = []
    for i in range(len(sequence)):
        result_index = str2dic(sequence[i])
        X_train.append(result_index)

    random.seed(0)
    X_test = np.array(X_train)

    name, input_sequence = file2str(filename)

    globel_input_sequence = []
    for kk in range(len(input_sequence)):
        result_index = str2dic(input_sequence[kk])
        globel_input_sequence.append(result_index)
    input_sequence = pad_sequences(globel_input_sequence, maxlen=2000)
    X_test2 = input_sequence[globals]

    return X_test, X_test2,indexs, name, site_types

def write_result(modelname,indexs,name,site_types,result_probe,outputfile):
    with open(modelname + outputfile, 'w') as f:
        num = 0
        for k in range(len(indexs)):
            f.write('>')
            f.write(name[k])
            f.write('\n')

            for u in range(len(indexs[k])):

                f.write(str(indexs[k][u]))
                f.write('\t')
                f.write(site_types[k][u])
                f.write('\t')
                f.write(str(result_probe[num]))
                f.write('\n')
                num +=1

    print('Successfully predict the phosphorylation site !')