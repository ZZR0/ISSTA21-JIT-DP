from torch._C import dtype
from jit_DExtended_model import DeepJITExtended
from jit_utils import mini_batches_DExtended, mini_batches_update_DExtended
from sklearn.metrics import roc_auc_score    
import torch 
from tqdm import tqdm
import numpy as np
import pickle

def eval(labels, predicts, thresh=0.5):
    TP, FN, FP, TN = 0, 0, 0, 0
    for lable, predict in zip(labels, predicts):
        # print(predict)
        if predict >= thresh and lable == 1:
            TP += 1
        if predict >= thresh and lable == 0:
            FP += 1
        if predict < thresh and lable == 1:
            FN += 1
        if predict < thresh and lable == 0:
            TN += 1
    
    # print(TP)
    try:
        P = TP/(TP+FP)
        R = TP/(TP+FN)

        A = (TP+TN)/len(labels)
        E = FP/(TP+FP)

        print('Test data at Threshold %.2f -- Accuracy: %.2f, False Positives: %.2f, Precision: %.2f, Recall: %.2f'%(thresh, A, E, P, R))
    except Exception:
        # division by zero
        pass

def save_result(labels, predicts, path):
    results = []
    for lable, predict in zip(labels, predicts):
        results.append('{}\t{}\n'.format(lable, predict))
    
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(results)
    

def evaluation_model(data, params, original_data):
    ids, labels, msgs, codes = original_data
    cc2ftr, pad_msg, pad_code, labels, dict_msg, dict_code = data
    batches = mini_batches_DExtended(X_ftr=cc2ftr, X_msg=pad_msg, X_code=pad_code, Y=labels)

    params.vocab_msg, params.vocab_code = len(dict_msg), len(dict_code)
    if len(labels.shape) == 1:
        params.class_num = 1
    else:
        params.class_num = labels.shape[1]
    params.embedding_ftr = cc2ftr.shape[1]

    # set up parameters
    params.cuda = (not params.no_cuda) and torch.cuda.is_available()
    del params.no_cuda
    params.filter_sizes = [int(k) for k in params.filter_sizes.split(',')]

    model = DeepJITExtended(args=params)
    if torch.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(torch.load(params.load_model))

    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        all_predict, all_label = list(), list()
        for i, (batch) in enumerate(tqdm(batches)):
            ftr, pad_msg, pad_code, label = batch
            if torch.cuda.is_available():
                ftr = torch.tensor(ftr).cuda()
                pad_msg, pad_code, labels = torch.tensor(pad_msg).cuda(), torch.tensor(
                    pad_code).cuda(), torch.cuda.FloatTensor(label)
            else:
                ftr = torch.tensor(ftr).long()
                pad_msg, pad_code, label = torch.tensor(pad_msg).long(), torch.tensor(pad_code).long(), torch.tensor(
                    labels).float()
            if torch.cuda.is_available():
                predict = model.forward(ftr, pad_msg, pad_code)
                predict = predict.cpu().detach().numpy().tolist()
            else:
                predict = model.forward(ftr, pad_msg, pad_code)
                predict = predict.detach().numpy().tolist()
            all_predict += predict
            all_label += labels.tolist()

    eval(all_label, all_predict, thresh=0.1)
    eval(all_label, all_predict, thresh=0.2)
    eval(all_label, all_predict, thresh=0.3)
    eval(all_label, all_predict, thresh=0.4)
    eval(all_label, all_predict, thresh=0.5)
    eval(all_label, all_predict, thresh=0.6)
    eval(all_label, all_predict, thresh=0.7)
    eval(all_label, all_predict, thresh=0.8)
    eval(all_label, all_predict, thresh=0.9)

    save_result(all_label, all_predict, params.load_model+'.result')

    auc_score = roc_auc_score(y_true=all_label,  y_score=all_predict)
    print('Test data -- AUC score:', auc_score)

    new_ids, new_predicts, new_labels, new_msgs, new_ftr, new_codes = [], [], [], [], [], []
    for _id, predict, label, msg, ftr, code in zip(ids, all_predict, all_label, msgs, cc2ftr, codes):
        new_ids.append(_id)
        new_labels.append(label)
        new_predicts.append(predict)
        new_msgs.append(msg)
        new_ftr.append(ftr)
        new_codes.append(code)

    new_data = [new_ids, new_labels, new_predicts, new_msgs, new_ftr, new_codes]
    
    with open(params.pred_data.replace('.pkl', '_eval.pkl'), 'wb') as f:
        pickle.dump(new_data, f)

def one_softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return x / np.sum(x, axis=0)

def evaluation_weight(data, params):
    cc2ftr, pad_msg, pad_code, labels, dict_msg, dict_code = data
    batches = mini_batches_update_DExtended(X_ftr=cc2ftr, X_msg=pad_msg, X_code=pad_code, Y=labels, mini_batch_size=2)
    batches = batches[:-1]

    params.vocab_msg, params.vocab_code = len(dict_msg), len(dict_code)
    if len(labels.shape) == 1:
        params.class_num = 1
    else:
        params.class_num = labels.shape[1]
    params.embedding_ftr = cc2ftr.shape[1]

    # set up parameters
    params.cuda = (not params.no_cuda) and torch.cuda.is_available()
    del params.no_cuda
    params.filter_sizes = [int(k) for k in params.filter_sizes.split(',')]

    model = DeepJITExtended(args=params)
    if torch.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(torch.load(params.load_model))

    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        ftr_weight, msg_weight, cod_weight = [], [], []
        for i, (batch) in enumerate(tqdm(batches)):
            ftr, pad_msg, pad_code, label = batch

            sw_ftr, sw_pad_msg, sw_pad_code = np.zeros(ftr.shape, dtype=ftr.dtype), \
                np.zeros(pad_msg.shape, dtype=pad_msg.dtype), np.zeros(pad_code.shape, dtype=pad_code.dtype)
            sw_ftr[[0,1]] = ftr[[1,0]]
            sw_pad_msg[[0,1]] = pad_msg[[1,0]]
            sw_pad_code[[0,1]] = pad_code[[1,0]]
            
            if torch.cuda.is_available():
                ftr = torch.tensor(ftr).cuda()
                sw_ftr = torch.tensor(sw_ftr).cuda()
                pad_msg, pad_code, labels = torch.tensor(pad_msg).cuda(), torch.tensor(
                    pad_code).cuda(), torch.cuda.FloatTensor(label)
                sw_pad_msg, sw_pad_code = torch.tensor(sw_pad_msg).cuda(), torch.tensor(
                    sw_pad_code).cuda()
            else:
                ftr = torch.tensor(ftr).long()
                pad_msg, pad_code, label = torch.tensor(pad_msg).long(), torch.tensor(pad_code).long(), torch.tensor(
                    label).float()
                sw_ftr = torch.tensor(sw_ftr).long()
                sw_pad_msg, sw_pad_code, label = torch.tensor(sw_pad_msg).long(), torch.tensor(sw_pad_code).long()

            predict = model.forward(ftr, pad_msg, pad_code)
            predict = predict.cpu().detach().numpy().tolist()
            ftr_predict = model.forward(sw_ftr, pad_msg, pad_code)
            ftr_predict = ftr_predict.cpu().detach().numpy().tolist()
            msg_predict = model.forward(ftr, sw_pad_msg, pad_code)
            msg_predict = msg_predict.cpu().detach().numpy().tolist()
            cod_predict = model.forward(ftr, pad_msg, sw_pad_code)
            cod_predict = cod_predict.cpu().detach().numpy().tolist()

            if predict[1] - predict[0] == 0:
                dirt = 0
            else:
                dirt = predict[1] - predict[0] / (np.abs(predict[1] - predict[0]))

            ftr_weight += [dirt * (ftr_predict[0] - predict[0])]
            msg_weight += [dirt * (msg_predict[0] - predict[0])]
            cod_weight += [dirt * (cod_predict[0] - predict[0])]

            ftr_weight += [-dirt * (ftr_predict[1] - predict[1])]
            msg_weight += [-dirt * (msg_predict[1] - predict[1])]
            cod_weight += [-dirt * (cod_predict[1] - predict[1])]
    
    ftr_weight, msg_weight, cod_weight = np.mean(ftr_weight), np.mean(msg_weight), np.mean(cod_weight)
    print(ftr_weight, msg_weight, cod_weight)
    weight = one_softmax(np.array([ftr_weight, msg_weight, cod_weight]))
    print('The weight of three input\ncc2tr: %.2f   message: %.2f   code: %.2f'%(weight[0], weight[1], weight[2]))