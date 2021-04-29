from model import DeepJIT
from utils import mini_batches_test
from sklearn.metrics import roc_auc_score    
import torch 
from tqdm import tqdm

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
    P = TP/(TP+FP)
    R = TP/(TP+FN)

    A = (TP+TN)/len(labels)
    E = FP/(TP+FP)

    print('Test data at Threshold %.2f -- Accuracy: %.2f, False Positives: %.2f, Precision: %.2f, Recall: %.2f'%(thresh, A, E, P, R))

def save_result(ids, labels, predicts, path):
    results = []
    for id, lable, predict in zip(ids, labels, predicts):
        results.append('{}\t{}\n'.format(lable, predict))
    
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(results)

def evaluation_model(data, params):
    ids, pad_msg, pad_code, labels, dict_msg, dict_code = data
    batches = mini_batches_test(ids=ids, X_msg=pad_msg, X_code=pad_code, Y=labels)

    params.vocab_msg, params.vocab_code = len(dict_msg), len(dict_code)
    if len(labels.shape) == 1:
        params.class_num = 1
    else:
        params.class_num = labels.shape[1]

    # set up parameters
    params.cuda = (not params.no_cuda) and torch.cuda.is_available()
    del params.no_cuda
    params.filter_sizes = [int(k) for k in params.filter_sizes.split(',')]

    model = DeepJIT(args=params)
    if torch.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(torch.load(params.load_model))

    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    all_id, all_predict, all_label = list(), list(), list()
    with torch.no_grad():
        for i, (batch) in enumerate(tqdm(batches)):
            _id, pad_msg, pad_code, label = batch
            if torch.cuda.is_available():                
                pad_msg, pad_code, labels = torch.tensor(pad_msg).cuda(), torch.tensor(
                    pad_code).cuda(), torch.cuda.FloatTensor(label)
            else:                
                pad_msg, pad_code, label = torch.tensor(pad_msg).long(), torch.tensor(pad_code).long(), torch.tensor(
                    label).float()
            if torch.cuda.is_available():
                predict = model.forward(pad_msg, pad_code)
                predict = predict.cpu().detach().numpy().tolist()
            else:
                predict = model.forward(pad_msg, pad_code)
                predict = predict.detach().numpy().tolist()
            all_predict += predict
            all_label += label.tolist()
            all_id += _id

    # with open('result.txt', 'w', encoding='utf-8') as f:
    #     results = ['{}, {}\n'.format(label, predict) for label, predict in zip(all_label, all_predict)]
    #     f.writelines(results)

    for thresh in [i/10 for i in range(1,10)]:
        try:
            eval(all_label, all_predict, thresh=thresh)
        except Exception as identifier:
            print("No predict larger than %f" % (thresh))

    save_result(all_id, all_label, all_predict, params.load_model+'.result')

    auc_score = roc_auc_score(y_true=all_label,  y_score=all_predict)
    print('Test data -- AUC score:', auc_score)