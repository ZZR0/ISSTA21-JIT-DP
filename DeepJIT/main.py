try:
    import traceback
    import argparse
    from padding import padding_data
    import pickle
    import numpy as np 
    from evaluation import evaluation_model
    from train import train_model
    from cam_eval import evaluation_model as cam_eval_model
    import matplotlib.pyplot as plt
    from utils import *
    import time
except ImportError:
    traceback.print_exc()
    print("Sorry, we didn't remind you to install this module, please use 'pip -install' to install it")

def read_args():
    parser = argparse.ArgumentParser()
     # Training our model
    parser.add_argument('-train', action='store_true', help='training DeepJIT model')  

    parser.add_argument('-train_data', type=str, help='the directory of our training data')   
    parser.add_argument('-dictionary_data', type=str, help='the directory of our dicitonary data')

    # Predicting our data
    parser.add_argument('-predict', action='store_true', help='predicting testing data')
    parser.add_argument('-cam', action='store_true', help='showing cam')
    parser.add_argument('-pred_data', type=str, help='the directory of our testing data')    

    # Predicting our data
    parser.add_argument('-load_model', type=str, help='loading our model')

    # Number of parameters for reformatting commits
    parser.add_argument('-msg_length', type=int, default=256, help='the length of the commit message')
    parser.add_argument('-code_line', type=int, default=10, help='the number of LOC in each hunk of commit code')
    parser.add_argument('-code_length', type=int, default=512, help='the length of each LOC of commit code')

    # Number of parameters for PatchNet model
    parser.add_argument('-embedding_dim', type=int, default=64, help='the dimension of embedding vector')
    parser.add_argument('-filter_sizes', type=str, default='1, 2, 3', help='the filter size of convolutional layers')
    parser.add_argument('-num_filters', type=int, default=64, help='the number of filters')
    parser.add_argument('-hidden_units', type=int, default=512, help='the number of nodes in hidden layers')
    parser.add_argument('-dropout_keep_prob', type=float, default=0.5, help='dropout for training DeepJIT')
    parser.add_argument('-l2_reg_lambda', type=float, default=1e-5, help='regularization rate')
    parser.add_argument('-learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('-batch_size', type=int, default=64, help='batch size')
    parser.add_argument('-num_epochs', type=int, default=25, help='the number of epochs')    
    parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')    

    # CUDA
    parser.add_argument('-device', type=int, default=-1,
                        help='device to use for iterate data, -1 mean cpu [default: -1]')
    parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the GPU')
    return parser

if __name__ == '__main__':
    params = read_args().parse_args()
    
    if params.train is True:
        data = pickle.load(open(params.train_data, 'rb'))
        ids, labels, msgs, codes = data 
        labels = np.array(labels)       

        dictionary = pickle.load(open(params.dictionary_data, 'rb'))   
        dict_msg, dict_code = dictionary

        pad_msg = padding_data(data=msgs, dictionary=dict_msg, params=params, type='msg')        
        pad_code = padding_data(data=codes, dictionary=dict_code, params=params, type='code')
    
        data = (pad_msg, pad_code, labels, dict_msg, dict_code)
        starttime = time.time()
        train_model(data=data, params=params)        
        endtime = time.time()
        dtime = endtime - starttime

        print("程序运行时间：%.8s s" % dtime)  #显示到微秒  
    elif params.predict is True:
        data = pickle.load(open(params.pred_data, 'rb'))
        ids, labels, msgs, codes = data 
        labels = np.array(labels)        

        dictionary = pickle.load(open(params.dictionary_data, 'rb'))   
        dict_msg, dict_code = dictionary

        pad_msg = padding_data(data=msgs, dictionary=dict_msg, params=params, type='msg')        
        pad_code = padding_data(data=codes, dictionary=dict_code, params=params, type='code')
        id2msg_world = get_world_dict(dict_msg)

        data = (ids, pad_msg, pad_code, labels, dict_msg, dict_code)
        if params.cam is True:
            word_rank = {}
            all_ids, all_msg, all_msg_mask, all_code, all_code_mask, all_predict, all_label = cam_eval_model(data=data, params=params)
            # all_msg_mask = all_msg_mask - np.min(all_msg_mask)
            # all_msg_mask = all_msg_mask / np.max(all_msg_mask)
            cam_data = [all_ids, all_msg, all_msg_mask, all_predict, all_label]
            # pickle.dump(cam_data, open(params.load_model+".cam.pkl", 'wb'))
            for _id, msg, maks, pred, label in zip(all_ids, all_msg, all_msg_mask, all_predict, all_label):
                msg = mapping_dict_world(msg, id2msg_world)
                for word, value in zip(msg,maks):
                    idx = str(int(pred // 0.1))
                    if not word in word_rank:
                        word_rank[word] = {'value':0, 'count':0}
                    # if value > 0:
                    
                    word_rank[word]['value'] += value
                    word_rank[word]['count'] += 1

            words = []
            for word in word_rank.items():
                if word[1]['count'] < 2: continue
                words.append((word[0], word[1]['value']/word[1]['count']))
            print('word size:', len(word_rank.keys()))
            words.sort(key=lambda x:x[1], reverse=True)
            
            with open(params.load_model+".cam.csv", 'w', encoding='utf-8') as f:
                f.write('word, value\n')
                for word in words:
                    line = '{},{}\n'.format(word[0], word[1])
                    f.write(line)
        else:
            starttime = time.time()
            evaluation_model(data=data, params=params)
            endtime = time.time()
            dtime = endtime - starttime

            print("程序运行时间：%.8s s" % dtime)  #显示到微秒  
    else:
        print('--------------------------------------------------------------------------------')
        print('--------------------------Something wrongs with your command--------------------')
        print('--------------------------------------------------------------------------------')
        exit()
