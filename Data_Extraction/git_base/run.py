try:
    import traceback
    import os
    import argparse
    import pandas as pd
    import pickle
    from Dict import Dict

    from replication_data import replication_data, component_data, cam_data
except ImportError:
    traceback.print_exc()
    print("Sorry, we didn't remind you to install this module, please use 'pip -install' to install it")

parser = argparse.ArgumentParser()
parser.add_argument('-RQ1_T2', action='store_true')
parser.add_argument('-RQ1_T3', action='store_true')
parser.add_argument('-RQ1_T4', action='store_true')
parser.add_argument('-RQ2_F5', action='store_true')
parser.add_argument('-RQ2_F6', action='store_true')
parser.add_argument('-RQ2_T5', action='store_true')
parser.add_argument('-RQ2_T6', action='store_true')
parser.add_argument('-RQ3_F7', action='store_true')
parser.add_argument('-RQ4_T8', action='store_true')

projects = ['qt', 'openstack', 'jdt', 'platform', 'gerrit', 'go']
afters = ['2015-01-01', '2016-01-01', '2014-01-01', '2016-01-01', '2016-01-01', '2016-01-01']
befores = ['2018-01-01', '2019-01-01', '2017-01-01', '2019-01-01', '2019-01-01', '2019-01-01']
old_afters = ['2011-06-01', '2011-06-01', '2011-06-01', '2011-06-01', '2011-06-01', '2011-06-01']
old_befores = ['2014-03-01', '2014-03-01', '2014-03-01', '2014-03-01', '2014-03-01', '2014-03-01']
sizes = [('10k', 10000), ('20k', 20000), ('30k', 30000), ('40k', 40000), ('50k', 50000)]

gen_deep_cmd = 'python git_extraction.py -get_model_data -project {} -after {} -before {} -save_path datasets/{}/'
gen_deep_cross_cmd = 'python git_extraction.py -get_model_data -project {} -after {} -before {} -save_path datasets/{}/{}/'
gen_k_cmd = 'python extract_k_feature.py -to_csv -project {} -after {} -before {} -save_path datasets/{}/{}/'
gen_deep_size_cmd = 'python git_extraction.py -get_model_data -project {} -before {} -size {} -save_path datasets/{}/{}/'
gen_k_size_cmd = 'python extract_k_feature.py -to_csv -project {} -before {} -save_path datasets/{}/{}/'


def gen_deep_data():
    for project, after, before in zip(projects, afters, befores):
        print(gen_deep_cmd.format(project, after, before, project))
        os.system(gen_deep_cmd.format(project, after, before, project))


def gen_deep_old_data():
    for project, after, before in zip(projects, old_afters, old_befores):
        print(gen_deep_cmd.format(project, after, before, "{}/old".format(project)))
        os.system(gen_deep_cmd.format(project, after, before, "{}/old".format(project)))


def get_deep_rest(datas, project, key):
    train_data = []
    new_msg_dict = Dict(lower=True)
    new_code_dict = Dict(lower=True)

    for p in datas.items():
        if p[0] == project: continue
        data = p[1]

        if key == 'dict':
            msg_dict, code_dict = data[key]
            for word in msg_dict.keys():
                new_msg_dict.add(word)
            for word in code_dict.keys():
                new_code_dict.add(word)
            continue

        if not train_data:
            for idx, elem in enumerate(data[key]):
                train_data.append([])
                train_data[-1].extend(elem)
        else:
            for idx, elem in enumerate(data[key]):
                train_data[idx].extend(elem)
    new_msg_dict = new_msg_dict.prune(100000)
    new_code_dict = new_code_dict.prune(100000)
    if key == 'dict': return [new_msg_dict.get_dict(), new_code_dict.get_dict()]
    return train_data


def gen_deep_cross_data():
    for project, after, before in zip(projects, afters, befores):
        for size in ['cross']:
            print(gen_deep_cross_cmd.format(project, after, before, project, size))
            os.system(gen_deep_cross_cmd.format(project, after, before, project, size))

    datas = {}
    for project in projects:
        deepjit_raw_path = 'datasets/{}/cross/deepjit/{}_all_raw.pkl'.format(project, project)
        deepjit_path = 'datasets/{}/cross/deepjit/{}_all.pkl'.format(project, project)
        cc2vec_dex_path = 'datasets/{}/cross/cc2vec/{}_all_dextend.pkl'.format(project, project)
        cc2vec_path = 'datasets/{}/cross/cc2vec/{}_all.pkl'.format(project, project)
        dict_path = 'datasets/{}/cross/deepjit/{}_dict.pkl'.format(project, project)
        
        datas[project] = {}
        datas[project]['deepjit_raw'] = pickle.load(open(deepjit_raw_path, 'rb'))
        datas[project]['deepjit'] = pickle.load(open(deepjit_path, 'rb'))
        datas[project]['cc2vec_dex'] = pickle.load(open(cc2vec_dex_path, 'rb'))
        datas[project]['cc2vec'] = pickle.load(open(cc2vec_path, 'rb'))
        datas[project]['dict'] = pickle.load(open(dict_path, 'rb'))


    for project in projects:
        deepjit_raw_path = 'datasets/{}/cross/deepjit/{}_train_raw.pkl'.format(project, project)
        deepjit_path = 'datasets/{}/cross/deepjit/{}_train.pkl'.format(project, project)
        cc2vec_dex_path = 'datasets/{}/cross/cc2vec/{}_train_dextend.pkl'.format(project, project)
        cc2vec_path = 'datasets/{}/cross/cc2vec/{}_train.pkl'.format(project, project)
        cc2vec_dict = 'datasets/{}/cross/cc2vec/{}_dict.pkl'.format(project, project)
        deepjit_dict = 'datasets/{}/cross/deepjit/{}_dict.pkl'.format(project, project)


        deepjit_raw = get_deep_rest(datas, project, 'deepjit_raw')
        with open(deepjit_raw_path, 'wb') as f:
            pickle.dump(deepjit_raw, f)

        deepjit = get_deep_rest(datas, project, 'deepjit')
        with open(deepjit_path, 'wb') as f:
            pickle.dump(deepjit, f)

        cc2vec_dex = get_deep_rest(datas, project, 'cc2vec_dex')
        with open(cc2vec_dex_path, 'wb') as f:
            pickle.dump(cc2vec_dex, f)

        cc2vec = get_deep_rest(datas, project, 'cc2vec')
        with open(cc2vec_path, 'wb') as f:
            pickle.dump(cc2vec, f)

        p_dict = get_deep_rest(datas, project, 'dict')
        with open(cc2vec_dict, 'wb') as f:
            pickle.dump(p_dict, f)
        with open(deepjit_dict, 'wb') as f:
            pickle.dump(p_dict, f)


def k_feature():
    for project, after, before in zip(projects, afters, befores):
        print(gen_k_cmd.format(project, after, before, project, ""))
        os.system(gen_k_cmd.format(project, after, before, project, ""))


def old_k_feature():
    for project, after, before in zip(projects, old_afters, old_befores):
        print(gen_k_cmd.format(project, after, before, project, "old"))
        os.system(gen_k_cmd.format(project, after, before, project, "old"))


def split_list(data):
    idx = int(len(data)*0.8)
    return data[:idx], data[idx:]


def get_rest(datas, project):
    train_data = None
    for p in datas.items():
        if p[0] == project: continue
        if train_data is None:
            train_data = p[1]['train']
        else:
            train_data = train_data.append(p[1]['train'], ignore_index=True)
        
        # print(train_data)
    train_data = train_data.sort_values(by=['date'])
    return train_data


def k_feature_cross():
    datas = {}
    for project, after, before in zip(projects, afters, befores):
        for size in ['cross']:
            print(gen_k_cmd.format(project, after, before, project, size))
            os.system(gen_k_cmd.format(project, after, before, project, size))

    for project in projects:
        path = 'datasets/{}/cross/{}_k_feature.csv'.format(project, project)
        p_data = pd.read_csv(path)
        p_train, p_test = split_list(p_data)
        datas[project] = {'train':p_data, 'test':p_test}

    for project in projects:
        train_data = get_rest(datas, project)
        test_data = datas[project]['test']
        train_path = 'datasets/{}/cross/k_train.csv'.format(project)
        test_path = 'datasets/{}/cross/k_test.csv'.format(project)
        train_data.to_csv(train_path,index=0)
        test_data.to_csv(test_path,index=0)


def gen_deep_size():
    for project, before in zip(projects, befores):
        for size in sizes:
            print(gen_deep_size_cmd.format(project, before, size[1], project, size[0]))
            os.system(gen_deep_size_cmd.format(project, before, size[1], project, size[0]))


def gen_k_size():
    for project, before in zip(projects, befores):
        for size in sizes:
            print(gen_k_size_cmd.format(project, before, project, size[0]))
            os.system(gen_k_size_cmd.format(project, before, project, size[0]))


def gen_replication_data():
    replication_data()


def gen_component_data():
    component_data()


def gen_cam_data():
    cam_data()


def gen_original_data():
    try:
        for project in ["qt", "openstack"]:
            os.system("mkdir datasets/{}/original".format(project))
            os.system("cp ../../DeepJIT/data/{}/original/*.pkl datasets/{}/original/".format(project, project))
        gen_component_data()
        gen_replication_data()
        gen_cam_data()
    except:
        traceback.print_exc()
        print("You may not have provided the original dataset of DeepJIT and CC2Vec, sorry we didn't remind you.\nYou can find the original dataset in `data/qt/original/original*.pkl` of the pre-processed dataset and put it under `Data_Extraction/datasets/qt/original/`")


def cp_data():
    os.system("cp -r datasets/* ../../CC2Vec/data")
    os.system("cp -r datasets/* ../../DeepJIT/data")
    os.system("cp -r datasets/* ../../JIT_Baseline/data")


def RQ1_T2():
    try:
        for project in ["qt", "openstack"]:
            os.system("mkdir datasets/{}/original".format(project))
            os.system("cp ../../DeepJIT/data/{}/original/*.pkl datasets/{}/original/".format(project, project))
        gen_replication_data()
    except:
        traceback.print_exc()
        print("You may not have provided the original dataset of DeepJIT and CC2Vec, sorry we didn't remind you.\nYou can find the original dataset in `data/qt/original/original*.pkl` of the pre-processed dataset and put it under `Data_Extraction/datasets/qt/original/`")

    
def RQ1_T3():
    try:
        for project in ["qt", "openstack"]:
            os.system("mkdir -p datasets/{}/original".format(project))
            os.system("cp ../../DeepJIT/data/{}/original/*.pkl datasets/{}/original/".format(project, project))
        gen_cam_data()
    except:
        traceback.print_exc()
        print("You may not have provided the original dataset of DeepJIT and CC2Vec, sorry we didn't remind you.\nYou can find the original dataset in `data/qt/original/original*.pkl` of the pre-processed dataset and put it under `Data_Extraction/datasets/qt/original/`")


def RQ1_T4():
    try:
        for project in ["qt", "openstack"]:
            os.system("mkdir -p datasets/{}/original".format(project))
            os.system("cp ../../DeepJIT/data/{}/original/*.pkl datasets/{}/original/".format(project, project))
        gen_component_data()
    except:
        traceback.print_exc()
        print("You may not have provided the original dataset of DeepJIT and CC2Vec, sorry we didn't remind you.\nYou can find the original dataset in `data/qt/original/original*.pkl` of the pre-processed dataset and put it under `Data_Extraction/datasets/qt/original/`")


def RQ2_F5():
    gen_deep_data()
    k_feature()


def RQ2_F6():
    gen_k_size()
    gen_deep_size()


def RQ2_T5():
    gen_deep_data()
    gen_deep_cross_data()
    k_feature()
    k_feature_cross()


def RQ2_T6():
    gen_deep_old_data()
    old_k_feature()


def RQ3_F7():
    k_feature()
    k_feature_cross()


def RQ4_T8():
    gen_deep_data()
    gen_deep_cross_data()
    k_feature()
    k_feature_cross()


if __name__ == "__main__":
    args = parser.parse_args()
    if args.RQ1_T2:
        RQ1_T2()
    elif args.RQ1_T3:
        RQ1_T3()
    elif args.RQ1_T4:
        RQ1_T4()
    elif args.RQ2_F5:
        RQ2_F5()
    elif args.RQ2_F6:
        RQ2_F6()
    elif args.RQ2_T5:
        RQ2_T5()
    elif args.RQ2_T6:
        RQ2_T6()
    elif args.RQ3_F7:
        RQ3_F7()
    elif args.RQ4_T8:
        RQ4_T8()
    else:
        print("Please select a -RQ.")