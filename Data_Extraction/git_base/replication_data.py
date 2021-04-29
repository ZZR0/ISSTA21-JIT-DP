try:
    import traceback
    import pymongo
    import pickle
    import os
except ImportError:
    traceback.print_exc()
    print("Sorry, we didn't remind you to install this module, please use 'pip -install' to install it")

SAVE_PATH = "./datasets/"

def get_one(db, query):
    for elem in db.find(query):
        return elem
    return None

def split_sentence(sentence):
    sentence = sentence.replace('.', ' . ').replace('_', ' ').replace('@', ' @ ')\
        .replace('-', ' - ').replace('~', ' ~ ').replace('%', ' % ').replace('^', ' ^ ')\
        .replace('&', ' & ').replace('*', ' * ').replace('(', ' ( ').replace(')', ' ) ')\
        .replace('+', ' + ').replace('=', ' = ').replace('{', ' { ').replace('}', ' } ')\
        .replace('|', ' | ').replace('\\', ' \ ').replace('[', ' [ ').replace(']', ' ] ')\
        .replace(':', ' : ').replace(';', ' ; ').replace(',', ' , ').replace('<', ' < ')\
        .replace('>', ' > ').replace('?', ' ? ').replace('/', ' / ')
    sentence = ' '.join(sentence.split())
    return sentence


def create_dir(save_path):
    if not os.path.exists('{}/deepjit'.format(save_path)):
        os.makedirs('{}/deepjit'.format(save_path))
    if not os.path.exists('{}/cc2vec'.format(save_path)):
        os.makedirs('{}/cc2vec'.format(save_path))


def get_data(data, ppcommits_db, time=False):
    data_ids, data_labels, data_msgs, data_codes = data 
    ids, labels, msgs, codes, deepjit_codes, deepjit_raw_codes, history, k_feature = [], [], [], [], [], [], [], []
    dates = []
    for _id, _label, _msg, _code in zip(data_ids, data_labels, data_msgs, data_codes):
        # print(_code)
        commit = get_one(ppcommits_db, {'_id':_id})
        if not commit: continue
        label = _label

        # if label == 0 and random.random() <= 0.7:
        #     continue

        commit_id = commit['commit_id']
        commit_date = commit['commit_date']

        msg = _msg

        format_code = []
        files_code = []
        raw_code = []
        for diff_file in commit['files_diff']:
            diff = commit['files_diff'][diff_file]
            added_code, removed_code, file_codes = [], [], []

            for line in diff['a']:
                # if len(diff['a'][line]['code'].split()) > 3:
                remove_code = diff['a'][line]['code'].strip()
                remove_code = ' '.join(split_sentence(remove_code).split())
                remove_code = ' '.join(remove_code.split(' '))
                # remove_code = 'added _ code removed _ code'
                removed_code.append(remove_code)
                file_codes.append((line, remove_code))
                if len(removed_code) > 10: break

            for line in diff['b']:
                # if len(diff['b'][line]['code'].split()) > 3:
                add_code = diff['b'][line]['code'].strip()
                add_code = ' '.join(split_sentence(add_code).split())
                add_code = ' '.join(add_code.split(' '))
                # add_code = 'added _ code removed _ code'
                added_code.append(add_code)
                file_codes.append((line, add_code))
                if len(added_code) > 10: break

            file_codes.sort(key=lambda x: x[0])
            raw_code.extend([code[1] for code in file_codes])
            raw_code = raw_code[:10]
            format_code.append("added _ code removed _ code")
            files_code.append({'added_code': added_code, 'removed_code': removed_code})
            # shuffle(code)

            if len(format_code) == 10: break


        ids.append(commit_id)
        labels.append(label)
        msgs.append(msg)
        deepjit_codes.append(format_code)
        deepjit_raw_codes.append(raw_code)
        codes.append(files_code)
        dates.append(commit_date)

    deepjit_raw_data = [ids, labels, msgs, deepjit_raw_codes]
    deepjit_data = [ids, labels, msgs, deepjit_codes]
    print('Data size: {}, Bug size: {}'.format(
        len(labels), sum(labels)))
    cc2vec_data = [ids, labels, msgs, codes]
    if time:
        return deepjit_data, deepjit_raw_data, cc2vec_data, dates
    return deepjit_data, deepjit_raw_data, cc2vec_data

def get_project(project):
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    pjdb = client[project]
    ppcommits_db = pjdb['preprocess_commit']
    save_path = "{}/{}/original/".format(SAVE_PATH, project)

    train_pkl = '{}/original_train.pkl'.format(save_path)
    test_pkl = '{}/original_test.pkl'.format(save_path)

    train_data = pickle.load(open(train_pkl, 'rb'))
    test_data = pickle.load(open(test_pkl, 'rb'))

    deepjit_train_data, deepjit_train_raw_data, cc2vec_train_data = get_data(train_data, ppcommits_db)
    deepjit_test_data, deepjit_test_raw_data, cc2vec_test_data = get_data(test_data, ppcommits_db)

    dextend_train_data = deepjit_train_raw_data
    dextend_test_data = deepjit_test_raw_data

    create_dir(save_path)

    with open('{}/deepjit/{}_train.pkl'.format(save_path, project), 'wb') as f:
        pickle.dump(deepjit_train_data, f)
    with open('{}/deepjit/{}_test.pkl'.format(save_path, project), 'wb') as f:
        pickle.dump(deepjit_test_data, f)

    with open('{}/deepjit/{}_train_raw.pkl'.format(save_path, project), 'wb') as f:
        pickle.dump(deepjit_train_raw_data, f)
    with open('{}/deepjit/{}_test_raw.pkl'.format(save_path, project), 'wb') as f:
        pickle.dump(deepjit_test_raw_data, f)

    with open('{}/cc2vec/{}_train.pkl'.format(save_path, project), 'wb') as f:
        pickle.dump(cc2vec_train_data, f)
    with open('{}/cc2vec/{}_test.pkl'.format(save_path, project), 'wb') as f:
        pickle.dump(cc2vec_test_data, f)

    with open('{}/cc2vec/{}_train_dextend.pkl'.format(save_path, project), 'wb') as f:
        pickle.dump(dextend_train_data, f)
    with open('{}/cc2vec/{}_test_dextend.pkl'.format(save_path, project), 'wb') as f:
        pickle.dump(dextend_test_data, f)

    os.system('\cp {}/{}_dict.pkl {}/deepjit/'.format(save_path, project, save_path))
    os.system('\cp {}/{}_dict.pkl {}/cc2vec/'.format(save_path, project, save_path))

def get_cv(all_data, i):
    train_data = [[] for _ in range(len(all_data[0]))]
    test_data = None
    for idx, split_data in enumerate(all_data):
        if idx == i: 
            test_data = split_data
        else:
            for jdx, data in enumerate(split_data):
                train_data[jdx].extend(data)
    
    return train_data, test_data

def get_project_cv(project):
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    pjdb = client[project]
    ppcommits_db = pjdb['preprocess_commit']
    save_path = "{}/{}/original/".format(SAVE_PATH, project)

    train_pkl = '{}/original_train.pkl'.format(save_path)
    test_pkl = '{}/original_test.pkl'.format(save_path)

    train_data = pickle.load(open(train_pkl, 'rb'))
    test_data = pickle.load(open(test_pkl, 'rb'))

    data = []
    for train, test in zip(train_data, test_data):
        train.extend(test)
        data.append(train)

    all_data = []
    size = int(len(data[0])/5)
    for i in range(5):
        start = i*size
        end = start+size
        split_data = []
        for elem in data:
            split_data.append(elem[start:end])
        all_data.append(split_data)
    
    for i in range(5):
        train, test = get_cv(all_data, i)
            
        deepjit_train_data, deepjit_train_raw_data, cc2vec_train_data = get_data(train, ppcommits_db)
        deepjit_test_data, deepjit_test_raw_data, cc2vec_test_data = get_data(test, ppcommits_db)

        save_path = "{}/{}/{}/".format(SAVE_PATH, project, 'cv{}'.format(i))
        create_dir(save_path)
        with open('{}/deepjit/{}_train.pkl'.format(save_path, project), 'wb') as f:
            pickle.dump(deepjit_train_data, f)
        with open('{}/deepjit/{}_test.pkl'.format(save_path, project), 'wb') as f:
            pickle.dump(deepjit_test_data, f)

        with open('{}/deepjit/{}_train_raw.pkl'.format(save_path, project), 'wb') as f:
            pickle.dump(deepjit_train_raw_data, f)
        with open('{}/deepjit/{}_test_raw.pkl'.format(save_path, project), 'wb') as f:
            pickle.dump(deepjit_test_raw_data, f)

        with open('{}/cc2vec/{}_train.pkl'.format(save_path, project), 'wb') as f:
            pickle.dump(cc2vec_train_data, f)
        with open('{}/cc2vec/{}_test.pkl'.format(save_path, project), 'wb') as f:
            pickle.dump(cc2vec_test_data, f)

        with open('{}/cc2vec/{}_train_dextend.pkl'.format(save_path, project), 'wb') as f:
            pickle.dump(deepjit_train_data, f)
        with open('{}/cc2vec/{}_test_dextend.pkl'.format(save_path, project), 'wb') as f:
            pickle.dump(deepjit_test_data, f)

        with open('{}/cc2vec/{}_train_dextend_raw.pkl'.format(save_path, project), 'wb') as f:
            pickle.dump(deepjit_train_raw_data, f)
        with open('{}/cc2vec/{}_test_dextend_raw.pkl'.format(save_path, project), 'wb') as f:
            pickle.dump(deepjit_test_raw_data, f)

        os.system('\cp {}/{}/original/{}_dict.pkl {}/deepjit/'.format(SAVE_PATH, project, project, save_path))
        os.system('\cp {}/{}/original/{}_dict.pkl {}/cc2vec/'.format(SAVE_PATH, project, project, save_path))


def split_data(data):
    idx = int(len(data[0]) * 0.8)
    train, test = [], []
    for elem in data:
        train.append(elem[:idx])
        test.append(elem[idx:])

    return train, test


def replication_data():
    for project in ['qt', 'openstack']:
        get_project_cv(project)


def component_data():
    for project in ['qt', 'openstack']:
        get_project(project)

def cam_data():
    for project in ['qt', 'openstack']:
        os.system("mkdir -p {}/{}/cam/deepjit".format(SAVE_PATH, project))
        os.system("cp {}/{}/original/original_train.pkl {}/{}/cam/deepjit/{}_train.pkl".format(SAVE_PATH, project, project, SAVE_PATH, project, project))
        os.system("cp {}/{}/original/original_test.pkl {}/{}/cam/deepjit/{}_test.pkl".format(SAVE_PATH, project, project, SAVE_PATH, project, project))
        os.system("cp {}/{}/original/{}_dict.pkl {}/{}/cam/deepjit/{}_dict.pkl".format(SAVE_PATH, project, project, SAVE_PATH, project, project))


if __name__ == "__main__":
    component_data()
    replication_data()
    cam_data()