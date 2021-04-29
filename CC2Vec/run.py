import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-gid', type=str, default='0')
parser.add_argument('-p', type=str, default='')
parser.add_argument('-raw', action='store_true')
parser.add_argument('-sub', type=str, default='')
parser.add_argument('-task', type=str, default='')
parser.add_argument('-epoch', type=str, default='50')
parser.add_argument('-com', action='store_true')

parser.add_argument('-train_cc2vec', action='store_true')
parser.add_argument('-pred_cc2vec', action='store_true')
parser.add_argument('-train_deepjit', action='store_true')
parser.add_argument('-pred_deepjit', action='store_true')

parser.add_argument('-RQ1_T2', action='store_true')
parser.add_argument('-RQ1_T4', action='store_true')
parser.add_argument('-RQ2_F5', action='store_true')
parser.add_argument('-RQ2_T5', action='store_true')
parser.add_argument('-RQ2_T6', action='store_true')
parser.add_argument('-RQ4_T8', action='store_true')


com_task = {"f": "-ftr", "m":"-msg", "c":"-code", "mc":"-msg -code", "fc":"-ftr -code", "fm":"-ftr -msg"}

def rm_model(args):
    projects = ['qt', 'openstack', 'jdt', 'platform', 'gerrit', 'go']
    for project in projects:
        for i in range(45):
            cmd = "rm snapshot/{}/model/{}/epoch_{}.pt".format(project, args.sub, i)
            print(cmd)
            os.system(cmd)


def run_projects(projects, args):
    for project in projects:
        args.p = project
        run_one(args)


def run_one(args):
    cc2ftr_train = 'CUDA_VISIBLE_DEVICES={} python jit_cc2ftr.py -train -train_data data/{}/cc2vec/{}_train.pkl -test_data data/{}/cc2vec/{}_test.pkl -dictionary_data data/{}/cc2vec/{}_dict.pkl -save-dir snapshot/{}/ftr -num_epochs {} '

    cc2ftr_predict = 'CUDA_VISIBLE_DEVICES={} python jit_cc2ftr.py -batch_size 256 -predict -predict_data data/{}/cc2vec/{}_{}.pkl -dictionary_data data/{}/cc2vec/{}_dict.pkl -load_model snapshot/{}/ftr/epoch_{}.pt -name data/{}/cc2vec/{}_{}_cc2ftr.pkl '

    deepjit_train = 'CUDA_VISIBLE_DEVICES={} python jit_DExtended.py -train -train_data data/{}/cc2vec/{}_train_dextend.pkl -train_data_cc2ftr data/{}/cc2vec/{}_train_cc2ftr.pkl -dictionary_data data/{}/cc2vec/{}_dict.pkl -save-dir snapshot/{}/model -num_epochs {} '

    deepjit_predict = "CUDA_VISIBLE_DEVICES={} python jit_DExtended.py -predict -pred_data data/{}/cc2vec/{}_test_dextend.pkl -pred_data_cc2ftr data/{}/cc2vec/{}_test_cc2ftr.pkl -dictionary_data data/{}/cc2vec/{}_dict.pkl -load_model snapshot/{}/model/epoch_{}.pt "

    deepjit_train_raw = 'CUDA_VISIBLE_DEVICES={} python jit_DExtended.py -train -train_data data/{}/cc2vec/{}_train_dextend_raw.pkl -train_data_cc2ftr data/{}/cc2vec/{}_train_cc2ftr.pkl -dictionary_data data/{}/cc2vec/{}_dict.pkl -save-dir snapshot/{}/raw -num_epochs {} '

    deepjit_predict_raw = "CUDA_VISIBLE_DEVICES={} python jit_DExtended.py -predict -pred_data data/{}/cc2vec/{}_test_dextend_raw.pkl -pred_data_cc2ftr data/{}/cc2vec/{}_test_cc2ftr.pkl -dictionary_data data/{}/cc2vec/{}_dict.pkl -load_model snapshot/{}/raw/epoch_{}.pt "

    deepjit_com_train = 'CUDA_VISIBLE_DEVICES={} python jit_DExtended.py -train -train_data data/{}/cc2vec/{}_train_dextend.pkl -train_data_cc2ftr data/{}/cc2vec/{}_train_cc2ftr.pkl -dictionary_data data/{}/cc2vec/{}_dict.pkl -save-dir snapshot/{} -num_epochs {} '

    deepjit_com_predict = "CUDA_VISIBLE_DEVICES={} python jit_DExtended.py -predict -pred_data data/{}/cc2vec/{}_test_dextend.pkl -pred_data_cc2ftr data/{}/cc2vec/{}_test_cc2ftr.pkl -dictionary_data data/{}/cc2vec/{}_dict.pkl -load_model snapshot/{}/epoch_{}.pt "

    project = args.p
    path = args.p
    cmd = ""
    if args.sub:
        path = args.p + '/' + args.sub
    
    model_path = path
    if args.train_cc2vec:
        cmd = cc2ftr_train.format(args.gid, path, project, path, project, path, project, path, args.epoch)
    elif args.pred_cc2vec:
        task = "train"
        cmd = cc2ftr_predict.format(args.gid, path, project, task, path, project, path, args.epoch, path, project, task)
        print(cmd)
        os.system(cmd)
        task = "test"
        cmd = cc2ftr_predict.format(args.gid, path, project, task, path, project, path, args.epoch, path, project, task)
    elif args.train_deepjit:
        if args.raw:
            deepjit_train = deepjit_train_raw
        if args.com:
            deepjit_train = deepjit_com_train + com_task[args.task]
            model_path += "/" + args.task
        cmd = deepjit_train.format(args.gid, path, project, path, project, path, project, model_path, args.epoch)
    elif args.pred_deepjit:
        if args.raw:
            deepjit_predict = deepjit_predict_raw
        if args.com:
            deepjit_train = deepjit_com_predict + com_task[args.task]
            model_path += "/" + args.task
        cmd = deepjit_predict.format(args.gid, path, project, path, project, path, project, model_path, args.epoch)
    if not cmd:
        print("Please select a RQ by: python run.py $RQ $Task")
    print(cmd)
    os.system(cmd)


def RQ1_T2(args):
    projects = ['qt', 'openstack']
    for cv in ["cv0", "cv1", "cv2", "cv3", "cv4"]:
        args.sub = cv
        run_projects(projects, args)
    
    if args.train_deepjit or args.pred_deepjit:
        args.raw = True
        for cv in ["cv0", "cv1", "cv2", "cv3", "cv4"]:
            args.sub = cv
            run_projects(projects, args)


def RQ1_T4(args):
    projects = ['qt', 'openstack']
    args.sub = "original"
    args.com = True
    if args.train_cc2vec or args.pred_cc2vec:
        run_projects(projects, args)
    else:
        for task in com_task:
            args.task = task
            run_projects(projects, args)


def RQ2_F5(args):
    projects = ['qt', 'openstack', 'jdt', 'platform', 'gerrit', 'go']
    run_projects(projects, args)


def RQ2_T5(args):
    projects = ['qt', 'openstack', 'jdt', 'platform', 'gerrit', 'go']
    run_projects(projects, args)
    args.sub = "cross"
    run_projects(projects, args)


def RQ2_T6(args):
    projects = ['qt', 'openstack', 'jdt', 'platform', 'gerrit', 'go']
    args.sub = "old"
    run_projects(projects, args)


def RQ4_T8(args):
    projects = ['qt', 'openstack', 'jdt', 'platform', 'gerrit', 'go']
    run_projects(projects, args)
    args.sub = "cross"
    run_projects(projects, args)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.RQ1_T2:
        RQ1_T2(args)
    elif args.RQ1_T4:
        RQ1_T4(args)
    elif args.RQ2_F5:
        RQ2_F5(args)
    elif args.RQ2_T5:
        RQ2_T5(args)
    elif args.RQ2_T6:
        RQ2_T6(args)
    elif args.RQ4_T8:
        RQ4_T8(args)
    else:
        print("Please select a RQ by: python run.py $RQ $Task")
        # run_one(args)
