import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-gid', type=str, default='0')
parser.add_argument('-p', type=str, default='')
parser.add_argument('-raw', action='store_true')
parser.add_argument('-sub', type=str, default='')
parser.add_argument('-epoch', type=str, default='50')

parser.add_argument('-RQ1_T2', action='store_true')
parser.add_argument('-RQ1_T3', action='store_true')
parser.add_argument('-RQ2_F5', action='store_true')
parser.add_argument('-RQ2_F6', action='store_true')
parser.add_argument('-RQ2_T5', action='store_true')
parser.add_argument('-RQ2_T6', action='store_true')
parser.add_argument('-RQ4_T8', action='store_true')

parser.add_argument('-train_deepjit', action='store_true')
parser.add_argument('-pred_deepjit', action='store_true')


raw_task = 'CUDA_VISIBLE_DEVICES={} python main.py -train -train_data data/{}/deepjit/{}_train_raw.pkl -dictionary_data data/{}/deepjit/{}_dict.pkl -save-dir snapshot/{}/raw -num_epochs {}'
task = 'CUDA_VISIBLE_DEVICES={} python main.py -train -train_data data/{}/deepjit/{}_train.pkl -dictionary_data data/{}/deepjit/{}_dict.pkl -save-dir snapshot/{}/model -num_epochs {}'

raw_predict = "CUDA_VISIBLE_DEVICES={} python main.py -predict -pred_data data/{}/deepjit/{}_test_raw.pkl -dictionary_data data/{}/deepjit/{}_dict.pkl -load_model snapshot/{}/raw/epoch_{}.pt"
predict = "CUDA_VISIBLE_DEVICES={} python main.py -predict -pred_data data/{}/deepjit/{}_test.pkl -dictionary_data data/{}/deepjit/{}_dict.pkl -load_model snapshot/{}/model/epoch_{}.pt"


def run_projects(projects, args):
    for project in projects:
        args.p = project
        run_one(args)


def rm_model(args):
    projects = ['qt', 'openstack', 'jdt', 'platform', 'gerrit', 'go']
    for project in projects:
        for i in range(45):
            cmd = "rm snapshot/{}/{}/epoch_{}.pt".format(project, args.sub, i)
            print(cmd)
            os.system(cmd)


def run_one(args):
    project = args.p
    path = args.p
    cmd = ""
    if args.sub:
        path = args.p + '/' + args.sub
    if args.raw:
        if args.train_deepjit:
            cmd = raw_task.format(args.gid, path, project, path, project, path, args.epoch)
        elif args.pred_deepjit:
            cmd = raw_predict.format(args.gid, path, project, path, project, path, args.epoch)
    else:
        if args.train_deepjit:
            cmd = task.format(args.gid, path, project, path, project, path, args.epoch)
        elif args.pred_deepjit:
            cmd = predict.format(args.gid, path, project, path, project, path, args.epoch)
        if args.sub == "cam":
            cmd = cmd + " -cam"
    if not cmd:
        print("Please select a task by: python run.py $RQ $Task")
    print(cmd)
    os.system(cmd)


def RQ1_T2(args):
    projects = ['qt', 'openstack']
    args.epoch = '25'
    for cv in ["cv0", "cv1", "cv2", "cv3", "cv4"]:
        args.sub = cv
        run_projects(projects, args)
    
    args.raw = True
    for cv in ["cv0", "cv1", "cv2", "cv3", "cv4"]:
        args.sub = cv
        run_projects(projects, args)


def RQ1_T3(args):
    projects = ['qt', 'openstack']
    args.sub = "cam"
    run_projects(projects, args)


def RQ2_F5(args):
    projects = ['qt', 'openstack', 'jdt', 'platform', 'gerrit', 'go']
    run_projects(projects, args)


def RQ2_F6(args):
    projects = ['qt', 'openstack', 'jdt', 'platform', 'gerrit', 'go']
    for size in ["10k", "20k", "30k", "40k", "50k"]:
        args.sub = size
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
    elif args.RQ1_T3:
        RQ1_T3(args)
    elif args.RQ2_F5:
        RQ2_F5(args)
    elif args.RQ2_F6:
        RQ2_F6(args)
    elif args.RQ2_T5:
        RQ2_T5(args)
    elif args.RQ2_T6:
        RQ2_T6(args)
    elif args.RQ4_T8:
        RQ4_T8(args)
    else:
        print("Please select a -RQ.")
        # run_one(args)
