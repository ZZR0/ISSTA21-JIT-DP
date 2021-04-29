import os
import re

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-RQ2_F5', action='store_true')
parser.add_argument('-RQ2_F6', action='store_true')
parser.add_argument('-RQ2_T5', action='store_true')
parser.add_argument('-RQ2_T6', action='store_true')
parser.add_argument('-RQ3_F7', action='store_true')
parser.add_argument('-RQ4_T8', action='store_true')


size_cmd = "python baseline.py -project {} -algorithm lr -data "
only_cmd = "python baseline.py -data k -project {} -algorithm lr -only "
cross_only_cmd = "python baseline.py -data cross/k -project {} -algorithm lr -only "
old_cmd = "python baseline.py -data old/k -project {} -algorithm "
algo_cmd = "python baseline.py -data k -project {} -algorithm "
algo_cross_cmd = "python baseline.py -data cross/k -project {} -algorithm "
split_data_cmd = "python prepare_data.py -data {} -split_data -project {}"
split_size_data_cmd = "python prepare_data.py -data {} -split_size -project {}"
split_old_data_cmd = "python prepare_data.py -data {} -split_old -project {}"

def run_project(project, features, cmd=algo_cmd):
    results = []
    for key in features:
        result = os.popen(cmd.format(project) + key).readlines()
        results.append((key, result[-1]))
        print(result)
    
    return results

def run_project_size(projects, cmd=algo_cmd):
    results = []
    for project in projects:
        result = os.popen(cmd.format(project)).readlines()
        results.append((project, result[-1]))
        print(result)
    
    return results


def run_all(projects, features, cmd, result_file="./result/"):
    auc_result = [[feature] for feature in features]

    if cmd == only_cmd:
        result_file += "only_result.csv"
    elif cmd == cross_only_cmd:
        result_file += "cross_only_result.csv"
    elif cmd == algo_cmd:
        result_file += "algo_result.csv"
    elif cmd == algo_cross_cmd:
        result_file += "algo_cross_result.csv"
    elif cmd == size_cmd:
        result_file = "size_result.csv"
    else:
        result_file += "algo_result.csv"

    for p_idx, project in enumerate(projects):
        results = run_project(project, features, cmd=cmd)

        pattern = "AUC: (\d+.\d+)"
        for idx, line in enumerate(results):
            key, result = line
            auc = re.findall(pattern, result)
            auc = float(auc[0])
            auc_result[idx].append(auc)

    with open(result_file, "w", encoding="utf-8") as f:
        line = ""
        for project in projects:
            line += ", " + project
        line += "\n"
        f.writelines(line)

        for result in auc_result:
            line = ""
            for auc in result:
                line += str(auc) + ", "
            line += "\n"
            f.writelines(line)

def split_data():
    projects = ["qt", "openstack", "jdt", "platform", "gerrit", "go"]

    for project in projects:
        os.system(split_data_cmd.format("k", project))
        os.system(split_size_data_cmd.format("k", project))
        os.system(split_old_data_cmd.format("k", project))


def run():
    projects = ["qt", "openstack", "jdt", "platform", "gerrit", "go"]
    features = ["lr", "dbn", "la"]
    run_all(projects, features, algo_cmd)


def run_cross():
    projects = ["qt", "openstack", "jdt", "platform", "gerrit", "go"]
    features = ["lr", "dbn", "la"]
    run_all(projects, features, algo_cross_cmd)


def run_old():
    projects = ["qt", "openstack", "jdt", "platform", "gerrit", "go"]
    features = ["lr", "dbn"]
    run_all(projects, features, old_cmd)


def run_only():
    projects = ["qt", "openstack", "jdt", "platform", "gerrit", "go"]
    features = ["all", "ns","nd","nf","entrophy","la","ld","lt","fix","ndev","age","nuc","exp","rexp","sexp"]
    run_all(projects, features, only_cmd)


def run_cross_only():
    projects = ["qt", "openstack", "jdt", "platform", "gerrit", "go"]
    features = ["all", "ns","nd","nf","entrophy","la","ld","lt","fix","ndev","age","nuc","exp","rexp","sexp"]
    run_all(projects, features, cross_only_cmd)


def run_size():
    projects = ["qt", "openstack", "jdt", "platform", "gerrit", "go"]
    features = ["10k/k", "20k/k", "30k/k", "40k/k", "50k/k"]
    run_all(projects, features, size_cmd)


def RQ2_F5():
    run()


def RQ2_F6():
    run_size()


def RQ2_T5():
    run()
    run_cross()


def RQ2_T6():
    run_old()


def RQ3_F7():
    run_only()
    run_cross_only()


def RQ4_T8():
    run()
    run_cross()


def main():
    run()
    # run_cross()
    # run_only()
    # run_cross_only()
    # run_size()
    # run_old()

if __name__ == "__main__":
    split_data()
    args = parser.parse_args()
    if args.RQ2_F5:
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