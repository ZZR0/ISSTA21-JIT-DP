try:
    import traceback
    from sklearn.metrics import roc_auc_score
    import numpy as np
    import argparse

    from draw import draw_size, draw_violinplot, draw_fda, draw_acc

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


def to_csv(data, path):
    result = ''
    for line in data:
        for word in line:
            result += str(word)
            result += ", "
        result += "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(result)


def get_auc(file_path):
    y_true, y_pred = [], []
    with open(file_path, "r", encoding="utf-8") as f:
        results = f.readlines()
        for result in results:
            result = result.split()
            y_true.append(float(result[-2]))
            y_pred.append(float(result[-1]))
    auc_score = roc_auc_score(y_true=y_true,  y_score=y_pred)
    print("File: {}     AUC_Score: {}".format(file_path, auc_score))
    return auc_score


def RQ1_T2():
    projects = ["qt", "openstack"]
    prefix = ["github", "paper"]
    models = ["deepjit", "cc2vec"]
    table = [["Input", "QT", "Openstack", "Mean"], ["DeepJIT github"], ["DeepJIT paper"], ["CC2Vec github"], ["CC2Vec paper"]]
    for midx, m in enumerate(models):
        for idx, pref in enumerate(prefix):
            line = midx * 2 + idx + 1
            for project in projects:
                auc_score = []
                for cv in ["cv0", "cv1", "cv2", "cv3", "cv4"]:
                    auc_score.append(get_auc("RQ1-T2/{}/{}_{}_{}.result".format(m, project, cv, pref)))
                auc_score = np.mean(auc_score)
                table[line].append(auc_score)
            table[line].append(np.mean(table[line][-2:]))
    to_csv(table, "RQ1-T2/result.csv")
    print("Result save in RQ1-T2/result.csv")


def RQ1_T3():
    with open("RQ1-T3/openstack_cam.csv", "r", encoding="utf-8") as f:
        openstack_cam = f.readlines()
    with open("RQ1-T3/qt_cam.csv", "r", encoding="utf-8") as f:
        qt_cam = f.readlines()

    results = ["qt,qt,openstack,openstack\n"]
    for qt, op in zip(qt_cam, openstack_cam):
        results.append(qt.strip() + ", " + op.strip() + "\n")
    
    with open("RQ1-T3/result.csv", "w", encoding="utf-8") as f:
        f.writelines(results)

    print("Result save in RQ1-T3/result.csv")


def RQ1_T4():
    projects = ["qt", "openstack"]
    only_prefix = ["f", "m", "c", "mc", "fc", "fm"]
    table = [["Input", "QT", "Openstack", "Mean"], ["CC2Vec code"], ["DeepJIT msg"], ["DeepJIT code"], ["-CC2Vec code"], ["-DeepJIT msg"], ["-DeepJIT code"]]
    for idx, pref in enumerate(only_prefix):
        for project in projects:
            auc_score = get_auc("RQ1-T4/{}_{}.result".format(project, pref))
            table[idx+1].append(auc_score)
        table[idx+1].append(np.mean(table[idx+1][-2:]))
    to_csv(table, "RQ1-T4/result.csv")
    print("Result save in RQ1-T4/result.csv")


def RQ2_T5():
    projects = ["qt", "openstack", "jdt", "platform", "gerrit", "go"]
    table = [["Project"], ["Task"], ["DeepJIT"], ["CC2Vec"], ["LR-JIT"], ["DBN-JIT"]]
    tasks = ["WP", "CP"]

    model = ["deepjit", "cc2vec", "lr", "dbn"]
    wp_means, cp_means = [0 for _ in model], [0 for _ in model]

    for project in projects:
        for task in tasks:
            table[0].append(project)
            table[1].append(task)
            for idx, m in enumerate(model):
                auc_score = get_auc("RQ2-T5/{}/{}_{}.result".format(m, project, task))
                table[2+idx].append(auc_score)
                if task == "WP":
                    wp_means[idx] += auc_score
                else:
                    cp_means[idx] += auc_score
    
    table[0].append("Mean")
    table[1].append("WP")
    table[0].append("Mean")
    table[1].append("CP")
    for idx in range(len(model)):
        table[2+idx].append(wp_means[idx]/6)
        table[2+idx].append(cp_means[idx]/6)
    
    to_csv(table, "RQ2-T5/result.csv")
    print("Result save in RQ2-T5/result.csv")


def RQ2_T6():
    projects = ["qt", "openstack", "jdt", "platform", "gerrit", "go"]
    table = [["Project"], ["DeepJIT"], ["CC2Vec"], ["LR-JIT"], ["DBN-JIT"]]

    model = ["deepjit", "cc2vec", "lr", "dbn"]
    wp_means = [0 for _ in model]

    for project in projects:
        table[0].append(project)
        for idx, m in enumerate(model):
            auc_score = get_auc("RQ2-T6/{}/{}.result".format(m, project))
            table[1+idx].append(auc_score)
            wp_means[idx] += auc_score
    
    table[0].append("Mean")
    for idx in range(len(model)):
        table[1+idx].append(wp_means[idx]/6)
    
    to_csv(table, "RQ2-T6/result.csv")
    print("Result save in RQ2-T6/result.csv")


def RQ2_F5():
    projects = ['qt', 'openstack', 'jdt', 'platform', 'gerrit', 'go']
    methods = ['deepjit', 'cc2vec', 'lr', 'dbn']
    draw_fda(projects, methods, result_path='./RQ2-F5/', save_path='./RQ2-F5/fda.png')
    draw_acc(projects, methods, result_path='./RQ2-F5/', save_path='./RQ2-F5/acc.png')
    print("Result save in RQ2-F5/fda.png and RQ2-F5/acc.png")

def RQ2_F6():
    lobels = ['QT', 'OpenStack', 'JDT', 'Platform', 'Gerrit', 'Go']
    projects = ["qt", "openstack", "jdt", "platform", "gerrit", "go"]
    sizes = ['10k', '20k', '30k', '40k', '50k']
    p_size = [5, 5, 1, 3, 3, 5]

    results = [[] for _ in projects]
    for lidx, ps, project in zip(range(len(projects)), p_size, projects):
        for size in sizes:
            auc_score = get_auc("RQ2-F6/deepjit/{}_{}.result".format(project, size))
            if len(results[lidx]) < ps:
                print("Project: {}  Size: {}    AUC: {}".format(project, size, auc_score))
                results[lidx].append(auc_score)
    draw_size(lobels, sizes, results, save_path="RQ2-F6/ts_deepjit.png")

    results = [[] for _ in projects]
    for lidx, ps, project in zip(range(len(projects)), p_size, projects):
        for size in sizes:
            auc_score = get_auc("RQ2-F6/lr/{}_{}.result".format(project, size))
            if len(results[lidx]) < ps:
                print("Project: {}  Size: {}    AUC: {}".format(project, size, auc_score))
                results[lidx].append(auc_score)
    draw_size(lobels, sizes, results, save_path="RQ2-F6/ts_lr.png")
    print("Result save in RQ2-F6/ts_deepjit.png and RQ2-F6/ts_lr.png")


def RQ3_F7():
    projects = ['QT', 'OpenStack', 'JDT', 'Platform', 'Gerrit', 'Go', 'Mean']
    labels = ['All', 'NS', 'ND', 'NF', 'Entrophy', 'LA', 'LD', 'LT', 'FIX', 'NDEV', 'AGE', 'NUC', 'EXP', 'REXP', 'SEXP']
    
    results = [[] for _ in range(len(labels))]
    with open("RQ3-F7/only_result.csv", "r", encoding="utf-8") as f:
        lines = f.readlines()
        for idx, line in enumerate(lines[1:]):
            line = list(map(float, line.split(",")[1:-1]))
            for auc in line:
                print("Project: {}  Feature: {}    AUC: {}".format(projects[len(results[idx])], labels[idx], auc))
                results[idx].append(auc)
            results[idx].append(np.mean(results[idx]))
    draw_violinplot(projects, labels, results, save_path="RQ3-F7/within_only.png")

    results = [[] for _ in range(len(labels))]
    with open("RQ3-F7/cross_only_result.csv", "r", encoding="utf-8") as f:
        lines = f.readlines()
        for idx, line in enumerate(lines[1:]):
            line = list(map(float, line.split(",")[1:-1]))
            for auc in line:
                print("Project: {}  Feature: {}    AUC: {}".format(projects[len(results[idx])], labels[idx], auc))
                results[idx].append(auc)
            results[idx].append(np.mean(results[idx]))
    draw_violinplot(projects, labels, results, save_path="RQ3-F7/cross_only.png")
    print("Result save in RQ3-F7/within_only.png and RQ3-F7/cross_only.png")
        

def RQ4_T8():
    projects = ["qt", "openstack", "jdt", "platform", "gerrit", "go"]
    table = [["Project"], ["Task"], ["DeepJIT"], ["CC2Vec"], ["LR-JIT"], ["DBN-JIT"], ["LApredict"]]
    tasks = ["WP", "CP"]

    model = ["deepjit", "cc2vec", "lr", "dbn", "la"]
    wp_means, cp_means = [0 for _ in model], [0 for _ in model]

    for project in projects:
        for task in tasks:
            table[0].append(project)
            table[1].append(task)
            for idx, m in enumerate(model):
                auc_score = get_auc("RQ4-T8/{}/{}_{}.result".format(m, project, task))
                table[2+idx].append(auc_score)
                if task == "WP":
                    wp_means[idx] += auc_score
                else:
                    cp_means[idx] += auc_score
    
    table[0].append("Mean")
    table[1].append("WP")
    table[0].append("Mean")
    table[1].append("CP")
    for idx in range(len(model)):
        table[2+idx].append(wp_means[idx]/6)
        table[2+idx].append(cp_means[idx]/6)
    
    to_csv(table, "RQ4-T8/result.csv")
    print("Result save in RQ4-T8/result.csv")


if __name__ == "__main__":
    args = parser.parse_args()
    try:
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
    except TypeError:
        traceback.print_exc()
        print("You may be using the wrong version of python, please try python 3.7+")
    except Exception:
        traceback.print_exc()