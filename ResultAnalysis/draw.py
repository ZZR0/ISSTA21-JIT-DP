import numpy as np
import matplotlib.pyplot as plt

# plt.style.use('classic')
# plt.style.use('ggplot')

def eval(labels, predicts, thresh=0.5):
    TP, FN, FP, TN = 0, 0, 0, 0
    for lable, predict in zip(labels, predicts):
        if predict >= thresh and lable == 1:
            TP += 1
        if predict >= thresh and lable == 0:
            FP += 1
        if predict < thresh and lable == 1:
            FN += 1
        if predict < thresh and lable == 0:
            TN += 1
    
    try:
        P = TP/(TP+FP)
        R = TP/(TP+FN)

        A = (TP+TN)/len(labels)
        E = FP/(TP+FP)

        F1 = 2*R*P/(R+P)

        return R, E, A, F1
    except Exception:
        # division by zero
        return 0, 0, 0, 0

def get_recall_th(result, recall=0.9):
    result.sort(key=lambda x: x[1], reverse=True)
    tn = sum([elem[0] for elem in result])

    pn, th = 0, 1
    for elem in result:
        if elem[0] == 1:
            pn += 1
        
        if pn/tn >= recall:
            th = elem[1]
            return th
    
    return th

def get_result(file):
    with open(file, 'r', encoding='utf-8') as f:
        result = [line.strip('\n').split('\t') for line in f.readlines()]
        result = [[float(elem[0]), float(elem[1])] for elem in result]
        labels = [elem[0] for elem in result]
        predicts = [elem[1] for elem in result]

    return result, labels, predicts

def get_fda_acc(project, methods, result_path = './result/'):
    fda_result = [[method] for method in methods]
    acc_result = [[method] for method in methods]

    for idx, method in enumerate(methods):
        result_file = result_path + '{}/{}_WP.result'.format(method, project)
        for recall in range(1,10,1):
            recall = recall/10

            result, labels, predicts = get_result(result_file)

            th = get_recall_th(result, recall=recall)
            r, e, a, _ = eval(labels, predicts, thresh=th)

            fda_result[idx].append(e)
            acc_result[idx].append(a)

    return fda_result, acc_result

    
def draw_fda(projects, methods, result_path='./RQ2-F5/', save_path='figures/fda.png'):
    colors = ['C0', 'C1', 'C2', 'C3']
    styles = ['-', '-', '-', '-']
    marks = ['o', 'p', 'v', '*']
    bottom, top = 0.4, 0.8

    labels = [recall/10 for recall in range(1,10,1)]
    fda, acc = np.zeros((len(methods), 9)), np.zeros((len(methods), 9))
    for project in projects:
        fda_result, acc_result = get_fda_acc(project, methods, result_path)
        fda_result = np.array(fda_result)
        acc_result = np.array(acc_result)
        fda += np.array(fda_result[:,1:], dtype=float) / len(projects)
        acc += np.array(acc_result[:,1:], dtype=float) / len(projects)
    # plt.style.use('ggplot')
    plt.figure(figsize=(8, 3))
    plt.subplots_adjust(bottom=0.15, left=0.1, right=0.95)
    plt.yticks(np.arange(bottom, top, step=0.05))
    plt.grid(axis="y", ls="--")
    for method, data, c, s, m in zip(methods, fda, colors, styles, marks):
        # data = data * 100
        plt.plot(labels, data, c=c, linestyle=s, marker=m)
        # plt.plot(labels, data, linestyle=s, marker=m)
    
    plt.xlabel('Recall')
    plt.ylabel('False Discovery Rate')
    plt.legend(['DeepJIT','CC2Vec','LR-JIT', 'DBN-JIT'])
    plt.ylim(bottom, top)
    
    plt.savefig(save_path, dpi=1000)
    # plt.show()

def draw_acc(projects, methods, result_path='./RQ2-F5/', save_path='figures/acc.png'):
    colors = ['C0', 'C1', 'C2', 'C3']
    styles = ['-', '-', '-', '-']
    marks = ['o', 'p', 'v', '*']
    bottom, top = 0.4, 0.8

    labels = [recall/10 for recall in range(1,10,1)]
    fda, acc = np.zeros((len(methods), 9)), np.zeros((len(methods), 9))
    for project in projects:
        fda_result, acc_result = get_fda_acc(project, methods, result_path)
        fda_result = np.array(fda_result)
        acc_result = np.array(acc_result)
        fda += np.array(fda_result[:,1:], dtype=float) / len(projects)
        acc += np.array(acc_result[:,1:], dtype=float) / len(projects)

    plt.figure(figsize=(8, 3))
    plt.subplots_adjust(bottom=0.15, left=0.1, right=0.95)
    plt.yticks(np.arange(bottom, top, step=0.05))
    plt.grid(axis="y", ls="--")

    for method, data, c, s, m in zip(methods, acc, colors, styles, marks):
        # data = data * 100
        plt.plot(labels, data, c=c, linestyle=s, marker=m)
        # plt.plot(labels, data, linestyle=s, marker=m)

    
    plt.xlabel('Recall')
    plt.ylabel('Accuracy')
    plt.legend(['DeepJIT','CC2Vec','LR-JIT', 'DBN-JIT'])
    plt.ylim(bottom, top)

    plt.savefig(save_path, dpi=1000)
    

def draw_size(projects, sizes, results, save_path='figures/ts.png'):
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
    styles = ['-', '-', '-', '-', '-', '-']
    marks = ['o', 'p', 's', '*', 'v', 'X']
    bottom, top = 0.4, 0.8

    plt.figure(figsize=(8, 3))
    plt.subplots_adjust(bottom=0.15, left=0.1, right=0.95)
    # plt.yticks(np.arange(bottom, top, step=0.05))
    plt.grid(axis="y", ls="--")

    for project, data, c, s, m in zip(projects, results, colors, styles, marks):
        # data = [v * 100 for v in data]
        plt.plot(sizes[:len(data)], data, c=c, linestyle=s, marker=m)
    
    plt.xlabel('Size')
    plt.ylabel('AUC Score')
    plt.legend(projects)
    plt.savefig(save_path, dpi=1000)


def draw_violinplot(projects, labels, results, save_path='figures/rq3-f7.png'):
    colors = ['C{}'.format(i) for i in range(len(projects))]
    colors[-1] = 'black'
    styles = ['' for _ in range(len(projects))]
    # styles[-1] = '-'
    marks = ['*', 'p', 's', 'X', 'v', 'd', 'o']
    bottom, top = 0.4, 0.8

    plt.figure(figsize=(8, 4))
    plt.subplots_adjust(bottom=0.15, left=0.1, right=0.95)
    plt.violinplot(results, positions=[i for i in range(len(labels))], showmeans=False, showextrema=False)
    plt.yticks(np.arange(bottom, top, step=0.05))
    plt.grid(axis="y", ls="--")

    results = np.transpose(np.array(results)).tolist()

    for project, data, c, s, m in zip(projects, results, colors, styles, marks):
        # data = [v * 100 for v in data]
        ms = 6
        if project == 'Mean':
            ms = 8
        plt.plot(labels, data, c=c, linestyle=s, marker=m, ms=ms)
        # plt.plot(labels, data, linestyle=s, marker=m)

    
    plt.xlabel('Features')
    plt.ylabel('AUC Score')
    plt.legend(projects, loc=1)
    plt.ylim(bottom, top)
    # plt.show()
    plt.savefig(save_path, dpi=1000)

