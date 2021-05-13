import os
import argparse

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

def RQ1_T2():
    cmd = "cp ../DeepJIT/snapshot/{}/{}/model/epoch_25.pt.result ./RQ1-T2/deepjit/{}_{}_github.result"
    
    for project in ["qt", "openstack"]:
        for cv in ["cv0", "cv1", "cv2", "cv3", "cv4"]:
            print(cmd.format(project, cv, project, cv))
            os.system(cmd.format(project, cv, project, cv))
    
    cmd = "cp ../DeepJIT/snapshot/{}/{}/raw/epoch_25.pt.result ./RQ1-T2/deepjit/{}_{}_paper.result"
    
    for project in ["qt", "openstack"]:
        for cv in ["cv0", "cv1", "cv2", "cv3", "cv4"]:
            print(cmd.format(project, cv, project, cv))
            os.system(cmd.format(project, cv, project, cv))

    cmd = "cp ../CC2Vec/snapshot/{}/{}/model/epoch_50.pt.result ./RQ1-T2/cc2vec/{}_{}_github.result"
    for project in ["qt", "openstack"]:
        for cv in ["cv0", "cv1", "cv2", "cv3", "cv4"]:
            print(cmd.format(project, cv, project, cv))
            os.system(cmd.format(project, cv, project, cv))

    cmd = "cp ../CC2Vec/snapshot/{}/{}/raw/epoch_50.pt.result ./RQ1-T2/cc2vec/{}_{}_paper.result"
    for project in ["qt", "openstack"]:
        for cv in ["cv0", "cv1", "cv2", "cv3", "cv4"]:
            print(cmd.format(project, cv, project, cv))
            os.system(cmd.format(project, cv, project, cv))


def RQ1_T3():
    cmd = "cp ../DeepJIT/snapshot/{}/cam/model/epoch_50.pt.cam.csv RQ1-T3/{}_cam.csv"

    for project in ["qt", "openstack"]:
            print(cmd.format(project, project))
            os.system(cmd.format(project, project))


def RQ1_T4():
    cmd = "cp ../CC2Vec/snapshot/{}/original/{}/epoch_50.pt.result ./RQ1-T4/{}_{}.result"

    for project in ["qt", "openstack"]:
        for com in ["f", "m", "c", "fc", "fm", "mc"]:
            print(cmd.format(project, com, project, com))
            os.system(cmd.format(project, com, project, com))


def RQ2_F5():
    cmd_wp = "cp ../JIT_Baseline/result/{}/{}_{}_k.result ./RQ2-F5/{}/{}_WP.result"

    for project in projects:
        for model in ["lr", "dbn"]:
            print(cmd_wp.format(project, project, model, model, project))
            os.system(cmd_wp.format(project, project, model, model, project))

    cmd_wp = "cp ../CC2Vec/snapshot/{}/model/epoch_50.pt.result ./RQ2-F5/{}/{}_WP.result"

    for project in projects:
        print(cmd_wp.format(project, "cc2vec", project))
        os.system(cmd_wp.format(project, "cc2vec", project))
    
    cmd_wp = "cp ../DeepJIT/snapshot/{}/model/epoch_50.pt.result ./RQ2-F5/{}/{}_WP.result"

    for project in projects:
        print(cmd_wp.format(project, "deepjit", project))
        os.system(cmd_wp.format(project, "deepjit", project))


def RQ2_F6():
    cmd = "cp ../JIT_Baseline/result/{}/{}_{}_{}_k.result ./RQ2-F6/{}/{}_{}.result"

    for project in projects:
        for size in ["10k", "20k", "30k", "40k", "50k"]:
            model = "lr"
            print(cmd.format(project, project, model, size, model, project, size))
            os.system(cmd.format(project, project, model, size, model, project, size))
    
    cmd = "cp ../DeepJIT/snapshot/{}/{}/model/epoch_50.pt.result ./RQ2-F6/{}/{}_{}.result"

    for project in projects:
        for size in ["10k", "20k", "30k", "40k", "50k"]:
            model = "deepjit"
            print(cmd.format(project, size, model, project, size))
            os.system(cmd.format(project, size, model, project, size))


def RQ2_T5():
    cmd_wp = "cp ../JIT_Baseline/result/{}/{}_{}_k.result ./RQ2-T5/{}/{}_WP.result"
    cmd_cp = "cp ../JIT_Baseline/result/{}/{}_{}_cross_k.result ./RQ2-T5/{}/{}_CP.result"

    for project in projects:
        for model in ["lr", "dbn"]:
            print(cmd_wp.format(project, project, model, model, project))
            os.system(cmd_wp.format(project, project, model, model, project))
            print(cmd_cp.format(project, project, model, model, project))
            os.system(cmd_cp.format(project, project, model, model, project))

    cmd_wp = "cp ../CC2Vec/snapshot/{}/model/epoch_50.pt.result ./RQ2-T5/{}/{}_WP.result"
    cmd_cp = "cp ../CC2Vec/snapshot/{}/cross/model/epoch_50.pt.result ./RQ2-T5/{}/{}_CP.result"

    for project in projects:
        for model in ["cc2vec"]:
            print(cmd_wp.format(project, model, project))
            os.system(cmd_wp.format(project, model, project))
            print(cmd_cp.format(project, model, project))
            os.system(cmd_cp.format(project, model, project))

    
    cmd_wp = "cp ../DeepJIT/snapshot/{}/model/epoch_50.pt.result ./RQ2-T5/{}/{}_WP.result"
    cmd_cp = "cp ../DeepJIT/snapshot/{}/cross/model/epoch_50.pt.result ./RQ2-T5/{}/{}_CP.result"

    for project in projects:
        for model in ["deepjit"]:
            print(cmd_wp.format(project, model, project))
            os.system(cmd_wp.format(project, model, project))
            print(cmd_cp.format(project, model, project))
            os.system(cmd_cp.format(project, model, project))


def RQ2_T6():
    cmd = "cp ../JIT_Baseline/result/{}/{}_{}_old_k.result ./RQ2-T6/{}/{}.result"

    for project in projects:
        for model in ["lr", "dbn"]:
            print(cmd.format(project, project, model, model, project))
            os.system(cmd.format(project, project, model, model, project))

    cmd_wp = "cp ../CC2Vec/snapshot/{}/old/model/epoch_50.pt.result ./RQ2-T6/{}/{}.result"

    for project in projects:
        for model in ["cc2vec"]:
            print(cmd_wp.format(project, model, project))
            os.system(cmd_wp.format(project, model, project))

    
    cmd_wp = "cp ../DeepJIT/snapshot/{}/old/model/epoch_50.pt.result ./RQ2-T6/{}/{}.result"

    for project in projects:
        for model in ["deepjit"]:
            print(cmd_wp.format(project, model, project))
            os.system(cmd_wp.format(project, model, project))


def RQ3_F7():
    cmd = "cp ../JIT_Baseline/result/*only* ./RQ3-F7/"

    print(cmd)
    os.system(cmd)


def RQ4_T8():
    cmd_wp = "cp ../JIT_Baseline/result/{}/{}_{}_k.result ./RQ4-T8/{}/{}_WP.result"
    cmd_cp = "cp ../JIT_Baseline/result/{}/{}_{}_cross_k.result ./RQ4-T8/{}/{}_CP.result"

    for project in projects:
        for model in ["lr", "dbn", "la"]:
            print(cmd_wp.format(project, project, model, model, project))
            os.system(cmd_wp.format(project, project, model, model, project))
            print(cmd_cp.format(project, project, model, model, project))
            os.system(cmd_cp.format(project, project, model, model, project))

    cmd_wp = "cp ../CC2Vec/snapshot/{}/model/epoch_50.pt.result ./RQ4-T8/{}/{}_WP.result"
    cmd_cp = "cp ../CC2Vec/snapshot/{}/cross/model/epoch_50.pt.result ./RQ4-T8/{}/{}_CP.result"

    for project in projects:
        for model in ["cc2vec"]:
            print(cmd_wp.format(project, model, project))
            os.system(cmd_wp.format(project, model, project))
            print(cmd_cp.format(project, model, project))
            os.system(cmd_cp.format(project, model, project))

    
    cmd_wp = "cp ../DeepJIT/snapshot/{}/model/epoch_50.pt.result ./RQ4-T8/{}/{}_WP.result"
    cmd_cp = "cp ../DeepJIT/snapshot/{}/cross/model/epoch_50.pt.result ./RQ4-T8/{}/{}_CP.result"

    for project in projects:
        for model in ["deepjit"]:
            print(cmd_wp.format(project, model, project))
            os.system(cmd_wp.format(project, model, project))
            print(cmd_cp.format(project, model, project))
            os.system(cmd_cp.format(project, model, project))


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
