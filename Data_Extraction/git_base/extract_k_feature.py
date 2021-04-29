import sys
sys.path.append(sys.path[0]+"/..")
import re
import os
import math
import random
import datetime

import argparse
import pymongo
from tqdm import tqdm
import numpy as np
import pandas as pd

from utils.utils import get_one, get_one_dict, conv_to_dict, dict_to_db, init_args


parser = argparse.ArgumentParser()
parser.add_argument('-init_project', action='store_true', default=False)
parser.add_argument('-extract_project_feature', action='store_true', default=False)
parser.add_argument('-analyse_project', action='store_true', default=False)
parser.add_argument('-to_csv', action='store_true', default=False)

parser.add_argument('-project', type=str,
                    default='qt')
parser.add_argument('-after', type=str,
                    default='2011-01-01')
parser.add_argument('-before', type=str,
                    default='2020-12-01')

parser.add_argument('-preprocess_commit', type=str,
                    default='preprocess_commit')
parser.add_argument('-commit', type=str,
                    default='commit')
parser.add_argument('-issues', type=str,
                    default='issues')
parser.add_argument('-diff', type=str,
                    default='diff')
parser.add_argument('-blame', type=str,
                    default='blame')
parser.add_argument('-file_bugs', type=str,
                    default='file_bugs')

parser.add_argument('-save_path', type=str,
                    default='')


random.seed(97818)
client = pymongo.MongoClient("mongodb://localhost:27017/")

class Metrics(object):
    def __init__(self, _id):
        self._id = _id
        self.ns = 0                                      # Number of modified subsystems
        self.nd = 0                                      # number of modified directories
        self.nf = 0                                      # Number of modified files
        self.entrophy = 0                                # entrophy: distriubtion of modified code across each file
        self.la = 0                                      # lines added
        self.ld = 0                                      # lines deleted
        self.lt = 0                                      # lines of code in each file (sum) before the commit
        self.fix = 0
        self.ndev = 0                                    # the number of developers that modifed the files in a commit
        self.age = 0                                     # the average time interval between the last and current change
        self.nuc = 0                                     # number of unique changes to the files
        self.exp = 0                                     # number of changes made by author previously
        self.rexp = 0                                    # experience weighted by age of files ( 1 / (n + 1))
        self.sexp = 0                                    # changes made previous by author in same subsystem
        self.date = 0                                    # changes made previous by author in same subsystem


    def to_dict(self):
        """
        docstring
        """
        return {'_id': self._id,
                'date': self.date,
                'ns': self.ns,
                'nd': self.nd,
                'nf': self.nf,
                'entrophy': self.entrophy,
                'la': self.la,
                'ld': self.ld,
                'lt': self.lt,
                'fix': self.fix,
                'ndev': self.ndev,
                'age': self.age,
                'nuc': self.nuc,
                'exp': self.exp,
                'rexp': self.rexp,
                'sexp': self.sexp}


def calu_modified_lines(file):
    add_line, del_line = 0, 0
    t_line = file['meta_a']['lines'] if 'meta_a' in file else 0
    for ab in file['content']:
        if 'a' in ab:
            del_line += len(ab['a'])
        if 'b' in ab:
            add_line += len(ab['b'])
    
    return add_line, del_line, t_line


def calu_entrophy(totalLOCModified, locModifiedPerFile):
    # Update entrophy
    entrophy = 0
    for fileLocMod in locModifiedPerFile:
        if (fileLocMod != 0 ):
            avg = fileLocMod/totalLOCModified
            entrophy -= ( avg * math.log( avg,2 ) )

    return entrophy


def check_fix(msg, patten):
    issue_keys = re.findall(patten, msg)
    if issue_keys:
        return 1
    return 0


def get_author_exp(author_exp):
    exp = 0
    for file in list(author_exp.items())[1:]:
        exp += len(file[1])
    return exp


def get_author_rexp(author_exp, now):
    rexp = 0
    for file in list(author_exp.items())[1:]:
        for t in file[1]:
            age = (now - t) / 86400
            age = max(age, 0)
            rexp += 1 / (age + 1)
    return rexp


def get_author_sexp(author_exp, subsystems):
    sexp = 0
    for file in author_exp.items():
        file_path = file[0]
        sub, _, _ = get_subs_dire_name(file_path)
        if sub in subsystems:
            sexp += 1
    return sexp


def get_subs_dire_name(fileDirs):
    fileDirs = fileDirs.split("/")
    if( len(fileDirs) == 1 ):
        subsystem = "root"
        directory = "root"
    else:
        subsystem = fileDirs[0]
        directory = "/".join(fileDirs[0:-1])
    file_name = fileDirs[-1]
    
    return subsystem, directory, file_name


def extract_commit_feature(args, commit, diff, blame, file_dict, author_dict):
    metrics = Metrics(commit['_id'])
    subs, dirs, files, authors, ages = [], [], [], [], []
    totalLOCModified = 0                        # Total modified LOC across all files
    locModifiedPerFile = []                     # List of modified loc in each file seen
    fix = commit['fix']
    nuc = 0
    author_name = commit['author']
    now = commit['commit_date']

    author_exp = get_one_dict(author_dict, author_name)
    if not author_exp:
        author_exp = {'_id': author_name}

    for file_elem in list(diff.items())[1:]:
        file_path = file_elem[0]
        val = file_elem[1]

        subsystem, directory, filename = get_subs_dire_name(file_path)

        if( subsystem not in subs ):
            subs.append( subsystem )

        if( directory not in dirs ):
            dirs.append( directory )

        if( filename not in files ):
            files.append( filename )

        la, ld, lt = calu_modified_lines(val)

        metrics.la += la
        metrics.ld += ld
        metrics.lt += lt

        totalLOCModified += la + ld
        locModifiedPerFile.append(la + ld)
        
        file = get_one_dict(file_dict, file_path)
        if not file:
            file = {'_id': file_path, 'author': [], 'nuc': 0}
        author = file['author']
        if author_name not in author:
            author.append(author_name)
        authors = list(set(authors) | set(author))

        prev_time = get_prev_time(blame, file_path)
        age = now - prev_time if prev_time else 0
        age = max(age, 0)
        ages.append(age)

        
        file_nuc = file['nuc'] + 1
        nuc += file_nuc

        file['author'] = author
        file['nuc'] = file_nuc
        file_dict[file['_id']] = file

        if file_path in author_exp:
            author_exp[file_path].append(now)
        else:
            author_exp[file_path] = [now]

    author_dict[author_exp['_id']] = author_exp

    metrics.date = now
    metrics.ns = len(subs)
    metrics.nd = len(dirs)
    metrics.nf = len(files)
    metrics.entrophy = calu_entrophy(totalLOCModified, locModifiedPerFile)
    metrics.fix = fix
    metrics.ndev = len(authors)
    metrics.age = np.mean(ages) / 86400 if ages else 0
    metrics.nuc = nuc
    metrics.exp = get_author_exp(author_exp)
    metrics.rexp = get_author_rexp(author_exp, now)
    metrics.sexp = get_author_sexp(author_exp, subs)
    
    return metrics.to_dict()


def extract_project_feature(args):
    pjdb = client[args.project]
    ppcommit_db = pjdb[args.preprocess_commit]
    feat_db = pjdb['k_feature']
    diff_db = pjdb[args.diff]
    blame_db = pjdb[args.blame]
    author_db = pjdb['author']
    file_db = pjdb['file']

    file_dict = conv_to_dict(file_db)
    author_dict = conv_to_dict(author_db)

    feat_db.drop()

    after = datetime.datetime.strptime(args.after,"%Y-%m-%d").timestamp()
    before = datetime.datetime.strptime(args.before,"%Y-%m-%d").timestamp()
    
    for commit in tqdm(list(ppcommit_db.find({"commit_date": {'$gte': after, '$lte': before}}))):
        diff = get_one(diff_db, {'_id': commit['_id']})
        blame = get_one(blame_db, {'_id': commit['_id']})
        if not blame or not diff: continue

        change_feature = extract_commit_feature(args, commit, diff, blame, file_dict, author_dict)

        try:
            result = feat_db.update(
                {'_id': change_feature['_id']}, change_feature, check_keys=False, upsert=True)
        except:
            # DocumentTooLarge
            print('Document Too Large Error.')


def get_prev_time(blame, file):
    if not file in blame: return 0

    max_time = 0
    for elem in blame[file]['id2line'].items():
        elem = elem[1]
        max_time = max(elem['time'], max_time)
    return max_time


def find_file_author(blame, file_path):
    if not file_path in blame: return [], []
    author = set()
    commit = set()
    file_blame = blame[file_path]['id2line']
    for elem in file_blame:
        name = file_blame[elem]['author']
        commit.add(file_blame[elem]['id'])
        author.add(name)
    return list(commit), list(author)


def map_file_to_author(blam, file_path, author_dict):
    if not file_path in blam: return
    file_blame = blam[file_path]['id2line']
    for elem in file_blame.items():
        elem = elem[1]
        author_name = elem['author']
        time = elem['time']

        author = get_one_dict(author_dict, author_name)
        if not author:
            author = {'_id': author_name}

        if file_path in author:
            if not time in author[file_path]:
                author[file_path].append(time)
        else:
            author[file_path] = [time]

        author_dict[author['_id']] = author
        

def init_project(args):
    pjdb = client[args.project]
    ppcommit_db = pjdb[args.preprocess_commit]
    file_db = pjdb['file']
    author_db = pjdb['author']

    file_db.drop()
    author_db.drop()
    
    blame_db = pjdb[args.blame]

    file_dict = conv_to_dict(file_db)
    author_dict = conv_to_dict(author_db)

    after = datetime.datetime.strptime(args.after,"%Y-%m-%d").timestamp()
    before = datetime.datetime.strptime(args.before,"%Y-%m-%d").timestamp()

    ppcommit_db.create_index('commit_date')
    for commit in tqdm(list(ppcommit_db.find({"commit_date": {'$gte': after, '$lte': before}}).sort([('commit_date',1)]))):
        blame = get_one(blame_db, {'_id': commit['_id']})
        
        files = commit['files']

        for file_path in files:
            file = get_one_dict(file_dict, file_path)

            if file:
                continue

            file = {'_id': file_path, 'author': [], 'nuc': 0}

            commit_list, author_list = find_file_author(blame, file_path)

            file['author'] = author_list
            file['nuc'] = len(commit_list)

            file_dict[file['_id']] = file

            map_file_to_author(blame, file_path, author_dict)
    
    dict_to_db(file_db, file_dict)
    dict_to_db(author_db, author_dict)

     
def to_csv(args, save=True):
    pjdb = client[args.project]
    ppcommit_db = pjdb[args.preprocess_commit]

    feat_db = pjdb['k_feature']

    _id, ns, nd, nf, entrophy, la, ld, lt, fix, ndev, age, nuc, exp, rexp, sexp, bug = \
        [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    date = []

    after = datetime.datetime.strptime(args.after,"%Y-%m-%d").timestamp()
    before = datetime.datetime.strptime(args.before,"%Y-%m-%d").timestamp()

    feat_db.create_index('date')
    for feat in feat_db.find({"date": {'$gte': after, '$lte': before}}).sort([('date',1)]):
        ppc = get_one(ppcommit_db, {'_id': feat['_id']})
        if not ppc: continue
        if ppc['median_issue'] == 1: continue
        _id.append(feat['_id'])
        date.append(feat['date'])
        ns.append(feat['ns'])
        nd.append(feat['nd'])
        nf.append(feat['nf'])
        entrophy.append(feat['entrophy'])
        la.append(feat['la'])
        ld.append(feat['ld'])
        lt.append(feat['lt'])
        fix.append(feat['fix'])
        ndev.append(feat['ndev'])
        age.append(feat['age'])
        nuc.append(feat['nuc'])
        exp.append(feat['exp'])
        rexp.append(feat['rexp'])
        sexp.append(feat['sexp'])
        bc = ppc['bug_count']
        bug.append(1 if bc > 0 else 0)

    data = {'_id': _id,
            'date': date,
            'bug': bug,
            '__': bug,
            'ns': ns,
            'nd': nd,
            'nf': nf,
            'entrophy': entrophy,
            'la': la,
            'ld': ld,
            'lt': lt,
            'fix': fix,
            'ndev': ndev,
            'age': age,
            'nuc': nuc,
            'exp': exp,
            'rexp': rexp,
            'sexp': sexp}

    if save:
        # print(args.save_path)
        # args.save_path = 'datasets/{}/'+args.save_path+'/' if not args.save_path == '' else 'datasets/{}/'
        if not os.path.exists(args.save_path.format(args.project)):
            os.makedirs(args.save_path.format(args.project))
        # print(args.save_path)
        save_to = args.save_path.format(args.project) + '/{}_k_feature.csv'.format(args.project)
        print('Save to ', save_to)
        pd.DataFrame(data).to_csv(save_to)
    return data


if __name__ == "__main__":
    args = parser.parse_args()
    init_args(args)

    if len(sys.argv) < 2:
        print('Usage: python gettit_extraction.py [model]')
    elif args.init_project:
        init_project(args)
    elif args.extract_project_feature:
        extract_project_feature(args)
    elif args.analyse_project:
        init_project(args)
        extract_project_feature(args)
    elif args.to_csv:
        to_csv(args)