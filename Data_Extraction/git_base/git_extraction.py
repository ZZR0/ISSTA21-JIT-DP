import sys
sys.path.append(sys.path[0]+"/..")
import os
import random
import pickle
import datetime
import re

import argparse
from tqdm import tqdm
import numpy as np
import pymongo

from utils.utils import init_args, get_one_dict, get_one, dict_to_db, split_sentence, conv_to_dict
from git_handler import get_repo, get_commits
from issues_handler import get_issues
from Dict import Dict


parser = argparse.ArgumentParser()
parser.add_argument('-get_issues', action='store_true', default=False)
parser.add_argument('-get_repo', action='store_true', default=False)
parser.add_argument('-get_commits', action='store_true', default=False)
parser.add_argument('-preprocess_commits', action='store_true', default=False)
parser.add_argument('-locating_fix_inducing', action='store_true', default=False)
parser.add_argument('-get_model_data', action='store_true', default=False)
parser.add_argument('-bug_analyse', action='store_true', default=False)

parser.add_argument('-drop', action='store_true', default=False)
parser.add_argument('-skip', action='store_true', default=False)
parser.add_argument('-year', type=int, default=2019)
parser.add_argument('-size', type=int, default=0)

parser.add_argument('-project', type=str,
                    default='qt')
parser.add_argument('-repo', type=str,
                    default='')
parser.add_argument('-url', type=str,
                    default='')
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
parser.add_argument('-train', type=int, default=10000)
parser.add_argument('-test', type=int, default=2000)

# parser.add_argument('-random_seed', type=int, default=97818)
random.seed(97818)


client = pymongo.MongoClient("mongodb://localhost:27017/")


def get_blame_id(blame_dict, file, a_line):
    blame_commit_id = ''
    if not blame_dict or file not in blame_dict:
        return -1, blame_commit_id
    blame_line = blame_dict[file]['line2id'][str(a_line)]
    blame_commit_id = blame_line['blame_id']
    return blame_line['blame_line_a'], blame_commit_id


def get_files_diff(diff, blame):
    files_diff = dict()
    for file in diff:
        if file == '_id': continue
        diff_file = diff[file]

        new_diff = {'a': dict(), 'b': dict()}
        a_line, b_line = 0, 0
        for ab in diff_file['content']:
            if 'ab' in ab:
                a_line += len(ab['ab'])
                b_line += len(ab['ab'])
            if 'a' in ab:
                for line in ab['a']:
                    a_line += 1
                    prev_line, blame_commit_id = get_blame_id(blame, file, a_line)
                    new_diff['a'][str(a_line)] = {'code': line,
                                              'blame_commit': blame_commit_id,
                                              'prev_line': prev_line}
            if 'b' in ab:
                for line in ab['b']:
                    b_line += 1
                    new_diff['b'][str(b_line)] = {'code': line, 'blame_commit': ''}
        files_diff[file] = new_diff
    return files_diff


def clean_commit(args, commit, diff, blame):
    pjdb = client[args.project]
    issues_db = pjdb[args.issues]

    pp_commit = dict()

    pp_commit['_id'] = commit['_id']
    pp_commit['commit_id'] = commit['commit_id']
    pp_commit['parent_id'] = commit['parent_id']
    pp_commit['subject'] = commit['subject']
    pp_commit['commit_msg'] = commit['commit_msg']
    pp_commit['author'] = commit['author']
    pp_commit['commit_date'] = commit['commit_date']
    pp_commit['files'] = commit['files']
    pp_commit['repo'] = commit['repo']

    pp_commit['fix'] = 0
    pp_commit['fix_issue'] = ''
    pp_commit['bug_count'] = 0
    pp_commit['fix_id'] = []
    pp_commit['fix_location'] = {}
    pp_commit['induce_issue'] = []

    pp_commit['median_issue'] = 0
    pp_commit['median_time'] = 0

    issue_keys = re.findall(args.issue_patten, pp_commit['commit_msg'], re.I)
    for issue_key in  issue_keys:
        issue_key = issue_key.strip()
        issue = get_one(issues_db, {'_id': issue_key})
        if not issue: continue
        # if issue['stamp'] > pp_commit['commit_date']: continue
        pp_commit['fix'] = 1
        # issue = re.findall('[0-9]+', issue_key)[0]
        pp_commit['fix_issue'] = issue_key

    pp_commit['files_diff'] = get_files_diff(diff, blame)

    if len(pp_commit['files_diff'].keys()) >= 100:
        return None
    
    for file_diff in pp_commit['files_diff'].items():
        if len(file_diff[1]['a'].keys()) > 10000:
            return None

    return pp_commit


def preprocess_commits(args):
    pjdb = client[args.project]
    ppcommits_db = pjdb[args.preprocess_commit]
    files_db = pjdb[args.file_bugs]
    ppcommits_db.drop()
    files_db.drop()
    commits_db = pjdb[args.commit]
    diff_db = pjdb[args.diff]
    blame_db = pjdb[args.blame]
    after = datetime.datetime.strptime(args.after,"%Y-%m-%d").timestamp()
    before = datetime.datetime.strptime(args.before,"%Y-%m-%d").timestamp()
    query = {"commit_date": {'$gte': after, '$lte': before}}
    for commit in tqdm(commits_db.find(query), total=commits_db.count_documents(query)):
        diff = get_one(diff_db, {'_id': commit['_id']})
        blame = get_one(blame_db, {'_id': commit['_id']})

        if not diff or not blame:
            commits_db.delete_many({'_id': commit['_id']})
            diff_db.delete_many({'_id': commit['_id']})
            blame_db.delete_many({'_id': commit['_id']})
            continue

        ppcommit = clean_commit(args, commit, diff, blame)
        if not ppcommit: continue

        try:
            ppcommits_db.update({'_id': ppcommit['_id']}, ppcommit, check_keys=False, upsert=True)
        except:
            # DocumentTooLarge
            ppcommits_db.delete_many({'_id': commit['_id']})


def run_locating(ppcommits_db, ppcommits_dict, issues_dict, files_dict, time_limit):
    for fixed in tqdm(list(ppcommits_db.find({'fix': 1}))):
        issue_id = fixed['fix_issue']
        issue = get_one_dict(issues_dict, issue_id)
        for file in fixed['files_diff']:
            diff = fixed['files_diff'][file]
            for line in list(diff['a'].keys()):
                code = diff['a'][line]['code']
                if re.findall('#|/{2}|/\*|\*/', code): continue
                if not code.strip(): continue
                blame_commit = diff['a'][line]['blame_commit']
                prev_line = diff['a'][line]['prev_line']

                if blame_commit == '':
                    continue
                
                fix_inducing = get_one_dict(ppcommits_dict, blame_commit)
                if not fix_inducing:
                    continue
                if fix_inducing['commit_date'] > issue['stamp']:
                    continue
                if fix_inducing['median_issue'] == 1:
                    continue
                # fix_time = fixed['commit_date'] - fix_inducing['commit_date']
                # fix_time = fix_time / 86400
                # if time_limit > 0 and fix_time > time_limit:
                #     continue

                if not file in fix_inducing['fix_location']:
                    fix_inducing['fix_location'][file] = []

                fix_inducing['fix_location'][file].append(prev_line)
                fix_inducing['bug_count'] += 1
                fix_inducing['fix_id'].append(fixed['_id'])

                fix_inducing['induce_issue'].append(fixed['fix_issue'])

                file_bug_history = get_one_dict(files_dict, file)
                if not file_bug_history:
                    file_bug_history = {'_id': file, 'bugs': {}}
                
                bugs = file_bug_history['bugs']

                if fixed['_id'] not in bugs:
                    bugs[fixed['_id']] = {'_id': fixed['_id'], 'date': fixed['commit_date']}

                file_bug_history['bugs'] = bugs

                files_dict[file_bug_history['_id']] = file_bug_history
                ppcommits_dict[fix_inducing['_id']] = fix_inducing
    return ppcommits_dict, files_dict


def locating_fix_inducing(args):
    pjdb = client[args.project]
    ppcommits_db = pjdb[args.preprocess_commit]
    files_db = pjdb[args.file_bugs]
    issues_db = pjdb[args.issues]

    ppcommits_dict = conv_to_dict(ppcommits_db)
    files_dict = conv_to_dict(files_db)
    issues_dict = conv_to_dict(issues_db)

    run_locating(ppcommits_db, ppcommits_dict, issues_dict, files_dict, -1)

    induce_count = []
    for elem in tqdm(ppcommits_dict.items()):
        fix_inducing = elem[1]
        count = len(set(fix_inducing['induce_issue']))
        if count > 0:
            induce_count.append(count)
    bug_median = np.median(induce_count)
    bug_mad = np.median(np.abs(induce_count-bug_median))
    if bug_mad == 0: bug_mad = 1
    bug_upper_limit = bug_median + (3*1.4826*bug_mad)
    print(bug_median, bug_mad, bug_upper_limit)

    fix_times = []
    for elem in tqdm(ppcommits_dict.items()):
        fix_inducing = elem[1]
        if fix_inducing['bug_count'] == 0: continue
        
        for fix_id in list(set(fix_inducing['fix_id'])):
            fix_commit = get_one_dict(ppcommits_dict, fix_id)
            fix_time = fix_commit['commit_date'] - fix_inducing['commit_date']
            fix_time = fix_time / 86400
            fix_times.append(fix_time)

    time_median = np.median(fix_times)
    time_mad = np.median(np.abs(fix_times-time_median))
    time_upper_limit = time_median + (3*time_mad)
    print(time_median, time_mad, time_upper_limit)

    for ppcommit in ppcommits_db.find():
        fix_inducing = get_one_dict(ppcommits_dict, ppcommit['_id'])
        if len(set(fix_inducing['induce_issue'])) > bug_upper_limit:
            ppcommit['median_issue'] = 1
            try:
                ppcommits_db.update(
                    {'_id': ppcommit['_id']}, ppcommit, check_keys=False, upsert=True)
            except:
                pass
    
    del ppcommits_dict
    del files_dict
    ppcommits_dict = conv_to_dict(ppcommits_db)
    files_dict = conv_to_dict(files_db)

    run_locating(ppcommits_db, ppcommits_dict, issues_dict, files_dict, time_upper_limit)
        
    dict_to_db(files_db, files_dict)
    dict_to_db(ppcommits_db, ppcommits_dict)
    
    print('Finish.')


def split_data(args, data):
    idx1 = args.size + args.test
    idx2 = args.test
    idx = int(len(data)*0.8)

    if args.size:
        return data[-idx1:-idx2], data[-idx2:]

    return data[:idx], data[idx:]


def create_dir(save_path):
    if not os.path.exists('{}/deepjit'.format(save_path)):
        os.makedirs('{}/deepjit'.format(save_path))
    if not os.path.exists('{}/cc2vec'.format(save_path)):
        os.makedirs('{}/cc2vec'.format(save_path))
    if not os.path.exists('{}/feature'.format(save_path)):
        os.makedirs('{}/feature'.format(save_path))


def get_model_data(args):
    pjdb = client[args.project]
    ppcommits_db = pjdb[args.preprocess_commit]

    create_dir(args.save_path)

    ids, labels, msgs, codes, deepjit_codes, deepjit_raw_codes = [], [], [], [], [], []

    after = datetime.datetime.strptime(args.after,"%Y-%m-%d").timestamp()
    before = datetime.datetime.strptime(args.before,"%Y-%m-%d").timestamp()
    ppcommits_db.create_index('commit_date')

    msg_dict = Dict(lower=True)
    code_dict = Dict(lower=True)

    for commit in tqdm(ppcommits_db.find({"commit_date": {'$gte': after, '$lte': before}, 
                                          "median_issue": 0}).sort([('commit_date',1)])):

        label = 0 if commit['bug_count'] == 0 else 1

        commit_id = commit['commit_id']

        msg = commit['commit_msg'].strip()
        msg = split_sentence(msg)
        msg = ' '.join(msg.split(' ')).lower()

        for word in msg.split():
            msg_dict.add(word)

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
                removed_code.append(remove_code)
                for word in remove_code.split():
                    code_dict.add(word)
                # remove_code = 'removed _ code'
                file_codes.append((line, remove_code))
                if len(removed_code) > 10: break

            for line in diff['b']:
                # if len(diff['b'][line]['code'].split()) > 3:
                add_code = diff['b'][line]['code'].strip()
                add_code = ' '.join(split_sentence(add_code).split())
                add_code = ' '.join(add_code.split(' '))
                added_code.append(add_code)
                for word in add_code.split():
                    code_dict.add(word)
                # add_code = 'added _ code'
                file_codes.append((line, add_code))
                if len(added_code) > 10: break

            file_codes.sort(key=lambda x: x[0])
            raw_code.extend([code[1] for code in file_codes])
            raw_code = raw_code[:10]
            format_code.append("added _ code removed _ code")
            files_code.append({'added_code': added_code, 'removed_code': removed_code})
            # shuffle(code)

            if len(format_code) == 10: break

        # if len(format_code) == 0:
        #     continue

        ids.append(commit_id)
        labels.append(label)
        msgs.append(msg)
        deepjit_codes.append(format_code)
        deepjit_raw_codes.append(raw_code)
        codes.append(files_code)
        
    train_ids, test_ids = split_data(args, ids)
    train_labels, test_labels = split_data(args, labels)
    train_msgs, test_msgs = split_data(args, msgs)
    deepjit_train_codes, deepjit_test_codes = split_data(args, deepjit_codes)
    deepjit_train_raw_codes, deepjit_test_raw_codes = split_data(args, deepjit_raw_codes)
    train_codes, test_codes = split_data(args, codes)


    deepjit_train_data = [train_ids, train_labels,
                          train_msgs, deepjit_train_codes]
    deepjit_train_raw_data = [train_ids, train_labels,
                          train_msgs, deepjit_train_raw_codes]
    deepjit_test_data = [test_ids, test_labels, test_msgs, deepjit_test_codes]
    deepjit_test_raw_data = [test_ids, test_labels, test_msgs, deepjit_test_raw_codes]

    deepjit_all_data = [ids, labels, msgs, deepjit_codes]
    deepjit_all_raw_data = [ids, labels, msgs, deepjit_raw_codes]


    cc2vec_train_data = [train_ids, train_labels, train_msgs, train_codes]
    cc2vec_test_data = [test_ids, test_labels, test_msgs, test_codes]

    cc2vec_all_data = [ids, labels, msgs, codes]

    dextend_train_data = [train_ids, train_labels,
                          train_msgs, deepjit_train_codes]
    dextend_test_data = [test_ids, test_labels, test_msgs, deepjit_test_codes]

    dextend_all_data = [ids, labels, msgs, deepjit_codes]

    raw_dextend_train_data = [train_ids, train_labels,
                          train_msgs, deepjit_train_raw_codes]
    raw_dextend_test_data = [test_ids, test_labels, test_msgs, deepjit_test_raw_codes]

    raw_dextend_all_data = [ids, labels, msgs, deepjit_raw_codes]
    
    with open('{}./deepjit/{}_train.pkl'.format(args.save_path, args.project), 'wb') as f:
        pickle.dump(deepjit_train_data, f)
    with open('{}./deepjit/{}_test.pkl'.format(args.save_path, args.project), 'wb') as f:
        pickle.dump(deepjit_test_data, f)
    with open('{}./deepjit/{}_all.pkl'.format(args.save_path, args.project), 'wb') as f:
        pickle.dump(deepjit_all_data, f)

    with open('{}./deepjit/{}_train_raw.pkl'.format(args.save_path, args.project), 'wb') as f:
        pickle.dump(deepjit_train_raw_data, f)
    with open('{}./deepjit/{}_test_raw.pkl'.format(args.save_path, args.project), 'wb') as f:
        pickle.dump(deepjit_test_raw_data, f)
    with open('{}./deepjit/{}_all_raw.pkl'.format(args.save_path, args.project), 'wb') as f:
        pickle.dump(deepjit_all_raw_data, f)

    with open('{}./cc2vec/{}_train.pkl'.format(args.save_path, args.project), 'wb') as f:
        pickle.dump(cc2vec_train_data, f)
    with open('{}./cc2vec/{}_test.pkl'.format(args.save_path, args.project), 'wb') as f:
        pickle.dump(cc2vec_test_data, f)
    with open('{}./cc2vec/{}_all.pkl'.format(args.save_path, args.project), 'wb') as f:
        pickle.dump(cc2vec_all_data, f)

    with open('{}./cc2vec/{}_train_dextend.pkl'.format(args.save_path, args.project), 'wb') as f:
        pickle.dump(dextend_train_data, f)
    with open('{}./cc2vec/{}_test_dextend.pkl'.format(args.save_path, args.project), 'wb') as f:
        pickle.dump(dextend_test_data, f)
    with open('{}./cc2vec/{}_all_dextend.pkl'.format(args.save_path, args.project), 'wb') as f:
        pickle.dump(dextend_all_data, f)

    with open('{}./cc2vec/{}_train_dextend_raw.pkl'.format(args.save_path, args.project), 'wb') as f:
        pickle.dump(raw_dextend_train_data, f)
    with open('{}./cc2vec/{}_test_dextend_raw.pkl'.format(args.save_path, args.project), 'wb') as f:
        pickle.dump(raw_dextend_test_data, f)
    with open('{}./cc2vec/{}_all_dextend_raw.pkl'.format(args.save_path, args.project), 'wb') as f:
        pickle.dump(raw_dextend_all_data, f)

    msg_dict = msg_dict.prune(100000)
    code_dict = code_dict.prune(100000)

    project_dict = [msg_dict.get_dict(), code_dict.get_dict()]

    pickle.dump(project_dict, open("{}/{}_dict.pkl".format(args.save_path, args.project), 'wb'))
    pickle.dump(project_dict, open("{}/deepjit/{}_dict.pkl".format(args.save_path, args.project), 'wb'))
    pickle.dump(project_dict, open("{}/cc2vec/{}_dict.pkl".format(args.save_path, args.project), 'wb'))
    pickle.dump(project_dict, open("{}/feature/{}_dict.pkl".format(args.save_path, args.project), 'wb'))

    print('Train data size: {}, Bug size: {}'.format(
        len(train_labels), sum(train_labels)))
    print('Test data size: {}, Bug size: {}'.format(
        len(test_labels), sum(test_labels)))


def bug_analyse(args):
    pjdb = client[args.project]
    ppcommits_db = pjdb[args.preprocess_commit]
    ppcommits_dict = conv_to_dict(ppcommits_db)
    fix_times = []
    after = datetime.datetime.strptime(args.after,"%Y-%m-%d").timestamp()
    before = datetime.datetime.strptime(args.before,"%Y-%m-%d").timestamp()
    for commit in ppcommits_db.find({'bug_count':{'$gte':1}, "commit_date": {'$gte': after, '$lte': before}}):
        for fix_id in list(set(commit['fix_id'])):
            fix_commit = get_one_dict(ppcommits_dict, fix_id)
            fix_time = fix_commit['commit_date'] - commit['commit_date']
            fix_time = fix_time / 86400
            fix_times.append(fix_time)

    fix_times.sort(reverse=True)
    idx = int(len(fix_times) * 0.1)
    print('90% Point:',fix_times[idx])
    print('Mean:', np.mean(fix_times))
    print('Median:', np.median(fix_times))


if __name__ == "__main__":

    args = parser.parse_args()
    init_args(args)

    if len(sys.argv) < 2:
        print('Usage: python git_extraction.py [model]')
    elif args.get_repo:
        get_repo(args)
    elif args.get_commits:
        get_commits(args)
    elif args.get_issues:
        get_issues(args)
    elif args.preprocess_commits:
        preprocess_commits(args)
    elif args.locating_fix_inducing:
        locating_fix_inducing(args)
    elif args.get_model_data:
        get_model_data(args)
    elif args.bug_analyse:
        bug_analyse(args)
    