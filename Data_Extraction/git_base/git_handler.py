import sys
sys.path.append(sys.path[0]+"/..")
import os

import pymongo
from tqdm import tqdm

import line_parser
from aggregator import aggregator
from utils.utils import get_one, exec_cmd

GIT_LOG_CMD = 'git log --all --after={} --before={} --no-decorate --no-merges --pretty=format:"%H"'
GIT_SHOW_CMD = "git show {} --name-only --pretty=format:'%H%n%P%n%an%n%ct%n%s%n%B%n[ALL CHANGE FILES]'"
GIT_DIFF_CMD = 'git diff --unified=999999999 {} {} {}'
GIT_SHOW_DIFF_CMD = 'git show {} --pretty=format: --unified=999999999'
GIT_BLAME_CMD = 'git blame -t -n -l {} "{}"'
GIT_LIST_BRANCH_CMD = 'git branch -a'


def process_one_line_blame(log):
    log = log.split()
    blame_id = log[0]
    while not log[1].isnumeric():
        log.remove(log[1])
    blame_line_a = int(log[1])
    for idx, word in enumerate(log[2:]):
        if word.isnumeric():
            break
    idx = idx+2
    blame_date = int(log[idx])
    blame_autor = ' '.join(log[2:idx])[1:]
    blame_line_b = int(log[idx+2][:-1])

    return {'blame_id':blame_id, 'blame_line_a':blame_line_a, 'blame_autor':blame_autor, 
            'blame_date':blame_date, 'blame_line_b':blame_line_b}


def get_file_blame(file_blame_log):
    file_blame_log = [log.strip('\t').strip() for log in file_blame_log]
    line2id, id2line = {}, {}
    for _, log in enumerate(file_blame_log):
        line_blame = process_one_line_blame(log)

        if not line_blame['blame_id'] in id2line:
            id2line[line_blame['blame_id']] = \
                {'id': line_blame['blame_id'], 'author': line_blame['blame_autor'], 'time': line_blame['blame_date'], 'ranges':[]}
        
        idb = id2line[line_blame['blame_id']]
        this_line = line_blame['blame_line_b']
        ranges = idb['ranges']
        if ranges:
            if this_line == ranges[-1]['end'] + 1:
                ranges[-1]['end'] += 1
            else:
                ranges.append({'start':this_line, 'end':this_line})
        else:
            ranges.append({'start':this_line, 'end':this_line})

        del line_blame['blame_autor']
        del line_blame['blame_date']
        del line_blame['blame_line_b']

        line2id[str(this_line)] = line_blame

    file_blame = {'line2id':line2id, 'id2line':id2line}
    return file_blame


def split_diff_log(file_diff_log):
    files_log, file_log = [], []
    for line in file_diff_log:
        if line[:10] == 'diff --git':
            if file_log:
                files_log.append(file_log)
                file_log = []

        file_log.append(line)

    if file_log:
        files_log.append(file_log) 
    
    return files_log


def get_commit_diff_blame(commit):
    commit_id = commit['commit_id']
    parent_id = commit['parent_id']
    commit_diff = {'_id': commit_id}
    commit_blame = {'_id': commit_id}
    files = []
    
    file_diff_log = exec_cmd(GIT_SHOW_DIFF_CMD.format(commit_id))
    files_log = split_diff_log(file_diff_log)

    for file_log in files_log:

        try:
            files_diff = aggregator(line_parser.parse_lines(file_log))
        
            for file_diff in files_diff:
                file_name_a = file_diff['from']['file'] if \
                    file_diff['rename'] or file_diff['from']['mode'] != '0000000' else file_diff['to']['file']
                file_name_b = file_diff['to']['file'] if \
                    file_diff['rename'] or file_diff['to']['mode'] != '0000000' else file_diff['from']['file']
                assert file_name_b in commit['files']
                if file_diff['is_binary'] or len(file_diff['content'])==0:
                    continue
                
                file_blame_log = exec_cmd(GIT_BLAME_CMD.format(parent_id, file_name_a))
                if not file_blame_log:
                    continue
                file_blame = get_file_blame(file_blame_log)

                commit_diff[file_name_b] = file_diff
                commit_blame[file_name_b] = file_blame
                files.append(file_name_b)
                
            # if len(files) > 100: break
        except:
            pass
    commit['files'] = files
    return commit_diff, commit_blame


def get_extention(file):
    ext = file.split('.')[-1]
    return ext


def get_git_log(args):
    log_list = exec_cmd(GIT_LOG_CMD.format(args.after, args.before))
    commit_list = []
    print('Getting Git Log...')
    for commit_log in tqdm(log_list):
        commit_log = commit_log.strip().split()
        commit_id = commit_log[0]

        show_msg = exec_cmd(GIT_SHOW_CMD.format(commit_id))
        show_msg = [msg.strip() for msg in show_msg]

        file_index = show_msg.index('[ALL CHANGE FILES]')

        subject = show_msg[4]
        head = show_msg[:5]
        commit_msg = show_msg[5:file_index]
        files = show_msg[file_index+1:]
        # files = [file for file in files if get_extention(file).upper() in code_extentions]

        assert show_msg[0] == commit_id

        parent_id = head[1]
        author = head[2]
        commit_date = head[3]

        commit_msg = ' '.join(commit_msg)

        commit = {'_id': commit_id, 'commit_id':commit_id, 'parent_id':parent_id, 'subject':subject,
                  'commit_msg':commit_msg, 'author':author, 'commit_date':int(commit_date), 'files':files}

        commit_list.append(commit)


    return commit_list


def extract_git_repo(args):
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    project_db = client[args.project]
    commit_db = project_db[args.commit]
    diff_db = project_db[args.diff]
    blame_db = project_db[args.blame]

    commits = get_git_log(args)
    for commit in tqdm(commits):
        # if len(commit['files']) > 100: continue
        if args.skip and get_one(commit_db, {'_id':commit['_id']}): continue
        commit['repo'] = args.repo
        
        try:
            diff, blame = get_commit_diff_blame(commit)
        except:
            continue

        if len(commit['files']) < 1: 
            continue

        try:
            result = commit_db.update(
                {'_id': commit['_id']}, commit, check_keys=False, upsert=True)
            result = diff_db.update(
                {'_id': diff['_id']}, diff, check_keys=False, upsert=True)
            result = blame_db.update(
                {'_id': blame['_id']}, blame, check_keys=False, upsert=True)
        except:
            # DocumentTooLarge
            result = commit_db.delete_many({'_id': commit['_id']})
            result = diff_db.delete_many({'_id': diff['_id']})
            result = blame_db.delete_many({'_id': blame['_id']})


def get_commits(args):
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    project_db = client[args.project]
    commit_db = project_db[args.commit]
    diff_db = project_db[args.diff]
    blame_db = project_db[args.blame]

    if args.drop:
        commit_db.drop()
        diff_db.drop()
        blame_db.drop()

    path = './git_datasets/{}'.format(args.project)
    os.chdir(path)
    repos = os.listdir()
    for repo in repos:
        if os.path.isfile(repo): continue
        print('Start Extract {}'.format(repo))
        args.repo = repo
        os.chdir(repo)
        extract_git_repo(args)
        os.chdir('../')
    
    print('Finish!')


def get_repo(args):
    repo_urls_file = 'git_datasets/{}/repo_urls.txt'.format(args.project)

    with open(repo_urls_file, 'r', encoding='utf-8') as f:
        repo_urls = [url.strip('\n') for url in f.readlines()]

    os.chdir('git_datasets/{}'.format(args.project))

    for cmd in repo_urls:
        project = cmd.split('/')[-1]

        if os.path.exists(project):
            continue
        
        result = 1
        try:
            result = os.system(cmd)
        except:
            IOError('Git Clone Error.')

        if result != 0:
            if os.path.exists(project):
                os.removedirs(project)

