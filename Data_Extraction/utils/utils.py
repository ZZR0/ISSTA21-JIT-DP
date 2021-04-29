import datetime
import time
import re
import os
from sklearn.metrics import roc_curve, auc

def get_str_time(t_str):
    timeArray = time.strptime(t_str, "%Y-%m-%d %H:%M:%S")
    timeStamp = int(time.mktime(timeArray))
    return timeStamp

def get_one(db, query):
    for elem in db.find(query):
        return elem
    return None

def get_one_dict(db, _id):
    try:
        return db[_id]
    except:
        return None

def cal_date(args):

    start_date = datetime.datetime.strptime(args.after, '%Y-%m-%d')
    stop_date = datetime.datetime.strptime(args.before, '%Y-%m-%d')
    dates = []

    current_date = start_date
    year = start_date.year
    month = start_date.month
    day = start_date.day
    while current_date < stop_date:
        year += month // 12
        month = month % 12 + 1
        current_date = datetime.datetime.strptime('%d-%d-%d' % (year, month, day), '%Y-%m-%d')
        dates.append(current_date.strftime('%Y-%m-%d'))

    date = [(date, dates[idx+1]) for idx, date in enumerate(dates[:-1])]
    return date

def conv_to_dict(db):
    db_dict = {}
    for elem in db.find():
        db_dict[elem['_id']] = elem
    
    return db_dict

def dict_to_db(db, data_dict):
    for elem in data_dict.items():
        _id = elem[0]
        value = elem[1]
        db.update({'_id': _id}, value, check_keys=False, upsert=True)


def init_args(args):
    if args.project == 'qt':
        args.url = 'https://codereview.qt-project.org/'
        args.save_path = args.save_path if args.save_path else './datasets/qt/'
        args.issue_datasets = './issue_datasets/qt/'
        args.issue_patten = '[A-Z]+-[0-9]+'
    elif args.project == 'openstack':
        args.url = 'https://review.opendev.org/'
        args.save_path = args.save_path if args.save_path else './datasets/openstack/'
        args.issue_datasets = './issue_datasets/openstack/'
        args.issue_patten = '[0-9]{7}|[0-9]{6}'
    elif args.project == 'gerrit':
        args.url = 'https://gerrit-review.googlesource.com/'
        args.issue_patten = 'Issue (\d+)'
        args.save_path = args.save_path if args.save_path else './datasets/gerrit/'
        args.issue_datasets = './issue_datasets/gerrit/'
    elif args.project == 'jdt':
        args.url = 'https://git.eclipse.org/r/'
        args.issue_patten = '[ |:|[][0-9]{6}[ |:|\]]'
        args.save_path = args.save_path if args.save_path else './datasets/jdt/'
        args.issue_datasets = './issue_datasets/jdt/'
    elif args.project == 'platform':
        args.url = 'https://git.eclipse.org/r/'
        args.issue_patten = '[ |:|[][0-9]{6}[ |:|\]]'
        args.save_path = args.save_path if args.save_path else './datasets/platform/'
        args.issue_datasets = './issue_datasets/platform/'
    elif args.project == 'go':
        args.url = 'https://go-review.googlesource.com/'
        args.issue_patten = '#([0-9]+)'
        args.save_path = args.save_path if args.save_path else './datasets/go/'
        args.issue_datasets = './issue_datasets/go/'
    else:
        print('Unknown Project.')
        exit()


def reduce_elem(elem):
    del elem[list(elem.keys())[-1]]

def update_large_elem(db, elem):
    while True:
        try:
            result = db.update(
                {'_id': elem['_id']}, elem, check_keys=False, upsert=True)
            break
        except:
            # DocumentTooLarge
            result = db.delete_many({'_id': elem['_id']})
            reduce_elem(elem)
            
    return result


def eval(labels, predicts, thresh=0.5):
    TP, FN, FP, TN = 0, 0, 0, 0
    for lable, predict in zip(labels, predicts):
        # print(predict)
        if predict >= thresh and lable == 1:
            TP += 1
        if predict >= thresh and lable == 0:
            FP += 1
        if predict < thresh and lable == 1:
            FN += 1
        if predict < thresh and lable == 0:
            TN += 1
    
    print(TP, FN, FP, TN)
    try:
        P = TP/(TP+FP)
        R = TP/(TP+FN)

        A = (TP+TN)/len(labels)
        E = FP/(TP+FP)

        print('Test data at Threshold %.2f -- Accuracy: %.2f, False Positives: %.2f, Precision: %.2f, Recall: %.2f'%(thresh, A, E, P, R))
    except Exception:
        # division by zero
        pass

def conv_date(args, date_str):
    QT_DATE = re.compile(r"(?P<day>\d*?) (?P<month>.*) (?P<year>.*) (?P<hour>\d*):(?P<minute>\d*)")
    ECLIPSE_DATE = re.compile(r"(?P<year>\d*)-(?P<month>\d*)-(?P<day>\d*) (?P<hour>\d*):(?P<minute>\d*):(?P<second>\d*)")
    GERRIT_DATE = re.compile(r"(?P<month>.*) (?P<day>\d*), (?P<year>\d*)")
    GO_DATE = re.compile(r"(?P<year>\d*)-(?P<month>\d*)-(?P<day>\d*)")
    project_date = {'qt': QT_DATE, 
                    'jdt': ECLIPSE_DATE,
                    'platform': ECLIPSE_DATE,
                    'eclipse': ECLIPSE_DATE,
                    'gerrit': GERRIT_DATE,
                    'go': GO_DATE,
                    'openstack': GO_DATE}

    DATE = project_date[args.project]

    date_mapping = {'Jan': '01',
                    'Feb': '02',
                    'Mar': '03',
                    'Apr': '04',
                    'May': '05',
                    'Jun': '06',
                    'Jul': '07',
                    'Aug': '08',
                    'Sep': '09',
                    'Oct': '10',
                    'Nov': '11',
                    'Dec': '12',
                    "'20": '2020',
                    "'19": '2019',
                    "'18": '2018',
                    "'17": '2017',
                    "'16": '2016',
                    "'15": '2015',
                    "'14": '2014',
                    "'13": '2013',
                    "'12": '2012',
                    "'11": '2011',
                    "'10": '2010',}

    match = DATE.search(date_str)
    date = match.groupdict()
    date['month'] = date_mapping[date['month']] if date['month'] in date_mapping else date['month']
    date['year'] = date_mapping[date['year']] if date['year'] in date_mapping else date['year']
    date['hour'] = date['hour'] if 'hour' in date else '23'
    date['minute'] = date['minute'] if 'minute' in date else '59'
    date['second'] = date['second'] if 'second' in date else '59'

    date = '{}-{}-{} {}:{}:{}'.format(date['year'], date['month'], date['day'],
                                    date['hour'], date['minute'], date['second'])

    str2date=datetime.datetime.strptime(date,"%Y-%m-%d %H:%M:%S")

    return date, str2date.timestamp()

def exec_cmd(cmd):
    pip = os.popen(cmd)
    output = pip.buffer.read().decode(encoding='utf8', errors="ignore")
    output = output.strip('\n').split('\n') if output else []
    return output

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

def split_data(data):
    idx = int(len(data)*0.8)
    return data[:idx], data[idx:]

def eval_auc(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_pred, pos_label=1)
    auc_ = auc(fpr, tpr)
    return auc_