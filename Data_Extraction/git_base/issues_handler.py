import sys
sys.path.append(sys.path[0]+"/..")
import os
import re
import requests

import pandas as pd
from lxml import etree
import pymongo

from utils.utils import cal_date, conv_date


client = pymongo.MongoClient("mongodb://localhost:27017/")


def get_issues(args):
    if args.project == 'qt':
        get_qt_issues(args)

    if args.project == 'openstack':
        get_openstack_issues(args)

    if args.project == 'eclipse' or args.project == 'jdt' or args.project == 'platform':
        get_eclipse_issues(args)

    if args.project == 'gerrit':
        get_gerrit_issues(args)

    if args.project == 'go':
        get_go_issues(args)

    print('Finish!')


def get_qt_issues(args):
    search_api = 'https://bugreports.qt.io/sr/jira.issueviews:searchrequest-excel-all-fields/temp/SearchRequest.xls?jqlQuery=issuetype+%3D+Bug+AND+status+%3D+Closed+AND+created+%3E%3D+{}+AND+created+%3C%3D+{}'
    all_issue_data = None

    for date in cal_date(args):
        search_url = search_api.format(date[0], date[1])
        print('Getting issue data form {} to {}'.format(date[0], date[1]))
        out_file = args.issue_datasets + './{}---{}.html'.format(date[0], date[1])
        if not os.path.exists(out_file):
            os.system("wget {} -O {}".format(search_url, out_file))
            # wget.download(search_url, out=out_file)

        table = pd.read_html(out_file)[1]
        if all_issue_data is None:
            all_issue_data = table
        else:
            all_issue_data = all_issue_data.append(
                table, ignore_index=True)
            # os.remove(out_file)

    pjdb = client[args.project]
    issues_db = pjdb['issues']

    for elem in all_issue_data.values:
        if 'bug' in elem[3].lower():
            date, stamp = conv_date(args, elem[10])
            issue = {'_id': elem[1], 'msg': elem[2], 'date': date, 'stamp': stamp}
            result = issues_db.update(
                {'_id': issue['_id']}, issue, check_keys=False, upsert=True)


def get_openstack_issues(args):
    headers = {
        'Cookie': 'anon-buglist-fields=show_id=true&show_information_type=true&show_tag=true&show_reporter=true&show_importance=true&show_assignee=true&show_date_last_updated=true&show_datecreated=true&show_targetname=true&show_heat=true&show_milestone_name=true&show_status=true',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36'
    }
    search_api = 'https://bugs.launchpad.net/openstack/+bugs?field.searchtext=&field.status%3Alist=FIXCOMMITTED&field.status%3Alist=FIXRELEASED&assignee_option=any&field.tags_combinator=ALL&field.omit_dupes=on&search=Search&orderby=-date_last_updated&memo={}&start={}'
    start = 0
    last_year = args.before[:4]
    stop_year = args.before[:4]

    while True:
        out_file = args.issue_datasets + './{}---{}.csv'.format(start, start)
        start += 75
        if os.path.exists(out_file):
            data = pd.read_csv(out_file)
            if data.values[-1][-1][:4] == stop_year:
                break
            continue
        search_url = search_api.format(start, 75)
        respond = requests.get(search_url, headers=headers)
        if respond.ok:
            html = etree.HTML(respond.content, etree.HTMLParser())

            bugs = html.xpath('//div[@class="buglisting-row"]')

            status, nums, titles, urls, dates = [], [], [], [], []
            for bug in bugs:
                statu = bug.xpath(
                    './/div[contains(@class, "status")]/text()')[0]
                num = bug.xpath('.//div[@class="buginfo"]/span/text()')[0]
                title = bug.xpath('.//div[@class="buginfo"]/a/text()')[0]
                url = bug.xpath('.//div[@class="buginfo"]/a/@href')[0]
                date = bug.xpath(
                    './/div[@class="buginfo-extra"]/span[@class="sprite milestone field"][2]/text()')[0]

                status.append(statu.strip())
                nums.append(num[1:])
                titles.append(title.strip())
                urls.append(url.strip())
                pattern = re.compile(r'[0-9]+-[0-9]+-[0-9]+')
                date = pattern.search(date).group()
                dates.append(date)

            print('Getting data before:', date)
            year = date[:4]
            if year == stop_year:
                break

            data = {'Key': nums, 'Status': status,
                    'Summary': titles, 'Url': urls, 'Last_Date': dates}

            table = pd.DataFrame(data, index=nums)
            table.to_csv(out_file)
        else:
            start -= 75
        if last_year != year:
            last_year = year
        
    files = os.popen('ls {}'.format(args.issue_datasets)).read().split('\n')
    pjdb = client[args.project]
    issues_db = pjdb['issues']

    for f in files:
        f = args.issue_datasets + f
        data = pd.read_csv(f)
        
        for elem in data.values:
            if not 'Fix' in elem[2]: continue
            date, stamp = conv_date(args, elem[5])
            issue = {'_id': str(elem[1]), 'msg': elem[3], 'date': date, 'stamp': stamp}
            
            result = issues_db.update(
                {'_id': issue['_id']}, issue, check_keys=False, upsert=True)

    print('Finish!')


def get_eclipse_issues(args):
    search_api = 'https://bugs.eclipse.org/bugs/buglist.cgi?f1=creation_ts&o1=greaterthan&query_format=advanced&v1={}&f2=creation_ts&o2=lessthaneq&query_format=advanced&v2={}&product={}&query_format=advanced&ctype=csv&human=1'
    all_issue_data = None

    projects = []
    with open('eclipse_project', 'r', encoding='utf-8') as f:
        projects = f.readlines()
        projects = [p.strip() for p in projects]

    for project in projects:
        search_url = search_api.format(args.after, args.before, project).replace(' ', '%20')
        print('Getting issue data form {} .'.format(project))
        out_file = args.issue_datasets + './{}_{}_{}.csv'.format(project, args.after, args.before)

        if not os.path.exists(out_file):
            os.system("""wget "{}" -O '{}'""".format(search_url, out_file))

        table = pd.read_csv(out_file)
        if all_issue_data is None:
            all_issue_data = table
        else:
            all_issue_data = all_issue_data.append(
                table, ignore_index=True)

    all_issue_data.to_csv(args.issue_datasets + './ALL.csv')

    pjdb = client[args.project]
    issues_db = pjdb[args.issues]

    for elem in all_issue_data.values:
        date, stamp = conv_date(args, elem[7])
        issue = {'_id': str(elem[0]), 'msg': elem[6], 'date': date, 'stamp': stamp}
        result = issues_db.update(
            {'_id': issue['_id']}, issue, check_keys=False, upsert=True)


def get_gerrit_issues(args):
    files = os.popen('ls {}'.format(args.issue_datasets + "./*.csv")).readlines()
    files = [file.strip('\n') for file in files]
    pjdb = client[args.project]
    issues_db = pjdb['issues']

    for f in files:
        data = pd.read_csv(f)[['ID', 'Type', 'Summary', 'Opened']]

        for elem in data.values:
            if type(elem[1]) == str and 'Bug' in elem[1] and not 'ago' in elem[3]:
                date, stamp = conv_date(args, elem[3])
                issue = {'_id': str(elem[0]), 'msg': elem[3], 'date': date, 'stamp': stamp}
                result = issues_db.update(
                    {'_id': issue['_id']}, issue, check_keys=False, upsert=True)


def get_go_issues(args):
    pjdb = client[args.project]
    issues_db = pjdb['issues']

    for id in range(50000):
        date, stamp = conv_date(args, args.before)
        issue = {'_id': str(id), 'msg': '', 'date': date, 'stamp': stamp}
        result = issues_db.update(
            {'_id': issue['_id']}, issue, check_keys=False, upsert=True)
