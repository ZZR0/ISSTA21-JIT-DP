import os

cmd_c = '\cp snapshot/{}/epoch_50.pt.result result/{}_deepjit.result'
cmd_r = '\cp snapshot/{}/raw/epoch_50.pt.result result/{}_deepjitraw.result'

for project in ['gerrit', 'go', 'jdt', 'qt', 'openstack', 'platform']:
    os.system(cmd_c.format(project, project))
    os.system(cmd_r.format(project, project))
