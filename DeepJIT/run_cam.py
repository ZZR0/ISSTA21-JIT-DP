import pickle
from utils import *

def visualize(_id, predict, label, masks, words, sent_attns=[1], msg=False, save_html=False, save_img=True):
    h5_string_list = list()
    h5_string_list.append('<div class="cam">')
    h5_string_list.append("<head><meta charset='utf-8'></head>")
    h5_string_list.append("Change Id: {}<br>".format(_id))
    h5_string_list.append("Change Label: {}，Predict: {}<br>".format(label, predict))

    save_path = '/data2/cg/DeepJIT/cam/code/'
    if msg:
        masks = [masks]
        words = [words]
        save_path = '/data2/cg/DeepJIT/cam/msg/'
    
    for line_mask, line_word, sent_attn in zip(masks, words, sent_attns):
        h5_string_list.append(
                    '<font style="background: rgba(0, 0, 255, %f)">&nbsp&nbsp&nbsp&nbsp&nbsp</font>' % sent_attn)
        for mask, word in zip(line_mask, line_word):
            h5_string_list.append('<font style="background: rgba(255, 0, 0, %f)">%s </font>' % (mask, word))
        h5_string_list.append('<br>')
    h5_string_list.append('</div>')

    h5_string = ''.join(h5_string_list)

    h5_path = os.path.join(save_path, "{}.html".format(_id))
    with open(h5_path, "w") as h5_file:
        h5_file.write(h5_string)

    if save_img:
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')

        ob=Screenshot_Clipping.Screenshot()
        driver = webdriver.Chrome(options=options)
        url = "file:///{}".format(h5_path)
        # print(url)
        driver.get(url)

        element=driver.find_element_by_class_name('cam')

        img_url=ob.get_element(driver, element, save_location=save_path)
        img_path = os.path.join(save_path, "{}.png".format(_id))

        os.system('mv {} {}'.format(img_url, img_path))
        driver.close()
        driver.quit()

    if not save_html:
        os.system('rm {}'.format(h5_path))

# project = 'qt'
project = 'openstack'


data = pickle.load(open('cam/{}_msg_cam.pkl'.format(project), 'rb'))
all_ids, all_msg, all_msg_mask, all_predict, all_label = data

dictionary = pickle.load(open('{}/{}_dict.pkl'.format(project, project), 'rb'))  
dict_msg, dict_code = dictionary
id2msg_world = get_world_dict(dict_msg)

for _id, msg, maks, pred, label in zip(all_ids, all_msg, all_msg_mask, all_predict, all_label):
    # plt.figure(figsize=(100,10))
    msg = mapping_dict_world(msg, id2msg_world)
    if 'task-number' in msg or 'fix' in msg or 'bug' in msg or 'failures' in msg \
        or 'resolves' in msg or 'fail' in msg or 'bugs' in msg:
        if label == 1 and pred > 0.5:
            visualize(_id, pred, label, maks, msg, msg=True)