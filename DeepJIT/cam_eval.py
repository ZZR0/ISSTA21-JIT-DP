from model import DeepJIT
from utils import mini_batches_test
from sklearn.metrics import roc_auc_score    
import torch 
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F


class MsgFeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """

    def __init__(self):
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x, convs):
        activations, outputs = [], []
        self.gradients = []
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        for conv in convs:
            conv_x = conv(x)
            conv_x.register_hook(self.save_gradient)
            activations += [conv_x]
            outputs += [F.relu(conv_x).squeeze(3)]

        outputs = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in outputs]  # [(N, Co), ...]*len(Ks)
        outputs = torch.cat(outputs, 1)
        return activations, outputs

class CodeFeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """

    def __init__(self, args):
        self.line_gradients = []
        self.hunk_gradients = []
        self.args = args

    def save_line_gradient(self, grad):
        self.line_gradients.append(grad)
    
    def save_hunk_gradient(self, grad):
        self.hunk_gradients.append(grad)

    def __call__(self, x, convs_line, convs_hunks):
        line_activations, line_outputs, hunk_activations, hunk_outputs = [], [], [], []
        self.line_gradients = []
        self.hunk_gradients = []

        n_batch, n_file = x.shape[0], x.shape[1]
        x = x.reshape(n_batch * n_file, x.shape[2], x.shape[3])

        x = x.unsqueeze(1)  # (N, Ci, W, D)
        for conv in convs_line:
            conv_x = conv(x)
            conv_x.register_hook(self.save_line_gradient)
            line_activations += [conv_x]
            line_outputs += [F.relu(conv_x).squeeze(3)]

        line_outputs = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in line_outputs]  # [(N, Co), ...]*len(Ks)
        line_outputs = torch.cat(line_outputs, 1)

        x = line_outputs.reshape(n_batch, n_file, self.args.num_filters * len(self.args.filter_sizes))
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        for conv in convs_hunks:
            conv_x = conv(x)
            conv_x.register_hook(self.save_hunk_gradient)
            hunk_activations += [conv_x]
            hunk_outputs += [F.relu(conv_x).squeeze(3)]

        hunk_outputs = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in hunk_outputs]  # [(N, Co), ...]*len(Ks)
        hunk_outputs = torch.cat(hunk_outputs, 1)

        return line_activations, line_outputs, hunk_activations, hunk_outputs

class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, args):
        self.model = model
        self.msg_feature_extractor = MsgFeatureExtractor()
        self.code_feature_extractor = CodeFeatureExtractor(args)

    def get_msg_gradients(self):
        return self.msg_feature_extractor.gradients

    def get_line_gradients(self):
        return self.code_feature_extractor.line_gradients
    
    def get_hunk_gradients(self):
        return self.code_feature_extractor.hunk_gradients

    def __call__(self, msg, code):
        msg_activations = []

        x_msg = self.model.embed_msg(msg)
        # x_msg = self.forward_msg(x_msg, self.convs_msg)
        msg_activations, x_msg = self.msg_feature_extractor(x_msg, self.model.convs_msg)

        line_activations, hunk_activations = None, None
        # x_code = self.model.embed_code(code)
        # x_code = self.model.forward_code(x_code, self.model.convs_code_line, self.model.convs_code_file)
        # line_activations, line_outputs, hunk_activations, x_code \
        #     = self.code_feature_extractor(x_code, self.model.convs_code_line, self.model.convs_code_file)
        # x_commit = torch.cat((x_msg, x_code), 1)

        x_commit = x_msg
        x_commit = self.model.dropout(x_commit)
        out = self.model.fc1(x_commit)
        out = F.relu(out)
        out = self.model.fc2(out)
        out = self.model.sigmoid(out).squeeze(1)
        return msg_activations, line_activations, hunk_activations, out


class GradCam:
    def __init__(self, model, use_cuda, args):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.args = args

        self.extractor = ModelOutputs(self.model, args)

    def __call__(self, pad_msg, pad_code):
        msg_features, line_features, hunk_features, output = self.extractor(pad_msg, pad_code)

        msg_cams, code_cams = [], []
        for index in range(output.shape[0]):
            one_hot = np.zeros((1, output.shape[0]), dtype=np.float32)
            one_hot[0][index] = 1
            one_hot = torch.from_numpy(one_hot).requires_grad_(True)
            if self.cuda:
                one_hot = torch.sum(one_hot.cuda() * output)
            else:
                one_hot = torch.sum(one_hot * output)

            self.model.zero_grad()
            one_hot.backward(retain_graph=True)

            msg_weights, msg_targets = [], []
            msg_len = self.args.msg_length

            for k in range(len(msg_features)):
                grads_val = self.extractor.get_msg_gradients()[-(1+k)].squeeze(-1)[index].cpu().data.numpy()

                target = msg_features[k].squeeze(-1)[index]
                target = target.cpu().data.numpy()

                weight = np.mean(grads_val, axis=0)
                target = np.mean(target, axis=0)

                msg_weights += [weight]
                msg_targets += [target]

            cam = np.zeros(msg_len, dtype=np.float32)
            for i in range(msg_len):
                if i == 0:
                    cam[i] += msg_weights[0][i] * msg_targets[0][i] + 1/2 * msg_weights[1][i] * msg_targets[1][i] + 1/3 * msg_weights[2][i] * msg_targets[2][i]
                elif i == 1:
                    cam[i] += msg_weights[0][i] * msg_targets[0][i] + 1/2 * msg_weights[1][i-1] * msg_targets[1][i-1] + 1/2 * msg_weights[1][i] * msg_targets[1][i] + 1/3 * msg_weights[2][i-1] * msg_targets[2][i-1] + 1/3 * msg_weights[2][i] * msg_targets[2][i]
                elif i == msg_len-2:
                    cam[i] += msg_weights[0][i] * msg_targets[0][i] + 1/2 * msg_weights[1][i-1] * msg_targets[1][i-1] + 1/2 * msg_weights[1][i] * msg_targets[1][i] + 1/3 * msg_weights[2][i-2] * msg_targets[2][i-2] + 1/3 * msg_weights[2][i-1] * msg_targets[2][i-1]
                elif i == msg_len-1:
                    cam[i] += msg_weights[0][i] * msg_targets[0][i] + 1/2 * msg_weights[1][i-1] * msg_targets[1][i-1] + 1/3 * msg_weights[2][i-2] * msg_targets[2][i-2]
                else:
                    cam[i] += msg_weights[0][i] * msg_targets[0][i] + 1/2 * msg_weights[1][i-1] * msg_targets[1][i-1] + 1/2 * msg_weights[1][i] * msg_targets[1][i] + 1/3 * msg_weights[2][i-2] * msg_targets[2][i-2] + 1/3 * msg_weights[2][i-1] * msg_targets[2][i-1] + 1/3 * msg_weights[2][i] * msg_targets[2][i]

            cam = np.maximum(cam, 0)
            # cam = cv2.resize(cam, input.shape[2:])
            cam = cam - np.min(cam)
            if np.max(cam) > 0:
                cam = cam / np.max(cam)
            msg_cams.append(cam.tolist())


            # line_weights, line_targets = [], []
            # code_len = self.args.code_length
            # hunk_weights, hunk_targets = [], []
            # hunk_len = self.args.code_line

            # for k in range(len(line_features)):
            #     grads_val = self.extractor.get_line_gradients()[-(1+k)].squeeze(-1)[index*hunk_len:(index+1)*hunk_len].cpu().data.numpy()

            #     target = line_features[k].squeeze(-1)[index*hunk_len:(index+1)*hunk_len]
            #     target = target.cpu().data.numpy()

            #     weight = np.mean(grads_val, axis=1)
            #     target = np.mean(target, axis=1)

            #     line_weights += [weight]
            #     line_targets += [target]

            # line_cam = np.zeros((hunk_len, code_len), dtype=np.float32)
            # for i in range(code_len):
            #     line_cam[:, i] += line_weights[0][:, i] * line_targets[0][:, i] #+ weights[1][i] * targets[1][i] + weights[2][i] * targets[2][i]

            # for k in range(len(hunk_features)):
            #     grads_val = self.extractor.get_hunk_gradients()[-(1+k)].squeeze(-1)[index].cpu().data.numpy()

            #     target = hunk_features[k].squeeze(-1)[index]
            #     target = target.cpu().data.numpy()

            #     weight = np.mean(grads_val, axis=0)
            #     target = np.mean(target, axis=0)

            #     hunk_weights += [weight]
            #     hunk_targets += [target]

            # hunk_cam = np.zeros(hunk_len, dtype=np.float32)
            # for i in range(hunk_len):
            #     hunk_cam[i] += hunk_weights[0][i] * hunk_targets[0][i] #+ weights[1][i] * targets[1][i] + weights[2][i] * targets[2][i]

            # code_cam = np.zeros((hunk_len, code_len), dtype=np.float32)
            # code_cam = hunk_cam.reshape(hunk_len, 1) * line_cam 

            # code_cam = np.maximum(code_cam, 0)
            # # cam = cv2.resize(cam, input.shape[2:])
            # code_cam = code_cam - np.min(code_cam)
            # code_cam = code_cam / np.max(code_cam)
            # code_cams.append(code_cam.tolist())
        return msg_cams, code_cams, output

def evaluation_model(data, params):
    ids, pad_msg, pad_code, labels, dict_msg, dict_code = data
    batches = mini_batches_test(ids=ids, X_msg=pad_msg, X_code=pad_code, Y=labels, mini_batch_size=32)

    params.vocab_msg, params.vocab_code = len(dict_msg), len(dict_code)
    if len(labels.shape) == 1:
        params.class_num = 1
    else:
        params.class_num = labels.shape[1]

    # set up parameters
    params.cuda = (not params.no_cuda) and torch.cuda.is_available()
    del params.no_cuda
    params.filter_sizes = [int(k) for k in params.filter_sizes.split(',')]

    model = DeepJIT(args=params)
    if torch.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(torch.load(params.load_model))

    # model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    grad_cam = GradCam(model=model, use_cuda=params.cuda, args=params)

    all_ids, all_msg, all_code, all_msg_mask, all_code_mask, all_predict, all_label = list(), list(), list(), list(), list(), list(), list()
    for i, (batch) in enumerate(tqdm(batches)):
        _id, pad_msg, pad_code, label = batch
        if torch.cuda.is_available():                
            pad_msg, pad_code, labels = torch.tensor(pad_msg).cuda(), torch.tensor(
                pad_code).cuda(), torch.cuda.FloatTensor(label)
        else:                
            pad_msg, pad_code, label = torch.tensor(pad_msg).long(), torch.tensor(pad_code).long(), torch.tensor(
                label).float()

        # If None, returns the map for the highest scoring category.
        # Otherwise, targets the requested index.
        msg_mask, code_mask, predict = grad_cam(pad_msg, pad_code)

        if torch.cuda.is_available():
            predict = predict.cpu().detach().numpy().tolist()
            pad_msg = pad_msg.cpu().detach().numpy().tolist()
            pad_code = pad_code.cpu().detach().numpy().tolist()
        else:
            predict = predict.detach().numpy().tolist()
            pad_msg = pad_msg.detach().numpy().tolist()
            pad_code = pad_code.detach().numpy().tolist()


        all_ids += _id
        all_msg += pad_msg
        all_code += pad_code
        all_msg_mask += msg_mask
        all_code_mask += code_mask
        all_predict += predict
        all_label += label.tolist()

    auc_score = roc_auc_score(y_true=all_label,  y_score=all_predict)
    print('Test data -- AUC score:', auc_score)

    return all_ids, all_msg, all_msg_mask, all_code, all_code_mask, all_predict, all_label