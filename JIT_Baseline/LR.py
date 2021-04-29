import torch
import torch.nn as nn

class LR(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LR, self).__init__()
        # self.fc = nn.Linear(input_size, 128)
        # self.fc1 = nn.Linear(128, 256)
        # self.fc2 = nn.Linear(256, 64)
        # self.fc3 = nn.Linear(64, num_classes)

        self.fc = nn.Linear(input_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_size):
        # out = self.fc(input_size)
        # out = self.fc1(out)
        # out = self.fc2(out)
        # out = self.fc3(out)

        out = self.fc(input_size)
        out = self.sigmoid(out).squeeze(1)
        return out

    def predict(self, data):
        with torch.no_grad():
            self.eval()  # since we use drop out
            all_predict, all_label = list(), list()
            for batch in data:
                x, y = batch
                x = torch.tensor(x).float()

                predict = self.forward(x).detach().numpy().tolist()
                all_predict += predict
                all_label += y.tolist()
            # acc, prc, rc, f1, auc_ = evaluation_metrics(y_pred=all_predict, y_true=all_label)
            # print('Accuracy: %f -- Precision: %f -- Recall: %f -- F1: %f -- AUC: %f' % (acc, prc, rc, f1, auc_))
            return all_predict, all_label
