import torch
import torch.nn as nn
from tsai.models.InceptionTimePlus import *

class ECG_Inception(nn.Module):
    def __init__(self, normal_hypo_ratio=4.0, require_dropout=True) -> None:
        super().__init__()
        self.name = "inception"
        self.normal_hypo_ratio = normal_hypo_ratio

        if require_dropout:
            self.layer = InceptionTimePlus(1, 1, conv_dropout=0.1, fc_dropout=0.1)
        else:
            self.layer = InceptionTimePlus(1, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.layer(x)
        x = self.sigmoid(x)
        x = x.squeeze()
        return x

    def loss(self, x, gt, weighted=False):
        assert x.shape == gt.shape

        if weighted:
            # Weighted BCE Loss
            pos_weight = torch.where(gt > 0.5, torch.tensor(self.normal_hypo_ratio), torch.tensor(1.0))
            return  nn.BCELoss(weight=pos_weight)(x, gt)
        else:
            return  nn.BCELoss()(x, gt)
        

class PPG_Inception(nn.Module):
    def __init__(self, normal_hypo_ratio=4.0, require_dropout=True) -> None:
        super().__init__()
        self.name = "ppg_inception"
        self.normal_hypo_ratio = normal_hypo_ratio

        if require_dropout:
            self.layer = InceptionTimePlus(1, 1, conv_dropout=0.1, fc_dropout=0.1)
        else:
            self.layer = InceptionTimePlus(1, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.layer(x)
        x = self.sigmoid(x)
        x = x.squeeze()
        return x

    def loss(self, x, gt, weighted=False):
        assert x.shape == gt.shape

        if weighted:
            # Weighted BCE Loss
            pos_weight = torch.where(gt > 0.5, torch.tensor(self.normal_hypo_ratio), torch.tensor(1.0))
            return  nn.BCELoss(weight=pos_weight)(x, gt)
        else:
            return  nn.BCELoss()(x, gt)
        
# class EDA_LSTM(nn.Module):
#     def __init__(self, normal_hypo_ratio=4.0):
#         super().__init__()
#         self.name = "eda_lstm"
#         self.normal_hypo_ratio = normal_hypo_ratio

#         self.encoder1 = nn.Sequential(
#             nn.Conv1d(1, 4, kernel_size=12, stride=6, padding=3),
#             nn.ReLU(),
#             nn.BatchNorm1d(4),
#             nn.Conv1d(4, 4, kernel_size=9, stride=5, padding=2),
#             nn.ReLU(),
#         ) 
#         self.encoder2 = nn.Sequential(
#             nn.Conv1d(1, 4, kernel_size=12, stride=6, padding=3),
#             nn.ReLU(),
#             nn.BatchNorm1d(4),
#             nn.Conv1d(4, 4, kernel_size=9, stride=5, padding=2),
#             nn.ReLU(),
#         )

#         self.num_layers = 1
#         self.hidden_size = 8
#         self.lstm = nn.LSTM(8, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)

#         self.fc = nn.Sequential(
#             nn.Linear(16, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, x1, x2):
#         x1 = x1.unsqueeze(1)
#         x1 = self.encoder1(x1)
#         x2 = x2.unsqueeze(1)
#         x2 = self.encoder2(x2)

#         x = torch.cat([x1, x2], dim=1)
#         x = x.permute(0, 2, 1).contiguous()

#         x, _ = self.lstm(x)
#         x = self.fc(x[:, -1, :])
#         x = x.squeeze()

#         return x
    
#     def loss(self, x, gt, weighted=False):
#         assert x.shape == gt.shape

#         if weighted:
#             # Weighted BCE Loss
#             pos_weight = torch.where(gt > 0.5, torch.tensor(self.normal_hypo_ratio), torch.tensor(1.0))
#             return  nn.BCELoss(weight=pos_weight)(x, gt)
#         else:
#             return  nn.BCELoss()(x, gt)


class EDA_LSTM(nn.Module):
    def __init__(self, normal_hypo_ratio=4.0):
        super().__init__()
        self.name = "eda_lstm"
        self.normal_hypo_ratio = normal_hypo_ratio

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=12, stride=6, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(4),
            nn.Conv1d(4, 8, kernel_size=9, stride=5, padding=2),
            nn.ReLU(),
        ) 
        self.num_layers = 1
        self.hidden_size = 8
        self.lstm = nn.LSTM(8, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)

        self.fc = nn.Sequential(
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        self._initialize_weights()

    def forward(self, x1, x2):
        x = x1 + x2
        x = x.unsqueeze(1)
        x = self.encoder(x)

        x = x.permute(0, 2, 1).contiguous()

        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        x = x.squeeze()

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def loss(self, x, gt, weighted=False):
        assert x.shape == gt.shape

        if weighted:
            # Weighted BCE Loss
            pos_weight = torch.where(gt > 0.5, torch.tensor(self.normal_hypo_ratio), torch.tensor(1.0))
            return  nn.BCELoss(weight=pos_weight)(x, gt)
        else:
            return  nn.BCELoss()(x, gt)

if __name__ == "__main__":
    input_data = torch.randn(32, 250)
    model = ECG_Inception()
    preds = model(input_data)
    print("Number of parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(preds.shape)