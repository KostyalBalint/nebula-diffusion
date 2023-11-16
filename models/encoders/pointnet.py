import torch
import torch.nn.functional as F
from torch import nn


class PointNetEncoder(nn.Module):
    def __init__(self, zdim, input_dim=3, size='original'):
        super().__init__()
        self.zdim = zdim
        self.size = size
        self.conv1 = nn.Conv1d(input_dim, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        if self.size == 'big':
            self.conv5 = nn.Conv1d(512, 1024, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
        if self.size == 'big':
            self.bn5 = nn.BatchNorm1d(1024)

        # Mapping to [c], cmean
        if self.size == 'big':
            self.fc0_m = nn.Linear(1024, 512)
        self.fc1_m = nn.Linear(512, 256)
        self.fc2_m = nn.Linear(256, 128)
        self.fc3_m = nn.Linear(128, zdim)

        if self.size == 'big':
            self.fc_bn0_m = nn.BatchNorm1d(512)
        self.fc_bn1_m = nn.BatchNorm1d(256)
        self.fc_bn2_m = nn.BatchNorm1d(128)

        # Mapping to [c], cmean
        if self.size == 'big':
            self.fc0_v = nn.Linear(1024, 512)
        self.fc1_v = nn.Linear(512, 256)
        self.fc2_v = nn.Linear(256, 128)
        self.fc3_v = nn.Linear(128, zdim)

        if self.size == 'big':
            self.fc_bn0_v = nn.BatchNorm1d(512)
        self.fc_bn1_v = nn.BatchNorm1d(256)
        self.fc_bn2_v = nn.BatchNorm1d(128)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        if self.size == 'big':
            x = self.bn5(self.conv5(x))
        x = torch.max(x, 2, keepdim=True)[0]
        if self.size == 'big':
            x = x.view(-1, 1024)
        if self.size == 'original':
            x = x.view(-1, 512)

        m = x
        if self.size == 'big':
            m = F.relu(self.fc_bn0_m(self.fc0_m(m)))
        m = F.relu(self.fc_bn1_m(self.fc1_m(m)))
        m = F.relu(self.fc_bn2_m(self.fc2_m(m)))
        m = self.fc3_m(m)

        v = x
        if self.size == 'big':
            v = F.relu(self.fc_bn0_v(self.fc0_v(v)))
        v = F.relu(self.fc_bn1_v(self.fc1_v(v)))
        v = F.relu(self.fc_bn2_v(self.fc2_v(v)))
        v = self.fc3_v(v)

        # Returns both mean and logvariance, just ignore the latter in deteministic cases.
        return m, v

