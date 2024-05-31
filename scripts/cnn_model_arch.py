import torch
from torch import nn
import torch.nn.functional as F

# python3 main2.py train  --use_trained_model=False --model='ALL_CNN_C' --checkpoint_save_name='ALL_CNN_C-1234' --lr1=0.15 --lr2=0.1 --lr3=0.05 --lr4=0.01 --weight_decay=0.0005 --use_clip=True --clip=2.0

# python3 main2.py train --max-epoch=100 --lr=0.15 --use_trained_model=False --lr_decay=1 --model='ALL_CNN_C' --checkpoint_save_name='ALL_CNN_C-1'

# python3 main2.py train --max-epoch=50 --lr=0.05 --use_trained_model=True --lr_decay=1 --model='ALL_CNN_C' --checkpoint_load_name='ALL_CNN_C-1' --checkpoint_save_name='ALL_CNN_C-2'

# python3 main2.py train --max-epoch=50 --lr=0.01 --use_trained_model=True --lr_decay=1 --model='ALL_CNN_C' --checkpoint_load_name='ALL_CNN_C-2' --checkpoint_save_name='ALL_CNN_C-3'

# python3 main2.py train --max-epoch=50 --lr=0.001 --use_trained_model=True --lr_decay=1 --model='ALL_CNN_C' --checkpoint_load_name='ALL_CNN_C-3' --checkpoint_save_name='ALL_CNN_C-4'


class cnn_model(nn.Module):
    
    def __init__(self, num_classes = 10):
        
        super(cnn_model, self).__init__()
        
        self.model_name = 'cnn_model'

        self.dp0 = nn.Dropout2d(p = 0.2)
        self.conv1 = nn.Conv2d(1, 12, 3, padding = 1)
        self.conv2 = nn.Conv2d(12, 12, 3, padding = 1)
        self.conv3 = nn.Conv2d(12, 12, 3, stride = 2, padding = 1)
        self.dp1 = nn.Dropout2d(p = 0.5)
        self.conv4 = nn.Conv2d(12, 24, 3, padding = 1)
        self.conv5 = nn.Conv2d(24, 24, 3, padding = 1)
        self.conv6 = nn.Conv2d(24, 24, 3, stride = 2, padding = 1)
        self.dp2 = nn.Dropout2d(p = 0.5)
        self.conv7 = nn.Conv2d(24, 36, 3)
        self.conv8 = nn.Conv2d(36, 36, 1)
        self.conv9 = nn.Conv2d(36, 10, 1)
        self.max = nn.MaxPool2d(5)
        
        
        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.conv2.weight)
        nn.init.xavier_normal_(self.conv3.weight)
        nn.init.xavier_normal_(self.conv4.weight)
        nn.init.xavier_normal_(self.conv5.weight)
        nn.init.xavier_normal_(self.conv6.weight)
        nn.init.xavier_normal_(self.conv7.weight)
        nn.init.xavier_normal_(self.conv8.weight)
        nn.init.xavier_normal_(self.conv9.weight)
        
        
    def forward(self, x):

        x = self.dp0(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.dp1(x)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.dp2(x)
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = self.max(x)
        x = torch.squeeze(x)
        return x