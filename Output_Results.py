'''
# Qingle Guo "Deep Multiscale Siamese Network with Parallel Convolutional Structure and Self-Attention for Change Detection" TGRS-2021
# showing the detection results of the test dataset (LEVIR-CD and SYSU)
'''

import torch.utils.data
import os
import cv2
from tqdm import tqdm
from utils.parser import get_parser_with_args
from utils.Related import get_test_loaders, initialize_metrics


if not os.path.exists('./Detection_Re'):
    os.mkdir('./Detection_Re')

# model setting, weighted setting and dataloder
parser, metadata = get_parser_with_args()
opt = parser.parse_args()
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
test_loader = get_test_loaders(opt, batch_size=1)
path = 'epoch_25.pt'   # the path of the model
model = torch.load(path)

# test processing
model.eval()
Img_index = 0
test_metrics = initialize_metrics()
with torch.no_grad():
    # Unpacking
    T = tqdm(test_loader)
    for Imgs1, Imgs2, labels in T:
        # Transferring to the device
        Imgs1 = Imgs1.float().to(dev)
        Imgs2 = Imgs2.float().to(dev)
        labels = labels.long().to(dev)

        # Model output
        Output = model(Imgs1, Imgs2)
        Output = Output[-1]
        _, Output = torch.max(Output, 1)
        Output = Output.data.cpu().numpy()
        Output = Output.squeeze() * 255

        # results saving
        file_path = './Detection_Re/' + str(Img_index).zfill(1)
        cv2.imwrite(file_path + '.png', Output)

        Img_index += 1
