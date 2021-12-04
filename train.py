"""
training and validation process
Please cited:
Q. Guo, J. Zhang, S. Zhu, C. Zhong, and Y. Zhang. "Deep Multiscale Siamese Network with Parallel Convolutional Structure and Self-Attention for Change Detection" TGRS, 2021
S. Fang, K. Li, J. Shao and Z. Li, “SNUNet-CD: A Densely Connected Siamese Network for Change Detection of VHR Images,” IEEE Geosci. Remote Sens. Lett., 2021.

we attempt to illustrate the whole process to make the reader understood.
"""
import datetime
import torch
import os
import logging
import random
import warnings
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support as prfs
from utils.parser import get_parser_with_args
from utils.Related import (get_loaders, get_criterion,
                           load_model, initialize_metrics, get_mean_metrics,
                           set_metrics)
from tensorboardX import SummaryWriter

warnings.filterwarnings("ignore")

"""Initialize Parser and define arguments"""
parser, metadata = get_parser_with_args()
opt = parser.parse_args()

"""Initialize experiments log"""
logging.basicConfig(level=logging.INFO)
writer = SummaryWriter(opt.log_dir + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/')

"""Set up environment: define paths, download data, and set device"""
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logging.info('GPU AVAILABLE? ' + str(torch.cuda.is_available()))
# ensuring the same value after shuffle
def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
seed_torch(seed=777)


if __name__ == '__main__':
    # training setting for model, loss, optimizer and learning rate
    train_loader, val_loader = get_loaders(opt)
    logging.info('Model LOADING')
    model = load_model(opt, dev)
    criterion = get_criterion(opt)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate) # Be careful when you adjust learning rate, you can refer to the linear scaling rule
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)
    best_metrics = {'cd_f1scores': -1, 'cd_recalls': -1, 'cd_precisions': -1}
    logging.info('Training Start')
    total_step = -1
    """ Training process"""
    for epoch in range(opt.epochs):
        train_metrics = initialize_metrics()
        val_metrics = initialize_metrics()
        model.train()
        logging.info('MSPSNet model training!!!')
        batch_iter = 0
        tbar = tqdm(train_loader)
        # load training dataset
        for batch_img1, batch_img2, labels in tbar:
            tbar.set_description("epoch {} info ".format(epoch) + str(batch_iter) + " - " + str(batch_iter+opt.batch_size))
            batch_iter = batch_iter+opt.batch_size
            total_step += 1

            batch_img1 = batch_img1.float().to(dev)
            batch_img2 = batch_img2.float().to(dev)
            labels = labels.long().to(dev)
            optimizer.zero_grad()

            cd_preds = model(batch_img1, batch_img2)
            cd_loss = criterion(cd_preds, labels)

            loss = cd_loss
            loss.backward()
            optimizer.step()

            cd_preds = cd_preds[-1]
            _, cd_preds = torch.max(cd_preds, 1)

            cd_corrects = (100 *
                           (cd_preds.squeeze().byte() == labels.squeeze().byte()).sum() /
                           (labels.size()[0] * (opt.patch_size**2)))

            cd_train_report = prfs(labels.data.cpu().numpy().flatten(),
                                   cd_preds.data.cpu().numpy().flatten(),
                                   average='binary',
                                   pos_label=1, zero_division=1)

            train_metrics = set_metrics(train_metrics,
                                        cd_loss,
                                        cd_corrects,
                                        cd_train_report,
                                        scheduler.get_last_lr())

            mean_train_metrics = get_mean_metrics(train_metrics)

            for k, v in mean_train_metrics.items():
                writer.add_scalars(str(k), {'train': v}, total_step)

            del batch_img1, batch_img2, labels

        scheduler.step()
        logging.info("EPOCH {} TRAIN METRICS".format(epoch) + str(mean_train_metrics))

        """Validation process"""
        model.eval()
        with torch.no_grad():
            for batch_img1, batch_img2, labels in val_loader:
                # Set variables for training
                batch_img1 = batch_img1.float().to(dev)
                batch_img2 = batch_img2.float().to(dev)
                labels = labels.long().to(dev)

                cd_preds = model(batch_img1, batch_img2)

                cd_loss = criterion(cd_preds, labels)

                cd_preds = cd_preds[-1]
                _, cd_preds = torch.max(cd_preds, 1)

                cd_corrects = (100 *
                               (cd_preds.squeeze().byte() == labels.squeeze().byte()).sum() /
                               (labels.size()[0] * (opt.patch_size**2)))

                cd_val_report = prfs(labels.data.cpu().numpy().flatten(),
                                     cd_preds.data.cpu().numpy().flatten(),
                                     average='binary',
                                     pos_label=1,zero_division=1)

                val_metrics = set_metrics(val_metrics,
                                          cd_loss,
                                          cd_corrects,
                                          cd_val_report,
                                          scheduler.get_last_lr())

                # log the batch mean metrics
                mean_val_metrics = get_mean_metrics(val_metrics)

                for k, v in mean_train_metrics.items():
                    writer.add_scalars(str(k), {'val': v}, total_step)

                # clear batch variables from memory
                del batch_img1, batch_img2, labels

            logging.info("EPOCH {} VALIDATION METRICS".format(epoch)+str(mean_val_metrics))

            """
            Store the weights of good epochs based on validation results
            """
            if ((mean_val_metrics['cd_precisions'] > best_metrics['cd_precisions'])
                    or
                    (mean_val_metrics['cd_recalls'] > best_metrics['cd_recalls'])
                    or
                    (mean_val_metrics['cd_f1scores'] > best_metrics['cd_f1scores'])):

                # Insert training and epoch information to metadata dictionary
                logging.info('updata the model')
                metadata['validation_metrics'] = mean_val_metrics

                # Save model and log
                if not os.path.exists('./tmp'):
                    os.mkdir('./tmp')

                torch.save(model, './tmp/epoch_'+str(epoch)+'.pt')

                # comet.log_asset(upload_metadata_file_path)
                best_metrics = mean_val_metrics


            print('An epoch finished.')
    writer.close()  # close tensor board
    print('Done!')
