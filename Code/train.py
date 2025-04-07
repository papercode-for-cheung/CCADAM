import csv
import os.path
import pickle
import torch
from sklearn.preprocessing import label_binarize

from dataset import read_pickle,mydatalist,prepare_twosides,twosides_pkl_loader,load_pkl
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score, precision_score, f1_score,recall_score
from sklearn.metrics import precision_recall_curve,auc,accuracy_score,average_precision_score
import numpy as np
from tqdm import tqdm
from util import log_util

from torch import optim
from DDI import DD_Pre
import time
import faulthandler
import warnings
warnings.filterwarnings("ignore")
faulthandler.enable()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_ids = range(torch.cuda.device_count())
torch.multiprocessing.set_sharing_strategy('file_system')




def test_DDI(test_loader, model):
    model.eval()
    y_pred = []
    y_label = []
    with torch.no_grad():
        bar = tqdm(test_loader, ncols=80)
        for i,batches in enumerate(bar):

            head_list, tail_list, rel_list, Label = [data.to(device) for data in batches]
            predictions,contrastive_loss,contrastive_logit = model(epoch+1,head_list, tail_list, rel_list, False)
            predictions = predictions.squeeze()

            predictions = torch.sigmoid(predictions)

            predictions = predictions.detach().cpu().numpy()
            Label = Label.detach().cpu().numpy()
            y_label = y_label + Label.flatten().tolist()
            y_pred = y_pred + predictions.flatten().tolist()
    y_pred1 = np.array(y_pred)
    y_label1 = np.array(y_label)
    y_pred1_label = (y_pred1>=0.5).astype(np.int32)
    roc_test_ACC,roc_test_AUROC,f1,roc_test_Pre,recall,roc_test_AUPR = accuracy_score(y_label1,y_pred1_label),roc_auc_score(y_label, y_pred1),f1_score(y_label1,y_pred1_label), precision_score(y_label1, y_pred1_label),recall_score(y_label1, y_pred1_label),average_precision_score(y_label1,y_pred1,average='micro')
    p, r, t = precision_recall_curve(y_label1, y_pred1)
    roc_test_AUC = auc(r, p)

    return  roc_test_ACC,roc_test_AUC,f1,roc_test_Pre,recall,roc_test_AUPR,roc_test_AUROC
if __name__ == '__main__':
    dataset = 'twosides'
    logs = log_util(dataset,'0.5')
    for fold in [1]:

        train_dir = './txt2/Durg/train{}.txt'.format(fold)
        test_dir = './txt2/Durg/test{}.txt'.format(fold)
        train_data = prepare_twosides(train_dir)
        test_data = prepare_twosides(test_dir)

        if torch.cuda.is_available():

            model = DD_Pre(56,0.7,0.7).cuda()

        train_loader = twosides_pkl_loader(train_data,batch_size=128,shuffle=True,num_workers=2,pin_memory = True)
        test_loader = twosides_pkl_loader(test_data,batch_size=256,shuffle=False,num_workers=0,pin_memory = True)
        # train_loader = twosides_pkl_loader(cold_train_data,batch_size=2048,shuffle=True,num_workers=2,pin_memory = True)
        #
        # c2 = twosides_pkl_loader(cold_c2_data,batch_size = 1024,shuffle=False,num_workers=2)
        #
        # c3 = twosides_pkl_loader(cold_c3_data,batch_size = 1024,shuffle=False,num_workers=2)
        print(model.parameters())
        optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.99))

        loss_history = []

        t_total=time.time()

        epochs=80

        for epoch in range(0,epochs):

            model.train()
            t = time.time()
            y_pred_train = []
            y_label_train = []
            bar =tqdm(train_loader,ncols=80)
            total_loss = 0
            batch = 0
            for i, batches in enumerate(bar):
                bar.set_description('Epoch ' + str(epoch))

                head_list, tail_list, rel_list,Label = [data.to(device) for data in batches]
                predictions,contrastive_loss,contrastive_logit = model(epoch,head_list, tail_list, rel_list,True)
                #predictions = model(epoch,head_list, tail_list, rel_list, True)
                print("logit shape:", predictions.shape)
                print("Label shape:", Label.shape)

                predictions = predictions.squeeze()
                #contrastive_loss=contrastive_loss.squeeze()
                loss1 = torch.nn.BCEWithLogitsLoss(reduction='sum')(predictions, Label)
                loss = loss1+0.5*contrastive_loss
                optimizer.zero_grad()
                #loss1.backward()
                loss.backward()
                optimizer.step()

                #predictions=contrastive_logit.squeeze()
                predictions = torch.sigmoid(predictions)
                predictions = predictions.detach().cpu().numpy()
                Label = Label.detach().cpu().numpy()
                y_label_train = y_label_train + Label.flatten().tolist()
                y_pred_train = y_pred_train + predictions.flatten().tolist()
                total_loss += loss.item()
                batch = len(y_label_train)
                bar.set_postfix(loss ='%.5f' %(total_loss/batch))
                # Our model requires high computing power and a long time. We can run the demonstration here. Please see the paper for specific results. If you want to run it completely, please modify the batchsize and comment the next line.
                break
            y_pred_train = np.array(y_pred_train)
            y_pred_train_label = (y_pred_train>=0.5).astype(np.int32)
            y_label_train = np.array(y_label_train)
            roc_train_ACC, roc_train_AUROC, train_f1, roc_train_Pre, train_recall, roc_train_AUPR = accuracy_score(y_label_train,y_pred_train_label),roc_auc_score(y_label_train, y_pred_train),f1_score(y_label_train,y_pred_train_label), precision_score(y_label_train, y_pred_train_label),recall_score(y_label_train, y_pred_train_label),average_precision_score(y_label_train,y_pred_train,average='micro')
            p,r,t = precision_recall_curve(y_label_train,y_pred_train)
            roc_train_AUC = auc(r,p)

            print(roc_train_AUC)
            logs.save_log(epoch,roc_train_ACC, roc_train_AUC, train_f1, roc_train_Pre, train_recall, roc_train_AUPR,roc_train_AUROC,'train',model,optimizer)
            # It is recommended to enable it during local training
            #if epoch % 2 == 0:
            #    roc_test_ACC, roc_test_AUC, f1, roc_test_Pre, recall, roc_test_AUPR,roc_test_AUROC = test_DDI(test_loader, model)
            #    logs.save_log(epoch,roc_test_ACC,roc_test_AUC,f1,roc_test_Pre,recall,roc_test_AUPR,roc_test_AUROC,'test',model,optimizer)           
            #   print(roc_test_Pre)

        roc_test_ACC, roc_test_AUC, f1, roc_test_Pre, recall, roc_test_AUPR, roc_test_AUROC = test_DDI(test_loader,model)
        logs.save_log(0, roc_test_ACC, roc_test_AUC, f1, roc_test_Pre, recall, roc_test_AUPR, roc_test_AUROC,
                      'test', model, optimizer)
        #torch.save(model.state_dict(),
        #       '../results/twosides/{}_checkpoint.pt'.format('last'))


class CustomData(Data):

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'line_graph_edge_index':
            return self.edge_index.size(1) if self.edge_index.nelement()!=0 else 0
        return super().__inc__(key, value, *args, **kwargs)


