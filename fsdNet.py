import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb, csv
import sys, time, numpy, os, subprocess, pandas, tqdm
from sklearn import metrics

from loss import lossAV, lossA, lossV
from model.fsd import FSD

class FSDNet(nn.Module):
    def __init__(self, lr = 0.0001, lrDecay = 0.95, **kwargs):
        super(FSDNet, self).__init__()        
        self.model = FSD().cuda()
        self.lossAV = lossAV().cuda()
        self.lossA = lossA().cuda()
        self.lossV = lossV().cuda()
        self.optim = torch.optim.Adam(self.parameters(), lr = lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size = 1, gamma=lrDecay)
        print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f"%(sum(param.numel() for param in self.model.parameters()) / 1024 / 1024))

    def train_network(self, loader, epoch, **kwargs):
        self.train()
        self.scheduler.step(epoch)
        index, top1, loss = 0, 0, 0
        lr = self.optim.param_groups[0]['lr']        
        for num, (audioFeature, visualFeature, labels, audio_labels) in enumerate(loader, start=1):
            self.zero_grad()
            pdb.set_trace()
            audioEmbed = self.model.forward_audio_frontend(audioFeature[0].cuda()) # feedForward
            visualEmbed = self.model.forward_visual_frontend(visualFeature[0].cuda())
            outsAV= self.model.forward_audio_visual_backend(audioEmbed, visualEmbed)  
            outsA = self.model.forward_audio_backend(audioEmbed)
            outsV = self.model.forward_visual_backend(visualEmbed)
            labels = labels[0].reshape((-1)).cuda() # Loss
            audio_labels = audio_labels[0].reshape((-1)).cuda() # Loss

            nlossAV, _, _, prec = self.lossAV.forward(outsAV, labels)
            nlossA = self.lossA.forward(outsA, labels)
            nlossV = self.lossV.forward(outsV, labels)
            nloss = nlossAV + 0.5 * nlossA + 0.5 * nlossV
            loss += nloss.detach().cpu().numpy()
            top1 += prec
            nloss.backward()
            self.optim.step()
            index += len(labels)
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
            " [%2d] Lr: %5f, Training: %.2f%%, "    %(epoch, lr, 100 * (num / loader.__len__())) + \
            " Loss: %.5f, ACC: %2.2f%% \r"        %(loss/(num), 100 * (top1/index)))
            sys.stderr.flush()  
        sys.stdout.write("\n")      
        return loss/num, lr

    def evaluate_network(self, loader, evalCsvSave, evalOrig, **kwargs):
        self.eval()
        predScores = []
        total_time = 0
        labels_num = []
        for audioFeature, visualFeature, labels, audio_labels in tqdm.tqdm(loader):
            with torch.no_grad():                
                t1 = time.time()
                audioEmbed  = self.model.forward_audio_frontend(audioFeature[0].cuda())
                visualEmbed = self.model.forward_visual_frontend(visualFeature[0].cuda())
                outsAV= self.model.forward_audio_visual_backend(audioEmbed, visualEmbed)  
                labels = labels[0].reshape((-1)).cuda()             
                _, predScore, _, _ = self.lossAV.forward(outsAV, labels)    
                t2 = time.time()

                total_time = total_time + t2 - t1
                predScore = predScore[:,1].detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()
                labels_num.extend(labels)
                predScores.extend(predScore)

        auc = self.calcAUC(predScores, labels_num)
        evalLines = open(evalOrig).read().splitlines()[1:]
        labels = []
        labels = pandas.Series( ['SPEAKING_AUDIBLE' for line in evalLines])
        scores = pandas.Series(predScores)
        evalRes = pandas.read_csv(evalOrig)
        evalRes['score'] = scores
        evalRes['label'] = labels
        evalRes.drop(['label_id'], axis=1,inplace=True)
        evalRes.drop(['instance_id'], axis=1,inplace=True)
        evalRes.to_csv(evalCsvSave, index=False)
        cmd = "python -O utils/get_ava_active_speaker_performance.py -g %s -p %s "%(evalOrig, evalCsvSave)
        mAP = float(str(subprocess.run(cmd, shell=True, capture_output =True).stdout).split(' ')[2][:5])
        return mAP,auc,total_time

    def saveParameters(self, path):
        torch.save(self.state_dict(), path)

    def loadParameters(self, path):
        selfState = self.state_dict()
        loadedState = torch.load(path)
        for name, param in loadedState.items():
            origName = name;
            if name not in selfState:
                name = name.replace("module.", "")
                if name not in selfState:
                    print("%s is not in the model."%origName)
                    continue
            if selfState[name].size() != loadedState[origName].size():
                sys.stderr.write("Wrong parameter length: %s, model: %s, loaded: %s"%(origName, selfState[name].size(), loadedState[origName].size()))
                continue
            selfState[name].copy_(param)

    
    def calcAUC(self, scores, labels):
        fpr_unsorted,tpr_unsorted,thresholds_unsorted=metrics.roc_curve(labels,scores)
        aucvalue_unsorted=metrics.auc(fpr_unsorted,tpr_unsorted)
        return aucvalue_unsorted
