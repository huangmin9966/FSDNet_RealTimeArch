import time, os, torch, argparse, warnings, glob
import pdb
from dataLoader import train_loader, val_loader, feature_extract_loader
from utils.tools import *
from fsdNet import FSDNet

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    # The structure of this code is learnt from https://github.com/clovaai/voxceleb_trainer
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description = "Evaluating")
    # Training details
    parser.add_argument('--lr',           type=float, default=0.0001,help='Learning rate')
    parser.add_argument('--lrDecay',      type=float, default=0.95,  help='Learning rate decay rate')
    parser.add_argument('--maxEpoch',     type=int,   default=35,    help='Maximum number of epochs')
    parser.add_argument('--testInterval', type=int,   default=1,     help='Test and save every [testInterval] epochs')
    parser.add_argument('--batchSize',    type=int,   default=1000,  help='Dynamic batch size, default is 2500 frames, other batchsize (such as 1500) will not affect the performance')
    parser.add_argument('--nDataLoaderThread', type=int, default=4,  help='Number of loader threads')
    # Data path
    parser.add_argument('--dataPathAVA',  type=str, default="./proccessed_ava_data/", help='Save path of AVA dataset')
    parser.add_argument('--savePath',     type=str, default="exps/exp8")
    # Data selection
    parser.add_argument('--evalDataType', type=str, default="val", help='Only for AVA, to choose the dataset for evaluation, val or test')
    # For download dataset only, for evaluation only
    parser.add_argument('--downloadAVA',     dest='downloadAVA', action='store_true', help='Only download AVA dataset and do related preprocess')
    parser.add_argument('--evaluation',      dest='evaluation', action='store_true', help='Only do evaluation by using pretrained model [pretrain_AVA.model]')
    parser.add_argument('--saveFeature',    dest='saveFeature', action='store_true', help='run forward ')
    args = parser.parse_args()
    # Data loader
    args = init_args(args)

    if args.downloadAVA == True:
        preprocess_AVA(args)
        quit()

    pdb.set_trace()
    loader = train_loader(trialFileName = args.trainTrialAVA, \
                          audioPath      = os.path.join(args.audioPathAVA , 'train'), \
                          visualPath     = os.path.join(args.visualPathAVA, 'train'), \
                          oriLabelPath   = "./proccessed_ava_data/csv/ava_activespeaker_train_v1.0/",\
                          **vars(args))
    trainLoader = torch.utils.data.DataLoader(loader, batch_size = 1, shuffle = True, num_workers = args.nDataLoaderThread)

    loader = val_loader(trialFileName = args.evalTrialAVA, \
                        audioPath     = os.path.join(args.audioPathAVA , args.evalDataType), \
                        visualPath    = os.path.join(args.visualPathAVA, args.evalDataType), \
                        oriLabelPath   = "./proccessed_ava_data/csv/ava_activespeaker_test_v1.0/",\
                        **vars(args))
    valLoader = torch.utils.data.DataLoader(loader, batch_size = 1, shuffle = False, num_workers = 16)
    


    if args.evaluation == True:
        download_pretrain_model_AVA()
        s = FSDNet(**vars(args))
        s.loadParameters('pretrain_AVA.model')
        print("Model %s loaded from previous state!"%('pretrain_AVA.model'))
        mAP,auc,t = s.evaluate_network(loader = valLoader, **vars(args))
        print("mAP %2.2f%% auc %f  %f s"%(mAP, auc, t))
        quit()

    # pdb.set_trace()
    modelfiles = glob.glob('%s/model_0*.model'%args.modelSavePath)
    modelfiles.sort()  
    if len(modelfiles) >= 1:
        print("Model %s loaded from previous state!"%modelfiles[-1])
        epoch = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][6:]) + 1
        s = FSDNet(epoch = epoch, **vars(args))
        s.loadParameters(modelfiles[-1])
    else:
        epoch = 1
        s = FSDNet(epoch = epoch, **vars(args))

    mAPs = []
    aucs = []
    scoreFile = open(args.scoreSavePath, "a+")

    while(1):        
        loss, lr = s.train_network(epoch = epoch, loader = trainLoader, **vars(args))
        
        if epoch % args.testInterval == 0:        
            s.saveParameters(args.modelSavePath + "/model_%04d.model"%epoch)
            mAP,auc,eval_time = s.evaluate_network(epoch = epoch, loader = valLoader, **vars(args))
            mAPs.append(mAP)
            aucs.append(auc)
            print(time.strftime("%Y-%m-%d %H:%M:%S"), "%d epoch, mAP %2.2f%%, bestmAP %2.2f%%"%(epoch, mAPs[-1], max(mAPs)))
            scoreFile.write("%d epoch, LR %f, LOSS %f, mAP %2.2f%%, bestmAP %2.2f%%, AUC %f, bestAUC %f, costs %fs\n"%(epoch, lr, loss, mAPs[-1], max(mAPs), aucs[-1], max(aucs), eval_time))
            scoreFile.flush()

        if epoch >= args.maxEpoch:
            quit()

        epoch += 1

if __name__ == '__main__':
    main()
