import os, torch, numpy, cv2, random, glob, python_speech_features
from scipy.io import wavfile
from torchvision.transforms import RandomCrop
import pdb, argparse

def generate_audio_set(dataPath, batchList):
    audioSet = {}
    for line in batchList:
        data = line.split('\t')
        videoName = data[0][:11]
        dataName = data[0]
        _, audio = wavfile.read(os.path.join(dataPath, videoName, dataName + '.wav'))
        audioSet[dataName] = audio
    return audioSet

def overlap(dataName, audio, audioSet):   
    noiseName =  random.sample(set(list(audioSet.keys())) - {dataName}, 1)[0]
    noiseAudio = audioSet[noiseName]    
    snr = [random.uniform(-5, 5)]
    if len(noiseAudio) < len(audio):
        shortage = len(audio) - len(noiseAudio)
        noiseAudio = numpy.pad(noiseAudio, (0, shortage), 'wrap')
    else:
        noiseAudio = noiseAudio[:len(audio)]
    noiseDB = 10 * numpy.log10(numpy.mean(abs(noiseAudio ** 2)) + 1e-4)
    cleanDB = 10 * numpy.log10(numpy.mean(abs(audio ** 2)) + 1e-4)
    noiseAudio = numpy.sqrt(10 ** ((cleanDB - noiseDB - snr) / 10)) * noiseAudio
    audio = audio + noiseAudio    
    return audio.astype(numpy.int16)

def slice_audio(audio, numFrames, real_frame_len, target_frame_len):
    #get frame center
    # pdb.set_trace()
    frame_center = [i * real_frame_len for i in range(numFrames)]
    if numFrames % 2 == 1:
        shift = int(audio.shape[0] / 2) - frame_center[int(numFrames / 2)]
    else:
        shift = int(audio.shape[0] / 2) + int(real_frame_len / 2) - frame_center[int(numFrames / 2)]
    frame_center = [i + shift for i in frame_center]
    frame_start = [i - int(target_frame_len / 2) for i in frame_center]
    frame_end = [i + int(target_frame_len / 2) for i in frame_center]

    #get frame
    frames = []
    for i in range(len(frame_center)):
        start = frame_start[i]
        end = frame_end[i]
        start = max(0, start)
        end = min(len(audio), end)
    
        one_frame = audio[start:end]

        if len(one_frame) < target_frame_len:
            # pdb.set_trace()
            shortage    = target_frame_len - len(one_frame)
            shortage0 = int(shortage / 2)
            shortage1 = shortage - shortage0
            one_frame     = numpy.pad(one_frame, pad_width=(shortage0, shortage1), mode = 'constant', constant_values=(0, 0))
        
        # pdb.set_trace()
        if len(one_frame) != target_frame_len: #caused by start > len(auido) or end < 0
            one_frame = numpy.zeros(target_frame_len, dtype = numpy.int16)
        frames.append(one_frame)
            
    return frames
            
            
def load_audio(data, dataPath, numFrames, audioAug, audioSet = None):
    dataName = data[0]
    fps = float(data[2])    
    audio = audioSet[dataName]    
    if audioAug == True:
        augType = random.randint(0,1)
        if augType == 1 and len(audioSet) > 1: #huangmin, audioSet must have audio(s) other than dataName itSelf to do aug
            audio = overlap(dataName, audio, audioSet)
        else:
            audio = audio
    # fps is not always 25, in order to align the visual, we modify the window and step in MFCC extraction process based on fps
    # audio = python_speech_features.mfcc(audio, 16000, numcep = 13, winlen = 0.025 * 25 / fps, winstep = 0.010 * 25 / fps)

    # pdb.set_trace()
    real_frame_len = int(16000 / fps) + 1
    target_frame_len = 640
    audio_frames = slice_audio(audio, numFrames, real_frame_len, target_frame_len)

    return audio_frames

def load_audio_label(data, orig_csvPath, visualPath,  numFrames):
    return [1]
    # pdb.set_trace()
    dataName = data[0]
    videoName = data[0][:11]
    faceFolderPath = os.path.join(visualPath, videoName, dataName)
    faceFiles = glob.glob("%s/*.jpg"%faceFolderPath)
    sortedFaceFiles = sorted(faceFiles, key=lambda data: (float(data.split('/')[-1][:-4])), reverse=False) 
    
    ori_csv_name = os.path.join(orig_csvPath, videoName + "-activespeaker.csv")
    ori_csv_lines = open(ori_csv_name).read().splitlines()
    ori_csv_lines = sorted(ori_csv_lines, key=lambda data: (float(data.split(',')[1])), reverse=False)
    
    audio_label = []
    for oneFile in sortedFaceFiles:
        names = oneFile.split('/')[-1]
        timestamp = float(names[0:-4])

        label = 0
        for line_in_csv in ori_csv_lines:
            line_in_csv = line_in_csv.split(',')
            t0 = float(line_in_csv[1])

            if t0 < timestamp - 0.005:
                continue
            elif t0 > timestamp + 0.005:
                break
            else:
                stat = line_in_csv[6]
                if stat == 'SPEAKING_AUDIBLE':
                    local_label = 1
                else:
                    local_label = 0
                label = max(label, local_label)
        
        audio_label.append(label)
    
    audio_label = numpy.array(audio_label[:numFrames])
    return audio_label

def load_visual(data, dataPath, numFrames, visualAug): 
    dataName = data[0]
    videoName = data[0][:11]
    faceFolderPath = os.path.join(dataPath, videoName, dataName)
    faceFiles = glob.glob("%s/*.jpg"%faceFolderPath)
    sortedFaceFiles = sorted(faceFiles, key=lambda data: (float(data.split('/')[-1][:-4])), reverse=False) 
    faces = []
    H = 112
    if visualAug == True:
        new = int(H*random.uniform(0.7, 1))
        x, y = numpy.random.randint(0, H - new), numpy.random.randint(0, H - new)
        M = cv2.getRotationMatrix2D((H/2,H/2), random.uniform(-15, 15), 1)
        augType = random.choice(['orig', 'flip', 'crop', 'rotate']) 
    else:
        augType = 'orig'
    for faceFile in sortedFaceFiles[:numFrames]:
        face = cv2.imread(faceFile)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (H,H))
        if augType == 'orig':
            faces.append(face)
        elif augType == 'flip':
            faces.append(cv2.flip(face, 1))
        elif augType == 'crop':
            faces.append(cv2.resize(face[y:y+new, x:x+new] , (H,H))) 
        elif augType == 'rotate':
            faces.append(cv2.warpAffine(face, M, (H,H)))
    faces = numpy.array(faces)
    return faces


def load_label(data, numFrames):
    res = []
    labels = data[3].replace('[', '').replace(']', '')
    labels = labels.split(',')
    for label in labels:
        res.append(int(label))
    res = numpy.array(res[:numFrames])
    return res

class train_loader(object):
    # def __init__(self, trialFileName, audioPath, visualPath, batchSize, **kwargs):
    def __init__(self, trialFileName, audioPath, visualPath, oriLabelPath, batchSize, **kwargs):
        self.audioPath  = audioPath
        self.oriLabelPath = oriLabelPath
        self.visualPath = visualPath
        self.miniBatch = []      
        mixLst = open(trialFileName).read().splitlines()
        # sort the training set by the length of the videos, shuffle them to make more videos in the same batch belong to different movies
        sortedMixLst = sorted(mixLst, key=lambda data: (int(data.split('\t')[1]), int(data.split('\t')[-1])), reverse=True)         
        start = 0        
        while True:
          length = int(sortedMixLst[start].split('\t')[1])
          end = min(len(sortedMixLst), start + max(int(batchSize / length), 1))
          self.miniBatch.append(sortedMixLst[start:end])
          if end == len(sortedMixLst):
              break
          start = end     


    def __getitem__(self, index):
        # pdb.set_trace()
        batchList    = self.miniBatch[index]
        numFrames   = int(batchList[-1].split('\t')[1])
        audioFeatures, visualFeatures, labels, audio_labels = [], [], [], []
        # print("batchList len:{}".format(len(batchList)))
        audioSet = generate_audio_set(self.audioPath, batchList) # load the audios in this batch to do augmentation
        for line in batchList:
            # print(line)
            # pdb.set_trace()
            data = line.split('\t')            
            audioFeatures.append(load_audio(data, self.audioPath, numFrames, audioAug = True, audioSet = audioSet))  
            # audioFeatures.append(load_audio(data, self.audioPath, numFrames, audioAug = False, audioSet = audioSet))  
            visualFeatures.append(load_visual(data, self.visualPath,numFrames, visualAug = True))
            audio_labels.append(load_audio_label(data, self.oriLabelPath,self.visualPath, numFrames))
            labels.append(load_label(data, numFrames))
        
        # pdb.set_trace()
        return torch.FloatTensor(numpy.array(audioFeatures)), \
               torch.FloatTensor(numpy.array(visualFeatures)), \
               torch.LongTensor(numpy.array(labels)),\
               torch.LongTensor(numpy.array(audio_labels))
               


    def __len__(self):
        return len(self.miniBatch)


class val_loader(object):
    def __init__(self, trialFileName, audioPath, visualPath, oriLabelPath, **kwargs):
        # pdb.set_trace()
        self.oriLabelPath = oriLabelPath
        self.audioPath  = audioPath
        self.visualPath = visualPath
        self.miniBatch = open(trialFileName).read().splitlines()

    def __getitem__(self, index):
        
        # pdb.set_trace()
        line       = [self.miniBatch[index]]
        numFrames  = int(line[0].split('\t')[1])
        # if numFrames > 25:
        #     numFrames = 20

        audioSet   = generate_audio_set(self.audioPath, line)        
        data = line[0].split('\t')

        # pdb.set_trace()
        audioFeatures, visualFeatures, labels, audio_labels = [], [], [], []

        audioFeatures = [load_audio(data, self.audioPath, numFrames, audioAug = False, audioSet = audioSet)]
        visualFeatures = [load_visual(data, self.visualPath,numFrames, visualAug = False)]
        labels = [load_label(data, numFrames)]         
        audio_labels = [load_audio_label(data, self.oriLabelPath, self.visualPath, numFrames)]
        return torch.FloatTensor(numpy.array(audioFeatures)), \
               torch.FloatTensor(numpy.array(visualFeatures)), \
               torch.LongTensor(numpy.array(labels)),\
               torch.LongTensor(numpy.array(audio_labels))

    def __len__(self):
        return len(self.miniBatch)

class feature_extract_loader(object):
    def __init__(self, trialFileName, audioPath, visualPath, oriLabelPath, **kwargs):
        self.oriLabelPath = oriLabelPath
        self.audioPath  = audioPath
        self.visualPath = visualPath
        self.miniBatch = open(trialFileName).read().splitlines()
        
    def get_frame_info(self, data): 
        dataName = data[0]
        videoName = data[0][:11]
        faceFolderPath = os.path.join(self.visualPath, videoName, dataName)
        faceFiles = glob.glob("%s/*.jpg"%faceFolderPath)
        sortedFaceFiles = sorted(faceFiles, key=lambda data: (float(data.split('/')[-1][:-4])), reverse=False) 
        
        return sortedFaceFiles

    def __getitem__(self, index):

        line       = [self.miniBatch[index]]
        numFrames  = int(line[0].split('\t')[1])

        audioSet   = generate_audio_set(self.audioPath, line)        
        data = line[0].split('\t')

        # pdb.set_trace()
        audioFeatures, visualFeatures, labels, audio_labels = [], [], [], []

        audioFeatures = [load_audio(data, self.audioPath, numFrames, audioAug = False, audioSet = audioSet)]
        visualFeatures = [load_visual(data, self.visualPath,numFrames, visualAug = False)]
        labels = [load_label(data, numFrames)]         
        frame_info = self.get_frame_info(data)
        return torch.FloatTensor(numpy.array(audioFeatures)), \
               torch.FloatTensor(numpy.array(visualFeatures)), \
               torch.LongTensor(numpy.array(labels)),\
               frame_info



    def __len__(self):
        return len(self.miniBatch)



## 2022-06-22 18:48:06 HuangMin@zjlab. 
if __name__=="__main__":
    parser = argparse.ArgumentParser(description = "rrrr")
    parser.add_argument('--batchSize',    type=int,   default=2500,  help='Dynamic batch size, default is 2500 frames, other batchsize (such as 1500) will not affect the performance')
    parser.add_argument('--nDataLoaderThread', type=int, default=4,  help='Number of loader threads')
    parser.add_argument('--dataPathAVA',  type=str, default="/media/huangmin/5c8bd230-cff7-4993-a013-c950f49ee160/home/huangmin/Pictures/AVA_processed_data/", help='Save path of AVA dataset')
    parser.add_argument('--savePath',     type=str, default="exps/exp1")
    args = parser.parse_args()
    # Data loader
    # pdb.set_trace()
    # loader = train_loader(trialFileName = "/home/cdev/Pictures/AVA_DataSet/csv/train_loader.csv", \
    #                       audioPath      = "/home/cdev/Pictures/AVA_DataSet/clips_audios/train", \
    #                       visualPath     = "/home/cdev/Pictures/AVA_DataSet/clips_videos/train", \
                          # oriLabelPath   = "/home/cdev/Pictures/AVA_DataSet/ava_activespeaker_train_v1.0/",\
    #                       **vars(args))
    
    # pdb.set_trace()
    # for i in range(loader.__len__()):
    #     print("{} in {}\n".format(i, loader.__len__()))
    #     video, audio, label, audio_label = loader.__getitem__(i)

    print("val..")
    loader = feature_extract_loader(trialFileName = "/media/huangmin/5c8bd230-cff7-4993-a013-c950f49ee160/home/huangmin/Pictures/AVA_processed_data/csv/val_loader.csv", \
                          audioPath      = "/media/huangmin/5c8bd230-cff7-4993-a013-c950f49ee160/home/huangmin/Pictures/AVA_processed_data/clips_audios/val", \
                          visualPath     = "/media/huangmin/5c8bd230-cff7-4993-a013-c950f49ee160/home/huangmin/Pictures/AVA_processed_data/clips_videos/val", \
                          oriLabelPath   = "/home/huangmin/Pictures/tempPicFromCdev/AVA_DataSet/ava_activespeaker_test_v1.0/",\
                          **vars(args))
    
    pdb.set_trace()
    for i in range(loader.__len__()):
        print("{} in {}\n".format(i, loader.__len__()))
        # feature0, feature1, label = loader.__getitem__(i)
        video, audio, label, audio_label, time_stamp = loader.__getitem__(i)


    print("test..")
    loader = val_loader(trialFileName = "/home/cdev/Pictures/AVA_DataSet/csv/test_loader.csv", \
                          audioPath      = "/home/cdev/Pictures/AVA_DataSet/clips_audios/test", \
                          visualPath     = "/home/cdev/Pictures/AVA_DataSet/clips_videos/test", \
                          **vars(args))
    
    pdb.set_trace()
    for i in range(loader.__len__()):
        print("{} in {}\n".format(i, loader.__len__()))
        video, audio, label, audio_label = loader.__getitem__(i)










