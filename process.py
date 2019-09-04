
import os
if 'Y' in input('force CPU?').upper():
  os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from keras.applications.vgg19  import VGG19 as Network
from keras.applications.vgg19  import preprocess_input, decode_predictions
from keras.models import Model

import subprocess as sp
import numpy as np
import cv2

import mimetypes

import random
import pathlib
import json

import shutil

MIN_VID_WIDTH   = 0
MIN_VID_HEIGHT  = 0

SEARCH_ROOT       = 'Z:\\Video'
REQUIED_SUBSTRING = '1080'

BATCH_SIZE      = 2000

BUFSIZE      = 10**5
SCALE_ALGO   = 'bilinear'
PIXEL_FORMAT = 'bgr24'
RATE         = 1
SAVE_DIR     = 'ScannedFiles'

os.path.exists(SAVE_DIR) or os.mkdir(SAVE_DIR)

model = Network(weights='imagenet',include_top=True)
model.summary()
model = Model(inputs=model.input, outputs=model.get_layer("fc2").output)


MODLE_INPUT_DIMS = model.input.shape[1],model.input.shape[2]

fileList = []
for root,dirs,files in os.walk(SEARCH_ROOT):
  for f in files:
    type_guess = mimetypes.guess_type(f)
    filePath = os.path.join(root,f)
    if type_guess is not None and type_guess[0] is not None and 'video' in type_guess[0] and REQUIED_SUBSTRING in filePath :
      
      fileList.append(filePath)
      print('Found',len(fileList),'video files')

random.shuffle(fileList)

tempFL=[]

for f in fileList:
  c = input(f+'Add/Skip/Run (nothing to run all)?').upper()
  if c=='R':
    break
  elif c=='A':
    tempFL.append(f)
  elif c=='S':
    pass
  else:
    tempFL=fileList
    break


for videoFileName in tempFL:
  
  safeName = pathlib.Path(videoFileName).as_posix().replace('/','.').replace(':','.').replace('..','.')
  print(videoFileName)
  print(safeName)


  metadataPath = os.path.join(SAVE_DIR,safeName,'MetaData')
  if os.path.exists(os.path.join(metadataPath,'meta.json')):
    continue

  popen_params = {"bufsize": BUFSIZE,
                  "stdout": sp.PIPE}

  procInfo = sp.Popen(['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', videoFileName], **popen_params)

  procInfo = json.loads( procInfo.stdout.read() )
  procInfo['srcFile']=videoFileName

  totalExpectedFrames=None
  totalDuration=None
  frameRate=None

  for stream in procInfo['streams']:
    
    if 'duration' in stream and totalDuration is None:
      totalDuration=float(stream['duration'])

    if 'nb_frames' in stream and totalExpectedFrames is None:
      totalExpectedFrames=int(stream['nb_frames'])

    if 'r_frame_rate' in stream and frameRate is None:
      frameRate=stream['r_frame_rate']


  if frameRate is None or totalDuration is None or totalExpectedFrames is None:
    continue

  os.path.exists(SAVE_DIR) or os.mkdir(SAVE_DIR)

  outPath = os.path.join(SAVE_DIR,safeName)
  os.path.exists(outPath) or os.mkdir(outPath)

  imagePath = os.path.join(SAVE_DIR,safeName,'Frames')
  os.path.exists(imagePath) or os.mkdir(imagePath)

  
  os.path.exists(metadataPath) or os.mkdir(metadataPath)

  dom,num=frameRate.split('/')
  frameRate=float(dom)/float(num)

  procInfo['totalDuration']=totalDuration
  procInfo['totalExpectedFrames']=totalExpectedFrames
  procInfo['frameRate']=frameRate

  print(procInfo)

  print(totalExpectedFrames,totalDuration,frameRate)

  cmd = (['ffmpeg'] + ['-i', videoFileName] +
        ['-loglevel',  'error',
         '-f',         'image2pipe',
         '-vf',        'scale=%d:%d' % MODLE_INPUT_DIMS,
         '-sws_flags',  SCALE_ALGO,
         "-pix_fmt",    PIXEL_FORMAT,
         '-vcodec',    'rawvideo', '-'])
          
  popen_params = {"bufsize": BUFSIZE,
                  "stdout": sp.PIPE}

  print(' '.join(cmd) )
  proc   = sp.Popen(cmd, **popen_params)
  nbytes = 3 * MODLE_INPUT_DIMS[0] * MODLE_INPUT_DIMS[1] 

  frameNum=-1
  features = []
  lastFrame=None
  total=0
  passed=0
  last_feat=None
  features=[]
  frameNums=[]
  skipsize=0
  maxSkipsize = 0
  maxSkipLimit=int(frameRate*4)
  while 1:
    frameNum += 1
    timeStamp = frameNum*(1/RATE)
    s = proc.stdout.read(nbytes)
    if len(s) == nbytes:

      result = np.frombuffer(s, dtype='uint8')
      result.shape = MODLE_INPUT_DIMS + (-1,)

      total+=1
      x = np.array([result])
      x = preprocess_input(x)
      pred_features = model.predict(x).flatten()

      if last_feat is not None:
        scores = np.square(pred_features-last_feat).mean()
        
        if scores>0.3 or skipsize>maxSkipLimit:  
          cv2.imshow('f',result)
          last_feat=pred_features
          passed+=1
          cv2.imwrite(os.path.join(imagePath,'{:0>8d}.png'.format(frameNum)),result,[cv2.IMWRITE_PNG_COMPRESSION, 9])
          features.append(pred_features)
          frameNums.append(frameNum)
          skipsize=0
        else:
          cv2.imshow('s',result)
          skipsize+=1
      else:
        passed+=1
        features.append(pred_features)
        frameNums.append(frameNum)

      maxSkipsize=max(skipsize,maxSkipsize)
      print('total:',total,'Passed:',str(int((passed/total)*100))+'%','features:',len(features),'skipsize:',skipsize,'maxSkip:',maxSkipsize,'remain:',round((frameNum/totalExpectedFrames)*100,3),'%')
      cv2.waitKey(1)
      if last_feat is None:
        last_feat=pred_features

    else:
      break
  
  featuresFilename = os.path.join(metadataPath,'features')
  np.savez(featuresFilename,np.array(features))
  frameNumsFileName = os.path.join(metadataPath,'frameNums')
  np.savez(frameNumsFileName,np.array(frameNums))
  
  metaFilename = os.path.join(metadataPath,'meta.json')
  with open(metaFilename,'w') as metaFile:
    metaFile.write(json.dumps(procInfo))
