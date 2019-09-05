

from skimage.measure import compare_ssim
from tensorflow import float32

import os
import glob
import numpy as np
import json
import random
import cv2
import math
import collections
import subprocess as sp
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

import tensorflow as tf
tf.enable_eager_execution()

MIN_DIST  = 60*10
SKIP_DIST = 1


CMP = 'L2'

CMP = 'MSE'
CMP = 'COS'

scannedFiles={}

allFeatures = None
allFrameTimes = None
fileOffsets = {}



IndexDetails = collections.namedtuple('IndexDetails', 'indexName index frameIndex timeStamp originalFileName ')
def getDataAtIndex(featureIndex):
  
  indexDetails = IndexDetails()
  indexDetails.index=featureIndex
  indexDetails.frameIndex=allFrameTimes[featureIndex]
  
  keyName=None
  for (a,b),name in fileOffsets.items():
    if a<featureIndex<b:
      keyName=name
      break
  
  indexDetails.indexName=keyName
  indexDetails.timeStamp=frameIndex/scannedFiles[sourceClipName]['metaData']['frameRate']
  return indexDetails


fl=list(glob.glob('ScannedFiles\\*'))
random.shuffle(fl)
for scanFolder in fl:
  frameTimesFileName = os.path.join(scanFolder,'MetaData','frameNums.npz')
  featuresFileName = os.path.join(scanFolder,'MetaData','features.npz')
  metadataFileName = os.path.join(scanFolder,'MetaData','meta.json')
  if os.path.exists(featuresFileName) and os.path.exists(metadataFileName):
    metaData  = json.loads(open(metadataFileName,'r').read())
    npArr     = np.load(featuresFileName)
    npTimeArr = np.load(frameTimesFileName)
    
    if allFeatures is None:
      allFeatures = npArr['arr_0']
      fileOffsets[(0,allFeatures.shape[0])] = scanFolder
    else:
      initial = allFeatures.shape[0]
      allFeatures = np.concatenate((allFeatures,npArr['arr_0']), axis=0) 
      fileOffsets[(initial,allFeatures.shape[0])] = scanFolder

    if allFrameTimes is None:
      allFrameTimes = npTimeArr['arr_0']
    else:
      allFrameTimes = np.concatenate((allFrameTimes,npTimeArr['arr_0']), axis=0) 

    scannedFiles[scanFolder] = {'metaData':metaData}
    
    print(scanFolder,'loaded')

print(allFeatures.shape,allFrameTimes.shape)

if False:
  pca = PCA(n_components=300)
  pca.fit(allFeatures)
  allFeatures=pca.transform(allFeatures)

mx,my = None,None
cx,cy = None,None

imgdim=224

def click(event, x, y, flags, param):
  global mx,my,cx,cy
  mx = math.floor(x/imgdim)
  my  = math.floor(y/imgdim)
  if event==1:
    cx = mx
    cy = my

cv2.namedWindow("matches")
cv2.setMouseCallback("matches", click)

index=None
addSkip=True
cnt=0
seq=[]
microSeek=0.0
flip=False
lastflip=False
while 1:
  microSeek=0.0
  flip=False
  if index is None:
    sourceInd      = random.randint(0,allFeatures.shape[0]-1)
    tindex=sourceInd
    if len(seq)==0:
      seq=[(sourceInd,None,None,False)]
      sourceInd+=1

      tname=None
      for (a,b),name in fileOffsets.items():
        if a<sourceInd<b:
          tname=name
          break

      while abs(allFrameTimes[sourceInd]-allFrameTimes[tindex])/scannedFiles[tname]['metaData']['frameRate'] <= SKIP_DIST:
        sourceInd+=1



  else:
    sourceInd      = index

    tname=None
    for (a,b),name in fileOffsets.items():
      if a<sourceInd<b:
        tname=name
        break

    if addSkip:
      while abs(allFrameTimes[sourceInd]-allFrameTimes[index])/scannedFiles[tname]['metaData']['frameRate'] <= SKIP_DIST:
        sourceInd+=1
    addSkip=True
    index          = None

  if len(seq)>1:
    lastflip=seq[-1][3]==True

  sourceIndFnum=0
  for (a,b),name in fileOffsets.items():
    if a<sourceInd<b:
      sourceClipName=name
      sourceIndFnum = sourceInd-a
      break

  print(sourceClipName,sourceInd)
  sourceFeature  = allFeatures[sourceInd]

  matches = []
  allScores=[]

  if CMP=='L2':
    scores = np.linalg.norm(sourceFeature-allFeatures,axis=1)*10000
  elif CMP=='MSE':
    scores = np.square(sourceFeature-allFeatures).mean(axis=1)
  elif CMP=='COS':
    den = np.sqrt(np.einsum('ij,ij->i',allFeatures,allFeatures)*np.einsum('j,j',sourceFeature,sourceFeature))
    scores = 1-(allFeatures.dot(sourceFeature) / den)

  for n,s in enumerate(scores):
    destFile = None
    for (a,b),name in fileOffsets.items():
      if a<n<b and name!=sourceClipName :
        allScores.append( (s,name, allFrameTimes[n],n) )
        break

  allScores=sorted(allScores)

  rows=6
  cols=11

  total = (rows*cols)+1

  tempScores = []
  seen=set([x[2] for x in seq if x[2] is not None ])
  for s,f,x,i in allScores:
    close=False
    for v in seen:
      if abs(allFrameTimes[v]-allFrameTimes[i])<10*60:
        close=True
        break
    if not close:
      tempScores.append((s,f,x,i))
      seen.add(i)
    if len(tempScores)>total:
      break

  compare_ssim

  orig = cv2.imread(sourceClipName+'\\Frames\\{:0>8d}.png'.format( allFrameTimes[sourceInd] ))
  orig = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
  origf = cv2.flip( orig ,1)
  def scmp(score):
    s,f,x,i = score
    print(x)
    tst = cv2.cvtColor(cv2.imread(f+'\\Frames\\{:0>8d}.png'.format(x)), cv2.COLOR_BGR2GRAY)
    return max( compare_ssim(orig,tst),compare_ssim(origf,tst) )


  allScores=tempScores

  #allScores = sorted(allScores,key=scmp,reverse=True)

  print(len(allScores))

  if lastflip:
    previews = [ [cv2.flip( cv2.resize(cv2.imread(sourceClipName+'\\Frames\\{:0>8d}.png'.format( allFrameTimes[sourceInd] )),(imgdim,imgdim),interpolation=cv2.INTER_CUBIC ),1) ] ]
  else:
    previews = [ [cv2.resize(cv2.imread(sourceClipName+'\\Frames\\{:0>8d}.png'.format( allFrameTimes[sourceInd] )),(imgdim,imgdim),interpolation=cv2.INTER_CUBIC )] ]
    



  for s,f,x,i in allScores:
    if len(previews)>=rows and len(previews[-1])>=cols:
      previews[-1]=np.hstack(previews[-1])
      break
    if len(previews[-1])>=cols:

      previews[-1]=np.hstack(previews[-1])
      previews.append([])

    previews[-1].append(
      cv2.resize(cv2.imread(f+'\\Frames\\{:0>8d}.png'.format(x)),(imgdim,imgdim),interpolation=cv2.INTER_CUBIC)
    )


  if type( previews[-1] ) == list:
    while len(previews[-1])<cols:
      previews[-1].append(np.zeros( (imgdim,imgdim,3)).astype(np.uint8))
    previews[-1]=np.hstack(previews[-1])


  matchImg =  np.vstack(previews+[np.zeros((30,previews[0].shape[1],3),np.uint8)])



  while 1:
    outMatchImg = matchImg.copy()

    message = 'SEQLEN:{sl} CURLEN:{cl}+({ms})s flip={flp} - Q=quit W=write E=eraseSeq A=seekBack D=feekFwd '.format(
                                             ms=microSeek,
                                             sl=len(seq),
                                             flp=flip,
                                             cl=((allFrameTimes[sourceInd]-allFrameTimes[seq[-1][0]])/scannedFiles[sourceClipName]['metaData']['frameRate']) + microSeek
                                             )

    cv2.putText(outMatchImg,message,(0,matchImg.shape[0]-5), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),1,cv2.LINE_AA)

    if mx is not None:
      outMatchImg = cv2.rectangle(outMatchImg,(imgdim*mx,imgdim*my),((imgdim*mx)+imgdim,(imgdim*my)+imgdim),(0,255,0),1)

    if cx is not None:

      index = cy*11+cx
      if index>0 and index<len(allScores)+1:
        print('scoreIndex',index)
        index = allScores[index-1][3]

        try:
          seqTail,(seqHead_s,msk,seqHead_e,flp) = seq[:-1],seq[-1]
        except:
          seqTail=[]
          seqHead_s,seqHead_e=None,None
        print(seqTail,seqHead_s,seqHead_e)

        seq = seqTail+[(seqHead_s,microSeek,sourceInd,flp),(index,0.0,None,flip)]
        print(seq)
        print('frameIndex',index)
        cx,xy = None,None
        break
      else:
        index=None
        cx,xy = None,None

    cv2.imshow('matches', outMatchImg)
    keyP= cv2.waitKey(1)
    
    if keyP==ord('w') and len(seq)>1:
      cv2.destroyAllWindows() 
      cutSequence=[]
      for s,ms,e,flp in seq:
        if s is None and e is not None:
          s=e-1
        elif s is not None and e is None:
          e=s+1

        s_filename  = None
        s_timestamp = None

        for (a,b),name in fileOffsets.items():
          if a<s<b:
            s_filename=name
        s_timestamp = allFrameTimes[s]

        e_filename  = None
        e_timestamp = None
        for (a,b),name in fileOffsets.items():
          if a<e<b:
            e_filename=name
        e_timestamp = allFrameTimes[e]

        print(ms)
        s_timestamp=s_timestamp/scannedFiles[s_filename]['metaData']['frameRate']
        e_timestamp=e_timestamp/scannedFiles[e_filename]['metaData']['frameRate']+ms

        print('file',s_filename,':',s_timestamp,'-',e_timestamp)
        
        cutSequence.append((
                           scannedFiles[s_filename]['metaData']['srcFile'],
                           s_timestamp,e_timestamp,flp
        ))

      for fn in glob.glob('output_*.mkv'):
        try:
          os.remove(fn)
        except Exception as e:
          print(e)

      with open('fileListForConcat.txt','w') as fl:
        for num,(fn,sts,ets,flp) in enumerate(cutSequence):

          flipcmd=',hflip'
          if not flp:
            flipcmd=''
          cmd = ['ffmpeg', '-y', '-ss', str(sts) , '-t', str(ets-sts),'-i', fn, '-vf', 
                 'scale=w={w}:h={h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2{flp}'.format(w=1920,h=1080,flp=flipcmd), '-r', '24', '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '0', 'output_{i}.mkv'.format(i=num)]
          print(' '.join(cmd) )
          proc = sp.Popen(cmd,stdout=sp.DEVNULL,stderr=sp.PIPE)
          proc.communicate()
          fl.write('file output_{i}.mkv\n'.format(i=num))
      import time

      proc = sp.Popen([ "ffmpeg","-y","-r","24","-f","concat","-safe","0","-i","fileListForConcat.txt", '-c:v','libx264','-preset','veryslow','-crf','23'  ,"transition_match_{}.mkv".format(time.time())])
      proc.communicate()




      seq=[]

    if keyP==ord('f'):
      flip=not flip    
    if keyP==ord('z'):
      microSeek-=0.2
    if keyP==ord('x'):
      microSeek+=0.2
    if keyP==ord('w'):
      seq=[]
      break
    if keyP==ord('e'):
      seq=[]
      break
    elif keyP==ord('a'):
      index = sourceInd-1
      addSkip=False
      break
    elif keyP==ord('d'):
      index = sourceInd+1
      addSkip=False
      break
    elif keyP==ord('q'):
      exit()