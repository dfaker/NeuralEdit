

import glob
import random
import numpy as np
import cv2

fl = glob.glob('ScannedFiles\\*\\Frames\\*.png')


imx,imy=224,126

rows=16
cols=18


while 1:
  selected = cv2.resize(cv2.imread(fl[random.randint(0,len(fl)-1)]),(imx,imy))
  hover    = cv2.resize(cv2.imread(fl[random.randint(0,len(fl)-1)]),(imx,imy))
  merge    = np.abs((selected.astype(np.float32)+hover.astype(np.float32))/2.0).astype(np.uint8)


  header          = np.hstack([selected,merge,hover,np.zeros((selected.shape[0],(selected.shape[1]//2)*(cols-6),selected.shape[2])).astype(np.uint8)])
  headerContinue  = np.vstack([
      np.hstack([cv2.resize(cv2.imread(fl[random.randint(0,len(fl)-1)]),(imx//2,imy//2)) for i in range(0,cols)]) for x in range(0,rows-1)
  ])

  header = np.vstack([header,headerContinue])

  cv2.imshow('out',header)
  cv2.waitKey(1)