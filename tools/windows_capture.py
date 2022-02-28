import numpy as np
import cv2
import yolop_detect
import time
from PIL import ImageGrab
from win32gui import FindWindow, GetWindowRect

from lib.config import cfg
import torch

while True:
    window_name = "Need for Speed™ Payback"
    id = FindWindow(None, window_name)
    bbox = GetWindowRect(id)
    img0 = np.array(ImageGrab.grab(bbox=bbox))
    img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    img0 = cv2.resize(img0, (1280,720), interpolation=cv2.INTER_LINEAR)

    with torch.no_grad():
        
        t0 = time.time()
        output_image = yolop_detect.detect(cfg, img0)
        cv2.imshow('screenshot', output_image)
        print('Done. (%.3fs)' % (time.time() - t0))

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break