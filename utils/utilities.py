import os, errno
import time
import logging
import torch

def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def set_log(path):
    if path:
        ts = time.time()
        logging.basicConfig(filename='%s.%s'%(path, ts),
                            format='%(asctime)s %(levelname)s %(module)s: %(message)s',
                            level=logging.INFO)
    else:
        logging.basicConfig(format='%(asctime)s %(levelname)s %(module)s: %(message)s',
                            level=logging.INFO)

def set_device(device, device_id):
    if device == 'cuda':
        logging.info("Use CUDA, set gpu device %d as default!"%device_id)
        torch.cuda.set_device(device_id)
    else:
        logging.info("Use CPU!")
