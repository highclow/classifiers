import os
import argparse
from glob import glob
import numpy as np
import torch
import torch.nn.functional as F

import logging
import utils
from configs import get_config
from funcs import train, validate, test, visualize


func_map = {"train" : train,
            "val": validate,
            "test" :  test,
            "visualize": visualize}


def get_command():
    parser = argparse.ArgumentParser(description=
                                     "Process for training classifier")
    parser.add_argument("func", choices=func_map.keys(),
                        help="functions. train/val/test")
    parser.add_argument("-f","--config", type=str, required=True,
                        help="configure file")
    parser.add_argument("-w","--weight", type=str, default=None,
                        help="weight for initialization or validate")
    parser.add_argument("-r","--result", type=str, default=None,
                        help="results for shows hard case")
    parser.add_argument("-l","--outlog", type=str,
                        help="output log file")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_command()
    utils.set_log(args.outlog)
    logging.info(args)
    cfgs = get_config(args)
    logging.info("Start %s the network"%args.func)
    func_map[args.func](cfgs)
