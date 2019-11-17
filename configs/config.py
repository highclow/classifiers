import logging
import configparser
import io
import torch

# Load the configuration file
def get_config(args):
    config = configparser.ConfigParser()
    config.read(args.config)

    if args.weight:
        config[args.func]["params"] = args.weight

    if not torch.cuda.is_available():
        logging.info("No GPU found! Use CPU instead!")
        config["train"]["device"] = "cpu"
        config["val"]["device"] = "cpu"
        config["test"]["device"] = "cpu"

    if args.result:
        logging.info("Load result from %s"%args.result)
        config["visualize"]["filename"] = args.result
    ## List all contents
    #logging.debug("List all contents")
    #for section in config.sections():
    #    logging.debug("Section: %s" % section)
    #    for options in config.options(section):
    #        logging.debug("x %s:::%s:::%s" % (options,
    #                                  config.get(section, options),
    #                                  str(type(options))))
    

    # Print some contents
#    logging.debug("Training Network: %s"%config.get("model", "net"))  # Just get the value
#    logging.debug("Base Learning Rate: %s"%config.getfloat("train", "base_lr"))  # You know the datatype?
#    config["train"]["base_lr"] = "0.02"
#    logging.debug("Base Learning Rate: %s"%config.getfloat("train", "base_lr"))  # You know the datatype?
    return config

