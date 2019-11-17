import logging
import configparser
import io
import torch

# Load the configuration file
def get_config(args):
    config = configparser.ConfigParser()
    config.read(args.config)

    ## List all contents
    #logging.debug("List all contents")
    #for section in config.sections():
    #    logging.info("Section: %s" % section)
    #    for options in config.options(section):
    #        logging.info("x %s:::%s:::%s" % (options,
    #                                  config.get(section, options),
    #                                  str(type(options))))

    print(args.func, args.weight, config[args.func])
    if args.weight:
        config[args.func]["params"] = args.weight

    if not torch.cuda.is_available():
        logging.info("No GPU found! Use CPU instead!")
        config[args.func]["device"] = "cpu"

    if args.func == "visualize" and args.result is not None:
        logging.info("Load result from %s"%args.result)
        config["visualize"]["filename"] = args.result
    

    # Print some contents
#    logging.debug("Training Network: %s"%config.get("model", "net"))  # Just get the value
#    logging.debug("Base Learning Rate: %s"%config.getfloat("train", "base_lr"))  # You know the datatype?
#    config["train"]["base_lr"] = "0.02"
#    logging.debug("Base Learning Rate: %s"%config.getfloat("train", "base_lr"))  # You know the datatype?
    return config

