import logging


def get_PIL_version()->list:
    import PIL

    return str(PIL.__version__).split('.')


def set_logger():
    logger = logging.getLogger('bayesian_nn')
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger
