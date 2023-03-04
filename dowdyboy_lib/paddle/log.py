import logging

_is_conf = False


def logging_conf(
        filename,
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        filemode='w',
        **args
):
    global _is_conf
    logging.basicConfig(filename=filename, level=level,
                        format=format, filemode=filemode, **args)
    _is_conf = True


def log(txt):
    global _is_conf
    print(txt)
    if _is_conf:
        logging.info(txt)


def warn(txt):
    global _is_conf
    print(txt)
    if _is_conf:
        logging.warning(txt)


