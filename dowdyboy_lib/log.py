import logging

_is_conf = False


def logging_conf(
        filename,
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        filemode='w',
        acc=None,
        **args
):
    global _is_conf
    if acc is None:
        logging.basicConfig(filename=filename, level=level,
                            format=format, filemode=filemode, **args)
        _is_conf = True
    else:
        if acc.is_local_main_process:
            logging.basicConfig(filename=filename, level=level,
                                format=format, filemode=filemode, **args)
            _is_conf = True


def log(txt, acc=None):
    global _is_conf
    if acc is None:
        print(txt)
        if _is_conf:
            logging.info(txt)
    else:
        if acc.is_local_main_process:
            acc.print(txt)
            if _is_conf:
                logging.info(txt)


def warn(txt, acc=None):
    global _is_conf
    if acc is None:
        print(txt)
        if _is_conf:
            logging.warning(txt)
    else:
        if acc.is_local_main_process:
            acc.print(txt)
            if _is_conf:
                logging.warning(txt)



