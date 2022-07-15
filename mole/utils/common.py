import contextlib
import logging
import json
import sys
from contextlib import contextmanager
import tqdm
import shutil
import os


def copy_and_overwrite(from_path, to_path):
    if os.path.exists(to_path):
        shutil.rmtree(to_path)
    shutil.copytree(from_path, to_path)


@contextmanager
def silence_stdout():
    old_target = sys.stdout
    try:
        with open(os.devnull, "w") as new_target:
            sys.stdout = new_target
            yield new_target
    finally:
        sys.stdout = old_target


def is_jsonable(k, v):
    try:
        json.dumps(k)
        json.dumps(v)
        return True
    except:
        return False


def get_time_stamp():
    from datetime import datetime
    now = datetime.now()
    stamp = str(now)[:16].replace("-", "-").replace(" ", "-").replace(":", "-")
    return stamp


def setup_logger(logger, save_dir=None,  propagate=False, level="DEBUG"):
    logging.basicConfig(format='%(asctime)s %(name)s : %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    if logger.hasHandlers():
        logger.handlers.clear()

    if save_dir:
        file_handler = logging.FileHandler(save_dir)
        logger.addHandler(file_handler)

    if level == "DEBUG":
        logger.setLevel(logging.DEBUG)
    elif level == "INFO":
        logger.setLevel(logging.INFO)
    elif level == "ERROR":
        logger.setLevel(logging.ERROR)
    logger.propagate = propagate

    return logger


class RedirectedLogger:
    def __init__(self, logger, level="INFO"):
        self.logger = logger
        self.level = getattr(logging, level)
        self._redirector = contextlib.redirect_stdout(self)

    def write(self, msg):
        if msg and not msg.isspace():
            self.logger.log(self.level, msg)

    def flush(self):
        pass

    def __enter__(self):
        self._redirector.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # let contextlib do any exception handling here
        self._redirector.__exit__(exc_type, exc_value, traceback)



def makedirs(path: str, isfile: bool = False) -> None:
    """
    Creates a directory given a path to either a directory or file.

    If a directory is provided, creates that directory. If a file is provided (i.e. :code:`isfile == True`),
    creates the parent directory for that file.

    :param path: Path to a directory or file.
    :param isfile: Whether the provided path is a directory or file.
    """
    if isfile:
        path = os.path.dirname(path)
    if path != "":
        os.makedirs(path, exist_ok=True)


def delete_folder(path_to_folder):
    if os.path.exists(path_to_folder):
        try:
            shutil.rmtree(path_to_folder)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))


def check_if_intended(text):
    answer = input(f"\n\t❓ {text} (y/n)  \n\t")
    if set("y").issubset(set(answer)):
        pass
    elif set("n").issubset(set(answer)):
        print("\t🚫 Aborting ...\n")
        exit()
    else:
        print("Please type y or n")


# def disable_tqdm():
#     def disabled_tqdm(it, *a, **k):
#         return it
#     tqdm.tqdm = disabled_tqdm

def disable_tqdm():
    return None