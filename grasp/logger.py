# ---- LOG -----------------------------------------------------------------------------------------
# Functions that access the internet must report the visited URL using the standard logging module.
# See also: https://docs.python.org/2/library/logging.html#logging.Formatter
import logging

import sys

SIGNED = '%(asctime)s %(filename)s:%(lineno)s %(funcName)s: %(message)s' # 12:59:59 original_grasp.py#1000

log = logging.getLogger(__name__)
log.level = logging.DEBUG

if not log.handlers:
    log.handlers.append(logging.NullHandler())


def debug(file=sys.stdout, format=SIGNED, date='%H:%M:%S'):
    """ Writes the log to the given file-like object.
    """
    h1 = getattr(debug, '_handler', None)
    h2 = file
    if h1 in log.handlers:
        log.handlers.remove(h1)
    if hasattr(h2, 'write') and hasattr(h2, 'flush'):
        h2 = logging.StreamHandler(h2)
        h2.formatter = logging.Formatter(format, date)
    if isinstance(h2, logging.Handler):
        log.handlers.append(h2)
        debug._handler = h2
