# ---- PARALLEL ------------------------------------------------------------------------------------
# Parallel processing uses multiple CPU's to execute multiple processes simultaneously.
import multiprocessing
import threading

import time


def parallel(f, values=[], *args, **kwargs):
    """ Returns an iterator of f(v, *args, **kwargs)
        for values=[v1, v2, ...], using available CPU's.
    """
    p = multiprocessing.Pool(processes=None)
    p = p.imap(_worker, ((f, v, args, kwargs) for v in values))
    return p


def _worker(x):
    f, v, args, kwargs = x
    return f(v, *args, **kwargs)


# ---- ASYNC ---------------------------------------------------------------------------------------
# Asynchronous functions are executed in a separate thread and notify a callback function
# (instead of blocking the main thread).

def asynchronous(f, callback=lambda v, e: None, daemon=True):
    """ Returns a new function that calls
        callback(value, exception=None) when done.
    """
    def thread(*args, **kwargs):
        def worker(callback, f, *args, **kwargs):
            try:
                v = f(*args, **kwargs)
            except Exception as e:
                callback(None, e)
            else:
                callback(v, None)
        t = threading.Thread
        t = t(target=worker, args=(callback, f) + args, kwargs=kwargs)
        t.daemon = daemon # False = program only ends if thread stops.
        t.start()
        return t
    return thread

# def ping(v, e=None):
#     if e:
#         raise e
#     print(v)
#
# pow = asynchronous(pow, ping)
# pow(2, 2)
# pow(2, 3) #.join(1)
#
# for i in range(10):
#     time.sleep(0.1)
#     print('...')

# Atomic operations are thread-safe, e.g., dict.get() or list.append(),
# but not all operations are atomic, e.g., dict[k] += 1 needs a lock.

_lock = threading.RLock()


def atomic(f):
    """ The @atomic decorator executes a function thread-safe.
    """
    def decorator(*args, **kwargs):
        with _lock:
            return f(*args, **kwargs)
    return decorator

# hits = collections.defaultdict(int)
#
# @atomic
# def hit(k):
#     hits[k] += 1

MINUTE, HOUR, DAY = 60, 60 * 60, 60 * 60 * 24


def scheduled(interval=MINUTE):
    """ The @scheduled decorator executes a function periodically (async).
    """
    def decorator(f):
        def timer():
            while 1:
                time.sleep(interval)
                f()
        t = threading.Thread(target=timer)
        t.start()
    return decorator

# @scheduled(1)
# @atomic
# def update():
#     print("updating...")


def retry(exception, tries, f, *args, **kwargs):
    """ Returns the value of f(*args, **kwargs).
        Retries if the given exception is raised.
    """
    for i in range(tries + 1):
        try:
            return f(*args, **kwargs)
        except exception as e:
            if i < tries:
                time.sleep(2 ** i) # exponential backoff (1, 2, 4, ...)
        except Exception as e:
            raise e
    raise exception
