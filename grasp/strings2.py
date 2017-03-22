# ---- UNICODE -------------------------------------------------------------------------------------
# The decode_string() function returns a Unicode string (Python 2 & 3).
# The encode_string() function returns a byte string, encoded as UTF-8.

# We use decode_string() as early as possible on all input (e.g. HTML).
# We use encode_string() on URLs.


def decode_string(v, encoding='utf-8'):
    """ Returns the given value as a Unicode string.
    """
    if isinstance(v, str):
        for e in ((encoding,), ('windows-1252',), ('utf-8', 'ignore')):
            try:
                return v.decode(*e)
            except:
                pass
        return v
    if isinstance(v, unicode):
        return v
    return (u'%s' % v) # int, float


def encode_string(v, encoding='utf-8'):
    """ Returns the given value as a byte string.
    """
    if isinstance(v, unicode):
        for e in ((encoding,), ('windows-1252',), ('utf-8', 'ignore')):
            try:
                return v.encode(*e)
            except:
                pass
        return v
    if isinstance(v, str):
        return v
    return (u'%s' % v).encode()
