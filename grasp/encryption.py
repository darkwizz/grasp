# ---- ENCRYPTION ----------------------------------------------------------------------------------
# The pw() function is secure enough for storing passwords; encrypt() and decrypt() are not secure.
import base64
import hashlib
import os

import binascii

import itertools

from grasp.strings2 import decode_string, encode_string


def key(n=32):
    """ Returns a new key of length n.
    """
    k = os.urandom(256)
    k = binascii.hexlify(k)[:n]
    return decode_string(k)


def stretch(k, n):
    """ Returns a new key of length n.
    """
    while len(k) < n:
        k += hashlib.md5(encode_string(k)[-1024:]).hexdigest()
    return decode_string(k[:n])


def encrypt(s, k=''):
    """ Returns the encrypted string.
    """
    k = stretch(k, len(s))
    k = bytearray(encode_string(k))
    s = bytearray(encode_string(s))
    s = bytearray(((i + j) % 256) for i, j in zip(s, itertools.cycle(k))) # Vigenère cipher
    s = binascii.hexlify(s)
    return decode_string(s)


def decrypt(s, k=''):
    """ Returns the decrypted string.
    """
    k = stretch(k, len(s))
    k = bytearray(encode_string(k))
    s = bytearray(binascii.unhexlify(s))
    s = bytearray(((i - j) % 256) for i, j in zip(s, itertools.cycle(k)))
    s = bytes(s)
    return decode_string(s)

# print(decrypt(encrypt('hello world', '1234'), '1234'))


def pw(s, f='sha256', n=100000):
    """ Returns the encrypted string, using PBKDF2.
    """
    k = base64.b64encode(os.urandom(32))
    s = hashlib.pbkdf2_hmac(f, encode_string(s)[:1024], k, n)
    s = binascii.hexlify(s)
    s = 'pbkdf2:%s:%s:%s:%s' % (f, n, decode_string(k), decode_string(s))
    return s


def pw_ok(s1, s2):
    """ Returns True if pw(s1) == s2.
    """
    _, f, n, k, s = s2.split(':')
    s1 = hashlib.pbkdf2_hmac(f, encode_string(s1)[:1024], encode_string(k), int(n))
    s1 = binascii.hexlify(s1)
    eq = True
    for ch1, ch2 in zip(s1, encode_string(s)):
        eq = ch1 == ch2 # contstant-time comparison
    return eq

# print(pw_ok('1234', pw('1234')))