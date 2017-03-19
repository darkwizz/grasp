import json
import sys

import re

__version__   =  '1.0'
__license__   =  'BSD'
__credits__   = ['Tom De Smedt', 'Guy De Pauw', 'Walter Daelemans']
__email__     =  'info@textgain.com'
__author__    =  'Textgain'
__copyright__ =  'Textgain'

###################################################################################################

# Copyright (c) 2016, Textgain BVBA
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation and/or
#    other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

###################################################################################################

# WWW  Web Mining                   search engines, servers, HTML DOM + CSS selectors, plaintext
# DB   Databases                    comma-separated values, dates, SQL
# NLP  Natural Language Processing  tokenization, part-of-speech tagging, sentiment analysis
# ML   Machine Learning             clustering, classification, confusion matrix, n-grams
# NET  Network Analysis             shortest paths, centrality, components, communities
# ETC                               recipes for functions, strings, lists, ...

###################################################################################################


try:
    # 3 decimal places (0.001)
    json.encoder.FLOAT_REPR = lambda f: format(f, '.3f')
except:
    pass

PY2 = sys.version.startswith('2')
PY3 = sys.version.startswith('3')

if PY3:
    str, unicode, basestring = bytes, str, str

if PY3:
    # Python 3.4+
    from html.parser import HTMLParser
    from html        import unescape
else:
    # Python 2.7
    from HTMLParser  import HTMLParser
    unescape = HTMLParser().unescape

if PY3:
    import http.server   as BaseHTTPServer
    import socketserver  as SocketServer
else:
    import BaseHTTPServer
    import SocketServer

if PY3:
    import urllib.request
    import urllib.parse as urlparse
    URLError, Request, urlopen, urlencode, urldecode, urlquote = (
        urllib.error.URLError,
        urllib.request.Request,
        urllib.request.urlopen,
        urllib.parse.urlencode,
        urllib.parse.unquote,
        urllib.parse.quote
    )
else:
    import urllib2
    import urllib
    import urlparse
    URLError, Request, urlopen, urlencode, urldecode, urlquote = (
        urllib2.URLError,
        urllib2.Request,
        urllib2.urlopen,
        urllib.urlencode,
        urllib.unquote,
        urllib.quote
    )

# In Python 2, Class.__str__ returns a byte string.
# In Python 3, Class.__str__ returns a Unicode string.

# @printable
# class X(object):
#     def __str__(self):
#         return unicode(' ')

# works on both Python 2 & 3.

def printable(cls):
    """ @printable class defines class.__unicode__ in Python 2.
    """
    if PY2:
        if hasattr(cls, '__str__'):
            cls.__unicode__ = cls.__str__
            cls.__str__ = lambda self: self.__unicode__().encode('utf-8')
    return cls

REGEX = type(re.compile(''))