# ---- TEXT ----------------------------------------------------------------------------------------
import collections
import glob
import re

import unicodedata

from grasp import printable
from grasp.db import cd
from grasp.iteration import nwise, chunks
from grasp.ml import top, Perceptron
from grasp.strings2 import decode_string


def readability(s):
    """ Returns the readability of the given string (0.0-1.0).
    """

    # Flesch Reading Ease; Farr, Jenkins & Patterson's formula.

    def syllables(w, v="aeiouy"):
        # syllables('several') => 2, se-ve-ral
        if w.endswith('e'):
            w = w[:-1]
        return sum(not ch1 in v and \
                   ch2 in v for ch1, ch2 in zip(w, w[1:])) or 1

    s = s.lower()
    s = s.strip()
    s = s.strip('.!?()\'"')
    s = re.sub(r'[\-/]+', ' ', s)
    s = re.sub(r'[\s,]+', ' ', s)
    s = re.sub(r'[.!?]+', '.', s)
    s = re.sub(r'(\. )+', '. ', s)
    y = map(syllables, s.split())  # syllables
    w = max(1, len(s.split(' ')))  # words
    s = max(1, len(s.split('.')))  # sentences
    r = 1.599 * sum(n == 1 for n in y) * 100 / w - 1.015 * w / s - 31.517
    r = 0.01 * r
    r = w == 1 and 1.0 or r
    r = max(r, 0.0)
    r = min(r, 1.0)
    return r


def destress(s, replace={}):
    """ Returns the string with no diacritics.
    """
    for k, v in replace.items():
        s = s.replace(k, v)
    for k, v in {
        u'ø': 'o',
        u'ß': 'ss',
        u'œ': 'ae',
        u'æ': 'oe',
        u'“': '"',
        u'”': '"',
        u'‘': "'",
        u'’': "'",
        u'⁄': '/',
        u'¿': '?',
        u'¡': '!'}.items():
        s = s.replace(k, v)
    f = unicodedata.combining  # f('´') == 0
    s = unicodedata.normalize('NFKD', s)  # é => e + ´
    s = ''.join(ch for ch in s if not f(ch))
    return s


# print(destress(u'pâté')) # 'pate'

def deflood(s, n=3):
    """ Returns the string with no more than n repeated characters.
    """
    if n == 0:
        return s
    return re.sub(r'((.)\2{%s,})' % (n - 1), lambda m: m.group(1)[0] * n, s)


# print(deflood('Woooooow!!!!!!', n=3)) # 'Wooow!!!'

def decamel(s, separator="_"):
    """ Returns the string with CamelCase converted to underscores.
    """
    s = re.sub(r'(.)([A-Z][a-z]{2,})', '\\1%s\\2' % separator, s)
    s = re.sub(r'([a-z0-9])([A-Z])', '\\1%s\\2' % separator, s)
    s = re.sub(r'([A-Za-z])([0-9])', '\\1%s\\2' % separator, s)
    s = re.sub(r'-', separator, s)
    s = s.lower()
    return s


# print decamel('HTTPError404NotFound') # http_error_404_not_found

def sg(w, language='en', known={'aunties': 'auntie'}):
    """ Returns the singular of the given plural noun.
    """
    if w in known:
        return known[w]
    if language == 'en':
        if re.search(r'(?i)ss|[^s]sis|[^mnotr]us$', w):  # ± 93%
            return w
        for pl, sg in (  # ± 98% accurate (CELEX)
                (r'          ^(son|brother|father)s-', '\\1-'),
                (r'      ^(daughter|sister|mother)s-', '\\1-'),
                (r'                          people$', 'person'),
                (r'                             men$', 'man'),
                (r'                        children$', 'child'),
                (r'                           geese$', 'goose'),
                (r'                            feet$', 'foot'),
                (r'                           teeth$', 'tooth'),
                (r'                            oxen$', 'ox'),
                (r'                        (l|m)ice$', '\\1ouse'),
                (r'                        (au|eu)x$', '\\1'),
                (r'                 (ap|cod|rt)ices$', '\\1ex'),  # -ices
                (r'                        (tr)ices$', '\\1ix'),
                (r'                     (l|n|v)ises$', '\\1is'),
                (r'(cri|(i|gn|ch|ph)o|oa|the|ly)ses$', '\\1sis'),
                (r'                            mata$', 'ma'),  # -a/ae
                (r'                              na$', 'non'),
                (r'                               a$', 'um'),
                (r'                               i$', 'us'),
                (r'                              ae$', 'a'),
                (r'           (l|ar|ea|ie|oa|oo)ves$', '\\1f'),  # -ves  +1%
                (r'                     (l|n|w)ives$', '\\1ife'),
                (r'                 ^([^g])(oe|ie)s$', '\\1\\2'),  # -ies  +5%
                (r'                  (^ser|spec)ies$', '\\1ies'),
                (r'(eb|gp|ix|ipp|mb|ok|ov|rd|wn)ies$', '\\1ie'),
                (r'                             ies$', 'y'),
                (r'    ([^rw]i|[^eo]a|^a|lan|y)ches$', '\\1che'),  # -es   +5%
                (r'  ([^c]ho|fo|th|ph|(a|e|xc)us)es$', '\\1e'),
                (r'([^o]us|ias|ss|sh|zz|ch|h|o|x)es$', '\\1'),
                (r'                               s$', '')):  # -s    +85%
            if re.search(r'(?i)' + pl.strip(), w):
                return re.sub(r'(?i)' + pl.strip(), sg, w)
    return w  # +1%


# print(sg('avalanches')) # avalanche

# ---- TOKENIZER -----------------------------------------------------------------------------------
# The tokenize() function identifies tokens (= words, symbols) and sentence breaks in a string.
# The task involves handling abbreviations, contractions, hyphenation, emoticons, URLs, ...

EMOJI = set((
    u'😊', u'☺️', u'😉', u'😌', u'😏', u'😎', u'😍', u'😘', u'😴', u'😀', u'😃', u'😄', u'😅',
    u'😇', u'😂', u'😭', u'😢', u'😱', u'😳', u'😜', u'😛', u'😁', u'😐', u'😕', u'😧', u'😦',
    u'😒', u'😞', u'😔', u'😫', u'😩', u'😠', u'😡', u'🙊', u'🙈', u'💔', u'❤️', u'💕', u'♥',
    u'👌', u'✌️', u'👍', u'🙏'
))

EMOTICON = set((
    ':-D', '8-D', ':D', '8)', '8-)',
    ':-)', '(-:', ':)', '(:', ':-]', ':=]', ':-))',
    ':-(', ')-:', ':(', '):', ':((', ":'(", ":'-(",
    ':-P', ':-p', ':P', ':p', ';-p',
    ':-O', ':-o', ':O', ':o', '8-o'
                              ';-)', ';-D', ';)', '<3'
))

abbreviations = {
    'en': set(('a.m.', 'cf.', 'e.g.', 'etc.', 'i.e.', 'p.m.', 'vs.', 'w/', 'Dr.', 'Mr.'))
}

contractions = {
    'en': set(("'d", "'m", "'s", "'ll", "'re", "'ve", "n't"))
}

for c in contractions.values():
    c |= set(s.replace("'", u'’') for s in c)  # n’t

_RE_EMO1 = '|'.join(re.escape(' ' + ' '.join(s)) for s in EMOTICON | EMOJI).replace(r'\-\ ', '\- ?')
_RE_EMO2 = '|'.join(re.escape(''.join(s)) for s in EMOTICON | EMOJI)


def tokenize(s, language='en', known=[]):
    """ Returns the string with punctuation marks split from words ,
        and sentences separated by newlines.
    """
    p = u'….¡¿?!:;,/(){}[]\'`‘’"“”„&–—'
    p = re.compile(r'([%s])' % re.escape(p))
    f = re.sub

    # Find tokens w/ punctuation (URLs, numbers, ...)
    w = set(known)
    w |= set(re.findall(r'https?://.*?(?:\s|$)', s))  # http://...
    w |= set(re.findall(r'(?:\s|^)((?:[A-Z]\.)+[A-Z]\.?)', s))  # U.S.
    w |= set(re.findall(r'(?:\s|^)([A-Z]\. )(?=[A-Z])', s))  # J. R. R. Tolkien
    w |= set(re.findall(r'(\d+[\.,:/″][\d\%]+)', s))  # 1.23
    w |= set(re.findall(r'(\w+\.(?:doc|html|jpg|pdf|txt|zip))', s, re.U))  # cat.jpg
    w |= abbreviations.get(language, set())  # etc.
    w |= contractions.get(language, set())  # 're
    w = '|'.join(f(p, r' \\\1 ', w).replace('  ', ' ') for w in w)
    e1 = _RE_EMO1
    e2 = _RE_EMO2

    # Split punctuation:
    s = f(p, ' \\1 ', s)
    s = f(r'\t', ' ', s)
    s = f(r' +', ' ', s)
    s = f(r'\. (?=\.)', '.', s)  # ...
    s = f(r'(\(|\[) (\.+) (\]|\))', '\\1\\2\\3', s)  # (...)
    s = f(r'(\(|\[) (\d+) (\]|\))', '\\1\\2\\3', s)  # [1]
    s = f(r'(^| )(p?p) \. (?=\d)', '\\1\\2. ', s)  # p.1
    s = f(r'(?<=\d)(mb|gb|kg|km|m|ft|h|hrs|am|pm) ', ' \\1 ', s)  # 5ft
    s = f(u'(^|\s)($|£|€)(?=\d)', '\\1\\2 ', s)  # $10
    s = f(r'& (#\d+|[A-Za-z]+) ;', '&\\1;', s)  # &amp; &#38;
    s = f(w, lambda m: ' %s ' % m.group().replace(' ', ''), s)  # e.g. 1.2
    s = f(w, lambda m: ' %s ' % m.group().replace(' ', ''), s)  # U.K.'s
    s = f(e1, lambda m: ' %s ' % m.group().replace(' ', ''), s)  # :-)
    s = f(r'(https?://.*?)([.?!,)])(?=\s|$)', '\\1 \\2', s)  # (http://)
    s = f(r'(https?://.*?)([.?!,)])(?=\s|$)', '\\1 \\2', s)  # (http://),
    s = f(r'www \. (.*?) \. (?=[a-z])', 'www.\\1.', s)  # www.goo.gl
    s = f(r' \. (com?|net|org|be|cn|de|fr|nl|ru|uk)(?=\s|$)', '.\\1', s)  # google.com
    s = f(r'(\w+) \. (?=\w+@\w+\.(?=com|net|org))', ' \\1.', s)  # a.bc@gmail
    s = f(r'(-)\n([a-z]\w+(?:\s|$))', '\\2\n', s, re.U)  # be-\ncause
    s = f(r' +', ' ', s)

    # Split sentences:
    s = f(r'\r+', '\n', s)
    s = f(r'\n+', '\n', s)
    s = f(u' ([….?!]) (?!=[….?!])', ' \\1\n', s)  # .\n
    s = f(u'\n([’”)]) ', ' \\1\n', s)  # “Wow.”
    s = f(r'\n(((%s) ?)+) ' % e2, ' \\1\n', s)  # Wow! :-)
    s = f(r' (%s) (?=[A-Z])' % e2, ' \\1\n', s)  # Wow :-) The
    s = f(r' (etc\.) (?=[A-Z])', ' \\1\n', s)  # etc. The
    s = f(r'\n, ', ' , ', s)  # Aha!, I said
    s = f(r' +', ' ', s)
    s = s.split('\n')

    # Balance quotes:
    for i in range(len(s) - 1):
        j = i + 1
        if s[j].startswith(('" ', "' ")) and \
                                s[j].count(s[j][0]) % 2 == 1 and \
                                s[i].count(s[j][0]) % 2 == 1:
            s[i] += ' ' + s[j][0]
            s[j] = s[j][2:]

    return '\n'.join(s).strip()


# s = u"RT @user: Check it out... 😎 (https://www.textgain.com) #Textgain cat.jpg"
# s = u"There's a sentence on each line, each a space-separated string of tokens (i.e., words). :)"
# s = u"Title\nA sentence.Another sentence. ‘A citation.’ By T. De Smedt."

def wc(s):
    """ Returns a (word, count)-dict, lowercase.
    """
    f = re.split(r'\s+', s.lower())
    f = collections.Counter(f)
    return f


# print(wc(tokenize('The cat sat on the mat.')))

# ---- PART-OF-SPEECH TAGGER -----------------------------------------------------------------------
# Part-of-speech tagging predicts the role of each word in a sentence: noun, verb, adjective, ...
# Different words have different roles according to their position in the sentence (context).
# In 'Can I have that can of soda?', 'can' is used as a verb ('I can') and a noun ('can of').

# We want to train a machine learning algorithm (Percepton) with context vectors as examples,
# where each context vector includes the word, word suffix, and words to the left and right.

# The part-of-speech tag of unknown words is predicted by their suffix (e.g., -ing, -ly)
# and their context (e.g., a determiner is often followed by an adjective or a noun).

def ctx(*w):
    """ Returns the given [token, tag] list parameters as
        a context vector (i.e., token, tokens left/right)
        that can be used to train a part-of-speech tagger.
    """
    m = len(w) // 2  # middle
    v = set()
    for i, (w, tag) in enumerate(w):
        i -= m
        if i == 0:
            v.add(' ')  # bias
            v.add('1 %+d %s' % (i, w[:+1]))  # capitalization
            v.add('* %+d %s' % (i, w[-6:]))  # token
            v.add('^ %+d %s' % (i, w[:+3]))  # token head
            v.add('$ %+d %s' % (i, w[-3:]))  # token suffix
        else:
            v.add('$ %+d %s' % (i, w[-3:]))  # token suffix left/right
            v.add('? %+d %s' % (i, tag))  # token tag
    return v


# print(ctx(['The', 'DET'], ['cat', 'NOUN'], ['sat', 'VERB'])) # context of 'cat'
#
# set([
#     ' '        ,
#     '$ -1 The' ,
#     '? -1 DET' ,
#     '1 +0 c'   ,
#     '* +0 cat' ,
#     '^ +0 cat' ,
#     '$ +1 sat' ,
#     '? +1 VERB',
# ])

@printable
class Sentence(list):
    def __init__(self, s=''):
        """ Returns the tagged sentence as a list of [token, tag]-values.
        """
        if isinstance(s, list):
            list.__init__(self, s)
        if isinstance(s, (str, unicode)) and s:
            for w in s.split(' '):
                w = decode_string(w)
                w = re.split(r'(?<!\\)/', w + '/')[:2]
                w = [w.replace('\/', '/') for w in w]
                self.append(w)

    def __str__(self):
        return ' '.join('/'.join(w.replace('/', '\\/') for w in w) for w in self)

    def __repr__(self):
        return 'Sentence(%s)' % repr(decode_string(self))


# s = 'The/DET cat/NOUN sat/VERB on/PREP the/DET mat/NOUN ./PUNC'
# for w, tag in Sentence(s):
#     print(w, tag)

TAGGER = {}  # {'en': Model}

for f in glob.glob(cd('*-pos.json')):
    TAGGER[f.split('/')[-1][:2]] = Perceptron.load(open(f))


def tag(s, language='en'):
    """ Returns the tokenized + tagged string.
    """
    return '\n'.join(decode_string(s) for s in parse(s, language))


def parse(s, language='en'):
    """ Returns the tokenized + tagged string,
        as an iterator of Sentence objects.
    """
    model = TAGGER[language]
    s = tokenize(s)
    s = s.replace('/', '\\/')
    s = s.split('\n')
    for s in s:
        a = Sentence()
        for w in nwise(Sentence('  %s  ' % s), n=5):
            if len(a) > 1:
                w[0][1] = a[-2][1]  # use predicted tag left
                w[1][1] = a[-1][1]
            tag, p = top(model.predict(ctx(*w)))
            a.append([w[2][0], tag])
        yield a


# for s in parse("We're all mad here. I'm mad. You're mad."):
#     print(repr(s))

PTB = {  # Penn Treebank tagset                                           # EN
    u'NOUN': set(('NN', 'NNS', 'NNP', 'NNPS', 'NP')),  # 30%
    u'VERB': set(('VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'MD')),  # 14%
    u'PUNC': set(('LS', 'SYM', '.', ',', ':', '(', ')', '``', "''", '$', '#')),  # 11%
    u'PREP': set(('IN', 'PP')),  # 10%
    u'DET': set(('DT', 'PDT', 'WDT', 'EX')),  # 9%
    u'ADJ': set(('JJ', 'JJR', 'JJS')),  # 7%
    u'ADV': set(('RB', 'RBR', 'RBS', 'WRB')),  # 4%
    u'NUM': set(('CD', 'NO')),  # 4%
    u'PRON': set(('PR', 'PRP', 'PRP$', 'WP', 'WP$')),  # 3%
    u'CONJ': set(('CC', 'CJ')),  # 2%
    u'X': set(('FW',)),  # 2%
    u'PRT': set(('POS', 'PT', 'RP', 'TO')),  # 2%
    u'INTJ': set(('UH',)),  # 1%
}

WEB = dict(PTB, **{
    u'NOUN': set(('NN', 'NNS', 'NP')),  # 14%
    u'NAME': set(('NNP', 'NNPS', '@')),  # 11%
    u'PRON': set(('PR', 'PRP', 'PRP$', 'WP', 'WP$', 'PR|MD', 'PR|VB', 'WP|VB')),  # 9%
    u'URL': set(('URL',)),  # 'youve'  'thats'  'whats'     #  1%
    u':)': set((':)',)),  # 1%
    u'#': set(('#',)),  # 1%
})
WEB['PUNC'].remove('#')
WEB['PUNC'].add('RT')


def universal(w, tag, tagset=WEB):
    """ Returns a simplified tag (e.g., NOUN) for the given Penn Treebank tag (e.g, NNS).
    """
    tag = tag.split('-')[0]
    tag = tag.split('|')[0]
    tag = tag.split('&')[0]
    if tag == w == '#':
        return w, 'PUNC'  # != hashtag
    for k in tagset:
        if tag in tagset[k]:
            return w, k
    return w, tag


# print(universal('rabbits', 'NNS'))

# The 1989 Wall Street Journal corpus contains 1 million manually annotated words, e.g.:
# Pierre/NNP Vinken/NNP ,/, 61/CD years/NNS old/JJ ,/, will/MD join/VB the/DT board/NN
# as/IN a/DT nonexecutive/JJ director/NN Nov./NNP 29/CD ./.

# corpus = cd('/corpora/wsj.txt')
# corpus = u(open(corpus).read())
# corpus = corpus.split('\n')
# corpus = corpus[:48000] # ~ 1M tokens

# Create context vectors from WSJ sentences, using the simplified universal tagset.

# data = []
# for s in corpus:
#     for w in nwise(Sentence('  %s  ' % s), n=5):
#         w = [universal(*w) for w in w]
#         data.append((ctx(*w), w[2][1]))
#
# print('%s sentences' % len(corpus))
# print('%s tokens'    % len(data))
#
# print(kfoldcv(Perceptron, data, k=3, n=5, weighted=True, debug=True)) # 0.96 0.96
#
# en = Perceptron(data, n=5)
# en.save(open('en.json', 'w'))
#
# print(tag('What a great day! I love it.'))

# ---- PART-OF-SPEECH SEARCH -----------------------------------------------------------------------
# The search() function yields parts of a part-of-speech-tagged sentence that match given pattern.
# For example, 'ADJ* NOUN' yields all nouns in a sentence and optionally the preceding adjectives.

# The chunked() function yields NP, VP, AP and PP phrases.
# A NP (noun phrase) is a noun + preceding determiners and adjectives (e.g., 'the big black cat').
# A VP (verb phrase) is a verb + preceding auxillary verbs (e.g., 'might be doing').

TAG = set((
    'NAME',
    'NOUN',
    'VERB',
    'PUNC',
    'PREP',
    'DET',
    'ADJ',
    'ADV',
    'NUM',
    'PRON',
    'CONJ',
    'X',
    'PRT',
    'INTJ',
    'URL',
    ':)',
    '#'
))

inflections = {
    'aux': r"can|shall|will|may|must|could|should|would|might|'ll|'d",
    'be': r"be|am|are|is|was|were|being|been|'m|'re|'s",
    'have': r"have|has|had|having|'ve"
}


class Phrase(Sentence):
    pass


def search(pattern, sentence, replace=[]):
    """ Yields an iterator of matching Phrase objects from the given Sentence.
        The search pattern is a sequence of tokens (talk-, -ing), tags (VERB),
        token/tags (-ing/VERB), and control characters:
        - ( ) group
        -  |  options: NOUN|PRON CONJ|,
        -  *  0+ tags: NOUN*
        -  +  1+ tags: NOUN+
        -  ?  <2 tags: NOUN?
    """
    R = r'(?<!\\)'  # = not preceded by \
    s = re.sub(r'\s+', ' ', pattern)
    s = re.sub(r' ([*+?]) ', ' -\\1 ', s)
    s = re.sub(R + r'([()^$])', ' \\1 ', s)
    s = s.strip()
    p = []

    for w in s.split(' '):
        if w in ('(', ')', '^', '$', '*', '+', '?', ''):
            p.append(w)
            continue
        for k, v in replace:
            w = w.replace(k.upper(), v)
        for k, v in inflections.items():
            w = w.replace(k.upper(), v)

        try:
            w, x, _ = re.split(R + r'([*+?])$', w)  # ['ADJ|-ing', '+']
        except ValueError:
            x = ''
        if not re.search(R + r'/', w):
            a = re.split(R + r'\|', w)  # ['ADJ', '-ing']
            for i, w in enumerate(a):
                if w in TAG:
                    a[i] = r'(?:\S+/%s)' % w  # '(?:\S+/ADJ)'
                else:
                    a[i] = r'(?:%s/[A-Z]{1,4})' % w  # '(?:-ing/[A-Z]{1,4})'
            w = '|'.join(a)  # '(?:\S+/ADJ)|(?:-ing/[A-Z]{1,4})'
        else:
            w = re.sub(R + r'/', ')/(?:', w)  # '(?:-ing)/(?:VERB|ADJ)'

        w = '(?:%s)' % w
        w = '(?:%s )%s' % (w, x)  # '(?:(?:-ing/[A-Z]{1,4}) )+'
        w = re.sub(r'\(\?:-', r'(?:\S*', w)  # '\S*ing/[A-Z]{1,4}'
        w = re.sub(R + r'-/', r'\S*/', w)
        p.append(w)

    p = '(%s)' % ''.join(p)
    for m in re.finditer(p, '%s ' % sentence, re.I):
        m = ((m or '').strip() for m in m.groups())
        m = map(Phrase, m)
        m = tuple(m)
        if len(m) > 1:
            yield m
        else:
            yield m[0]


# for m in \
#  search('ADJ',
#     tag('A big, black cat.')):
#      print(u(m))

# for m, g1, g2 in \
#  search('DET? (NOUN) AUX? BE (-ry)',
#     tag("The cats'll be hungry.")):
#     print(u(g1), u(g2))

# for m, g1, g2 in \
#  search('DET? (NOUN) AUX? BE (-ry)',
#     tag("The boss'll be angry!")):
#     print(u(g1), u(g2))

# for m, g1, g2, g3 in \
#  search('(NOUN|PRON) BE ADV? (ADJ) than (NOUN|PRON)',
#     tag("Cats are more stubborn than dogs.")):
#     print(u(g1), u(g2), u(g3))

def chunked(sentence, language='en'):
    """ Yields an iterator of (tag, Phrase)-tuples from the given Sentence,
        with tags NP (noun phrase), VP (verb phrase), AP (adjective phrase)
        or PP (prepositional phrase).
    """
    if language in ('de', 'en', 'nl'):  # Germanic
        P = (('NP', 'DET|PRON? NUM* (ADV|ADJ+ CONJ|, ADV|ADJ)* ADV|ADJ* -ing/VERB* NOUN|NAME+'),
             ('NP', 'NOUN|NAME DET NOUN|NAME'),
             ('NP', 'PRON'),
             ('AP', '(ADV|ADJ+ CONJ|, ADV|ADJ)* ADV* ADJ+'),
             ('VP', 'PRT|ADV* VERB+ ADV?'),
             ('PP', 'PREP+'),
             ('', '-')
             )
    s = decode_string(sentence)
    s = re.sub(r'\s+', ' ', s)
    while s:
        for tag, p in P:
            try:
                m = next(search('^(%s)' % p, s))[0];
                break
            except StopIteration:
                m = ''
        if not m:
            m = Phrase(s.split(' ', 1)[0])
        if not m:
            break
        s = s[len(decode_string(m)):]
        s = s.strip()
        yield tag, m


# s = tag('The black cat is dozing lazily on the couch.')
# for ch, s in chunked(s):
#     print(ch, u(s))

# ---- WORDNET -------------------------------------------------------------------------------------
# WordNet is a free lexical database of synonym sets, and relations between synonym sets.

SYNSET = r'^\d{8} \d{2} \w .. ((?:.+? . )+?)\d{3} ((?:..? \d{8} \w .... )*)(.*?)\| (.*)$'
# '05194874 07 n 02 grip 0 grasp 0 001 @ 05194151 n 0000 | an intellectual understanding'
#  https://wordnet.princeton.edu/wordnet/man/wndb.5WN.html#sect3

POINTER = {
    'antonym': '!',  #
    'hypernym': '@',  # grape -> fruit
    'hyponym': '~',  # grape -> muscadine
    'holonym': '#',  # grape -> grapevine
    'meronym': '%',  # grape -> wine
}


class Wordnet(dict):
    def __init__(self, path='WordNet-3.0'):
        """ Opens the WordNet database from the given path
            (that contains dict/index.noun, dict/data.noun, ...)
        """
        self._f = {}  # {'n': <open file 'dict/index.noun'>}

        for k, v in (('n', 'noun'), ('v', 'verb'), ('a', 'adj'), ('r', 'adv')):

            f = cd(path, 'dict', 'data.%s' % v)
            f = open(f, 'rb')
            self._f[k] = f

            f = cd(path, 'dict', 'index.%s' % v)
            f = open(f, 'r')
            for s in f:
                if not s.startswith(' '):
                    s = s.strip()
                    s = s.split(' ')
                    p = s[-int(s[2]):]
                    w = s[0]
                    w = w.replace('_', ' ')
                    self[w, k] = p  # {('grasp', 'n'): (offset1, ...)}
            f.close()

    def synset(self, offset, pos='n'):
        f = self._f[pos]
        f.seek(int(offset))
        s = f.readline()
        s = s.strip()
        s = s.decode('utf-8')
        m = re.match(SYNSET, s)
        w = m.group(1)
        p = m.group(2)
        g = m.group(4)
        p = tuple(chunks(p.split(' '), n=4))  # (pointer, offset, pos, source/target)
        w = tuple(chunks(w.split(' '), n=2))  # (word, sense)
        w = tuple(w.replace('_', ' ') for w, i in w)

        return Synset(offset, pos, lemma=w, pointers=p, gloss=g, factory=self.synset)

    def synsets(self, w, pos='n'):
        """ Returns a tuple of senses for the given word,
            where each sense is a Synset (= synonym set).
        """
        w = w.lower()
        w = w, pos
        return tuple(self.synset(offset, pos) for offset in self.get(w, ()))

    def __call__(self, *args, **kwargs):
        return self.synsets(*args, **kwargs)


class Synset(tuple):
    def __new__(self, offset, pos, lemma, pointers=[], gloss='', factory=None):
        return tuple.__new__(self, lemma)

    def __init__(self, offset, pos, lemma, pointers=[], gloss='', factory=None):
        """ A set of synonyms, with semantic relations and a definition (gloss).
        """
        self.synset = factory
        self.offset = offset
        self.pos = pos
        self.pointers = pointers
        self.gloss = gloss

    @property
    def id(self):
        return '%s-%s' % (self.offset, self.pos)

    # Synset.hypernyms, .hyponyms, ...
    def __getattr__(self, k):
        v = POINTER[k.replace('_', ' ')[:-1]]  # -s
        v = tuple(self.synset(p[1], p[2]) for p in self.pointers if p[0].startswith(v))
        setattr(self, k, v)  # lazy
        return v

    def __repr__(self):
        return 'Synset(%s)' % tuple.__repr__(self)

        # wn = Wordnet(path='WordNet-3.0')
        # for s in wn.synsets('grasp', pos='n'):
        #     print(s)
        #     print(s.gloss)
        #     print(s.hyponyms)
        #     print()
