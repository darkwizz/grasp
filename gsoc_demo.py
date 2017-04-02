"""
1. Divide string by spaces; also there may be words with punctuations but not
separated by space (Hello.My name is ...). So divide them;
2. Group parts in sentences:
    - there are dividing punctuation marks (. ... ! ?) and next word starts from upper letter;
    - there are dividing punctuation marks and next word starts from digit;
3. Take out punctuations from words (like "however, ...");
4. Set different combinations of punctuation marks to find them among other non-alphanumeric words;
5. Check each non-alphanumeric word on containing such combinations;
6. If it is, then split by these combinations and join them with space;
7. Finish with creating sentence joining rest of words and symbols with space;

OR
Alternative
1. Split text on sentences with regex;
2. Correct fake matching parts (floating point numbers);
3. Make split sentences correct completing them;
4. Divide punctuation marks from words with spaces;
"""
import re

SENTENCE_DIVIDER = u'([\\d\\w\\)](\\.|\\?|(\\.{3})|!)+\\s*(\\(|[A-Z]|[0-9]))'


def escaping_replacement(matched):
    return u'\\' + matched.group()


def spacing_replacement(matched):
    # this check is rough and imprecise enough, but I think that floating point numbers
    # happen more often than digits both in end of previous and start next of sentence
    start, end = matched.regs[0][0] - 1, matched.regs[0][1] + 1
    if re.match(u'\d.*\d', matched.string[start:end]):
        return matched.group()
    result = u' ' + matched.group() if end < matched.endpos \
        and matched.string[end] == u' ' \
        else u' ' + matched.group() + u' '
    return result


def tokenize(input_text):
    # 1. 2.
    input_text = input_text.strip()
    groups = [item.group() for item in re.finditer(SENTENCE_DIVIDER,
                                                   input_text) if not re.match(u'\d+.\d', item.group())]
    escaping_reg = u'(\\.|\\)|\\(|\\?|(\\.{3}))'
    splitting_pattern = u'(' + u'|'.join([re.sub(escaping_reg,
                                                 escaping_replacement, group) for group in groups]) + u')'
    sentences = []
    parts = re.split(splitting_pattern, input_text)
    spacing_reg = u'[^a-zA-Z\\d\\s]'
    for i in range(len(parts)):
        part = parts[i]
        if part in groups:  # this part never won't be when i = 0 nor i = -1, because split always
                            # divides text into at least two parts
            sentences[-1] = re.sub(spacing_reg, spacing_replacement, (parts[i - 1] + part[:-1]).strip()).strip()
            parts[i + 1] = part[-1] + parts[i + 1]
        elif part != u'':
            # part re.sub where all punctuation marks will be separated with space from words
            subbed = re.sub(spacing_reg, spacing_replacement, part.strip()).strip()
            sentences.append(subbed)
    return sentences

text = u"""
Welcome to RegExr v2.1 by gskinner.com, proudly hosted by Media Temple!

Edit the Expression & Text to see matches. Roll over matches or the expression for details. Undo mistakes with ctrl-z. Save Favorites & Share expressions with friends or the Community. Explore your results with Tools. A full Reference & Help is available in the Library, or watch the video Tutorial.

Sample text for testing:
abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ
0123456789 _+-.,!@#$%^&*();\/|<>"'
12345 -98.7 3.141 .6180 9,000 +42
555.123.4567	+1-(800)-555-2468
foo@demo.net	bar.ba@test.co.uk
www.demo.com	http://foo.co.uk/
http://regexr.com/foo.html?q=bar
https://mediatemple.net

Hello, my name is Artur). I very want to be NLP specialist!It's very interesting area of Data Science, which, I think, helps me to <enter your aim>. 5 of course!!
"""

sentences = tokenize(text)
print(u'\n'.join(map(lambda x: unicode(x[0] + 1) + u' Sentence - ' + x[1], enumerate(sentences))))
