from __future__ import print_function
import original_grasp as grasp

parsed = grasp.parse('The quick brown fox jumps over the lazy dog')
for sentence in parsed:
    for tag in sentence:
        print(tag, end=' ')
    print()

# prints not so correct output - most of tokens are tagged with :) (some troubles, I think, with training,
#                                                           because I didn't find where model was trained)
