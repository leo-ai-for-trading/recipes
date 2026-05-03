"""

ENUMERATE:
given lis of string we want to store the indices for each unique char
d = collection.defaultdict(list)
for index, char in enumerate(list_string):
    d[char].append(index)

ZIP:
given two list of string we want to find how many characters overlap 
counter = 0
for char1,char2 in zip(list1,list2):
    if char1 == char2:
        counter +=1
if we want to compare three string
for char1,char2,char3 in zip(list1,list2,list3);
    if char1 == char2 == char3:
        counter +=1

LIST COMPREHENSION
given all numebrs from 0 to 100 create an array of all numbers divisible by 7
print([c for c in range(0,100) if x % 7 == 0])

SORTING
sorting using lambdas given a list of strings sort them by string lenght
print(sorted(list1), key = lambda x:len(x))

sort using lambdas with two keys given a list of string and sort them by string leght
ascending order
print(sorted(list1),key=lambda x:(len(str(x)),x))
descending order
print(sorted(list1),key=lambda x:(len(str(x)),-x))

DICTIONARY
store the indices for unique chars in list of string
d = collections.defaultdict(list)
for index, char in enumerate(lists):
    d[char].append(index)

iterating dictionary
for key, value in d.items():
    print(key,value)

sorting dictionary
print(dict(sorted(d.items())))

sort by values
print(dict(sorted(d.items(),key = lambda item: item[1])))

given a list of string, print the chars sorted in order where they appear most
print(collections.Counter(lists))

CUSTOM INCREMENT FOR LOOP
we want to gather all even numbers between -1 to 10
print([x for x in range(0,10,2)])

Shortcuts:
- n = int(sys.stdin.readline())
- n,k = map(int,sys.stdin.readline().split())
- a = list(map(int,sys.stdin.readline().split()))
- for string: list(sys.stdin.readline())
- opening and read input: sys.stdin = open("cowsignal.in","r")
- for the output: sys.stdout = open("cowsignal.out","w")

#################################################################################
import sys

sys.stdin = open("problemname.in", "r")
sys.stdout = open("problemname.out", "w")

#################################################################################

Note: The second argument can be omitted in the open()
command for read-only files

fin = open("problemname.in", "r")
fout = open("problemname.out", "w")

# One way to read the file using .readline()
line1 = fin.readline()
# readline() will pick up where you left off
line2 = fin.readline()
line3 = fin.readline()

# Another way is to use a for loop and .readlines()
line_list = []
for line in fin.readlines():
	pass  # Process input here

# printing line_list would give [line1, line2, line3]

# Output:
fout.write(output_text)  # Write to the output file

# f-strings:
variable1 = 1
variable2 = 2
example_str = f"Hello {variable1} {variable2} World!"
# Printing example_str would give Hello 1 2 World!

#################################################################################
PRINTING SOLUTION

print(ans, end='')  # OK, no newline
print(ans)  # OK, newline
print(str(ans) + '\n', end='')  # OK, newline
print(str(ans) + " ", end='')  # NOT OK, extra space
print(str(ans) + '\n')  # NOT OK, extra newline
"""

import sys
from itertools import permutations
from collections import Counter
import math
M = 1000000007

t = int(sys.stdin.readline())
for _ in range(t):
    n,x,c = map(int,sys.stdin.readline().split())
    a = list(map(int,sys.stdin.readline().split()))
    