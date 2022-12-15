import sys
import numpy as np

filename = sys.argv[1]

f_to_compare = open('outputs/'+filename+'-pr-mpi.txt').readlines()
f_correct = open('test/'+filename+'-pr-p.txt').readlines()

threshold = 1e-4

differences = []
for i in range(len(f_to_compare)):
	if not f_to_compare[i].startswith('s'):
		pr1 = float(f_to_compare[i].split()[2])
		pr2 = float(f_correct[i].split()[2])
		differences.append(abs(pr1 - pr2))

h, b = np.histogram(differences, bins=np.logspace(-16, -5, 12))

with np.printoptions(precision=4):
	for i in range(len(h)):
		print(b[i], ':', h[i])
	print('Total = ', len(f_to_compare)-1)

# if different:
# 	print("Differences found")
# else:
# 	print("Same")