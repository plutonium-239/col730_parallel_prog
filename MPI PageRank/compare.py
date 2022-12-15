import pandas as pd
import sys, os
# import plotly.express as px
import subprocess


def compare(filename):
	f1 = open('outputs/'+filename+'-pr-mpi.txt').readlines()
	f2 = open('test/'+filename+'-pr-p.txt').readlines()

	all_diffs = []

	different = False
	for i in range(len(f1)):
		if f1[i].startswith('s'):
			continue
		c1 = f1[i].split()
		c2 = f2[i].split()
		assert(c1[0]==c2[0])
		diff = abs(float(c1[2]) - float(c2[2]))
		all_diffs.append(diff)
		if  diff > 1e-4:
			different = True

	px.histogram(all_diffs, range_x=[0, 1e-5], marginal='box').show()

	s = "DIFFERENT" if different else "SAME"
	print(s) 

executable = './mr-pr-mpi-base.o'
executable = sys.argv[1]

list_add = []
def run(filename, proc):
	if filename in no_4 and proc > 2:
		proc = 2
	if filename.startswith(('erdos', 'barabasi')):
		return
	cmd = ['mpirun', '-n', f'{proc}', executable, filename]
	if executable == './mr-pr-cpp.o':
		cmd = [executable, filename]
	print(cmd)
	p = subprocess.run(cmd, capture_output=True, text=True)
	# out = subprocess.check_output(cmd)
	# print(p.stdout)
	line = p.stdout.split('\n')[-2].split()
	print(line[1], line[-2])
	list_add.append([executable, filename, proc, line[-2], line[1]])


# filenames = []
# for x in os.listdir('test/'):
# 	if (not x.endswith('.txt')) or x.endswith(('-pr-p.txt', '-pr-j.txt')):
# 		continue
# 	filenames.append(x)

# all_tests = open('test/all-tests.txt').readlines() # REMOVE barabasi-10000

# all_tests = '''erdos-10000'''
# erdos-20000
# bull
# chvatal
# coxeter
# cubical
# diamond'''.split('\n')
no_4 = ['bull', 'diamond', 'house', 'housex', 'octahedral', 'tetrahedral'] # cannot use 4 procs
all_tests = ['erdos-10000']
print(all_tests)
proc = 2
if len(sys.argv) == 3:
	proc = int(sys.argv[2])

for f in all_tests:
	run(f.strip(), proc)
if os.path.isfile('results_a.csv'):
	df = pd.read_csv('results_a.csv', index_col=0)
	df = df.append(pd.DataFrame(list_add, columns=df.columns))
else:
	df = pd.DataFrame(list_add, columns=['executable', 'file', 'n_procs', 'time (s)', 'iters'])
print(df)
df.to_csv('results_a.csv')