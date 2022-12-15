import subprocess


dimensions = [3, 10, 500, 2000, 4000, 7000, 8000]
dname = {3:'3', 10:'10', 500:'500', 2000:'2k', 4000:'4k', 8000:'8k'}

for d in dimensions:
	subprocess.run(["hyperfine", "-L", "threads", "1,2,4,8,16", f"\"./omp {d} {{threads}} > txts/omp-{dname[d]}-{{threads}}.txt\"", "--export-csv", f"omp-res-{dname[d]}.csv"])

for d in dimensions:
	subprocess.run(["hyperfine", "-L", "threads", "1,2,4,8,16", f"\"./pth {d} {{threads}} > txts/omp-{dname[d]}-{{threads}}.txt\"", "--export-csv", f"pth-res-{dname[d]}.csv"])


# hyperfine -L threads 1,2,4,8,16 "omp.exe 4000 {threads} >> txts\4k-{threads}.txt" --export-csv omp-res-4k.csv
# hyperfine -L threads 1,2,4,8,16 "omp.exe 8000 {threads} >> txts\8k-{threads}.txt" --export-csv omp-res-8k.csv --runs 1