import subprocess
import re

def evaluate_map(resultfile):
    cmd = [ '../trec_eval.9.0/trec_eval', '../data/qrels-treceval.txt', resultfile ]
    output = subprocess.Popen( cmd, stdout=subprocess.PIPE ).communicate()[0]
    i = re.search("map", output).start()
    endl = re.search("\n", output[i:]).start()
    maprow = output[i:i+endl]
    return float(maprow.split('\t')[2])
