### script for converting babi-style files to my json format used here
import json
import sys
import os
from optparse import OptionParser

USAGE = """usage: python babi2json [options] [--help]"""

CONFIG = OptionParser(usage=USAGE)
CONFIG.add_option("--data_loc",dest="data_loc",default='',
                      help="the location of the data [default='']")
CONFIG.add_option("--odir",dest="odir",default='',
                      help="The output directory where to put files [default='']")

def main(argv):
    config,_ = CONFIG.parse_args(sys.argv[1:])

    if not config.data_loc or not os.path.isdir(config.data_loc):
        exit('Please put a valid data directory, via `--data_dir`')
    if not config.odir or not os.path.isdir(config.odir):
        exit('Please specify a valid output directory')

    for split in ["train","test","valid"]:
        target = [f for f in os.listdir(config.data_loc) if split in f]
        assert len(target) == 1, "multiple target files found!"
        target = target[0]

        ## output
        split_name = split if split != "valid" else "dev"
        ofile = os.path.join(config.odir,"%s.jsonl" % split_name)

        with open(ofile,'w') as new_out: 
        
            with open(os.path.join(config.data_loc,target)) as my_data:
                problem = []
            

                for k,line in enumerate(my_data):
                    identifier = "%d_%s" % (k,split)
                    line = line.strip()
                    line_num = line.split()[0]
                    detail = ' '.join(line.split()[1:])

                    if line_num == "1":
                        if problem: 
                            ## format current problem
                    
                            last_problem = problem[-1]
                            assert '?' in last_problem
                            question,answer = last_problem.split("?")
                            answer = answer.strip()
                            problem[-1] = "%s?" % question
                            problem_input = "%s $question$ %s" %\
                              (' '.join([p for p in problem[:-1] if '?' not in p]),problem[-1])
                      
                            ## create json
                            json_dict = {}
                            json_dict["id" ] = identifier
                            json_dict["question"] = {}
                            json_dict["question"]["stem"] = problem_input
                            json_dict["input"]            = problem_input
                            json_dict["answerKey"] = -1
                            json_dict["output"] = answer
                            json_dict["prefix"] = "answer:"

                            ## print to out
                            new_out.write(json.dumps(json_dict))
                            new_out.write('\n')

                        problem = []
                        ## added first item 
                        problem.append(detail)

                    else:
                        problem.append(detail)

    
if __name__ == "__main__":
    main(sys.argv[1:])
