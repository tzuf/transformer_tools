### script for converting babi-style files to my json format used here
import json
import sys
import re
import os
from optparse import OptionParser

USAGE = """usage: python babi2json [options] [--help]"""

CONFIG = OptionParser(usage=USAGE)
CONFIG.add_option("--data_loc",dest="data_loc",default='',
                      help="the location of the data [default='']")
CONFIG.add_option("--odir",dest="odir",default='',
                      help="The output directory where to put files [default='']")


NUM_RE = re.compile(r"(\d+)") # to capture supporting facts idxs

def main(argv):
    config,_ = CONFIG.parse_args(sys.argv[1:])

    if not config.data_loc or not os.path.isdir(config.data_loc):
        exit('Please put a valid data directory, via `--data_dir`')
    if not config.odir or not os.path.isdir(config.odir):
        exit('Please specify a valid output directory')

    for split in ["train","test","valid"]:
        target = [f for f in os.listdir(config.data_loc) if split in f and ".txt" in f]
        assert len(target) <= 1, "multiple target files found!"
        if target:
            target = target[0]
        else:
            continue

        ## output
        split_name = split if split != "valid" else "dev"
        ofile = os.path.join(config.odir,"%s.jsonl" % split_name)

        with open(ofile,'w') as new_out: 
        
            with open(os.path.join(config.data_loc,target)) as my_data:
                problem = []
                sub_question = 0
                story_idx = 0 # counts number of questions (=stories)
            

                for k,line in enumerate(my_data):
                    identifier = "%d_%s" % (k,split)
                    line = line.strip()
                    line_num = line.split()[0]
                    detail = ' '.join(line.split()[1:])

                    if line_num == "1":
                        problem = []
                        sub_question = 0
                        problem.append(detail)

                    elif "?" in line:
                        question,answer = detail.split("?")
                        
                        # extract supporting facts idxs if they exist
                        supp_facts = [int(x) for x in NUM_RE.findall(answer)]
                        
                        answer = ' '.join([i for i in answer.strip().split() if not i.isnumeric()])
                        question = "%s?" % question

                        #print(answer)
                        assert len(answer.split()) == 1, answer
                        
                        


                        problem_input = "%s $question$ %s" %\
                            (' '.join([p for p in problem if '?' not in p]),question)

                        json_dict = {}
                        json_dict["id" ] = "%s_sub_question_%s" % (identifier,sub_question)
                        json_dict["question"] = {}
                        json_dict["question"]["stem"] = problem_input
                        json_dict["answerKey"] = -1
                        json_dict["output"] = answer
                        json_dict["prefix"] = "answer:"
                        json_dict["input"] = problem_input
                        json_dict["story_idx"] = story_idx
                        json_dict["supporting_facts"] = supp_facts

                        new_out.write(json.dumps(json_dict))
                        new_out.write('\n')

                        ### 
                        sub_question += 1
                        story_idx += 1

                    else:
                        problem.append(detail)

    
if __name__ == "__main__":
    main(sys.argv[1:])