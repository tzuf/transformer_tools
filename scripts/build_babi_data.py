#####
import os
import sys
import json

babi_data="etc/data/mix_babi"

SPLITS = [
    ("qa29_train.txt","train"),
    ("qa29_test.txt","test"),
    ("qa29_valid.txt","dev"),
]

if __name__ == "__main__":

    for file_name,split in SPLITS:
        full_path = os.path.join(babi_data,file_name)
        json_out = os.path.join(babi_data,"%s.jsonl" % split)
        with open(json_out,'w') as new_out:

            with open(full_path) as my_data:
                problem = []
            
                for k,line in enumerate(my_data):
                    identifier = "%d_%s" % (k,split)
                
                    line = line.strip()
                    line_num = line.split()[0]
                    detail = ' '.join(line.split()[1:])

                    if line_num == "1":
                        problem = []
                        problem.append(detail)
                    elif '?' in line:
                        print(line)
                        
                        
                    #     ## format current problem
                    #     if problem:
                    #         last_problem = problem[-1]
                    #         assert '?' in last_problem
                    #         question,answer = last_problem.split("?")
                    #         answer = answer.strip()
                    #         problem[-1] = "%s?" % question
                    #         problem_input = "%s $question$ %s" %\
                    #         (' '.join([p for p in problem[:-1] if '?' not in p]),problem[-1])

                    #         ## create json
                    #         json_dict = {}
                    #         json_dict["id" ] = identifier
                    #         json_dict["question"] = {}
                    #         json_dict["question"]["stem"] = problem_input
                    #         json_dict["answerKey"] = -1
                    #         json_dict["output"] = answer
                    #         json_dict["prefix"] = "answer:"

                    #         ###
                    #         new_out.write(json.dumps(json_dict))
                    #         new_out.write("\n")

                    #     # problem = []
                    #     # ## added first item 
                    #     # problem.append(detail)

                    # else:
                    #     problem.append(detail)
