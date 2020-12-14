import os
import sys
import traceback
from transformer_tools import load_module

USAGE = "python -m transformer_tools mode [options]"

### library `modes`
MODES = {}
MODES["T5Classifier"]       = "transformer_tools.T5Classification"
MODES["T5Generator"]        = "transformer_tools.T5Generative"
MODES["T5Generative"]       = "transformer_tools.T5Generative"
MODES["NER"]                = "transformer_tools.NER"

def main(argv):
    """The main execution point 

    :param argv: the cli input
    """
    if not argv:
        exit('Please specify mode and settings! Current modes="%s" exiting...' % '; '.join(MODES))

    ## check the mode 
    mode = MODES.get(argv[0],None)
    if mode is None:
        exit('Unknown mode=%s, please choose from `%s`' % (argv[0],'; '.join(MODES)))

    ## try to execute the target module
    try:
        mod = load_module(mode)
        mod.main(argv)
    except Exception as e:
        print("Uncaught error encountered during execution!",file=sys.stderr)
        traceback.print_exc(file=sys.stdout)
        raise e
    finally:
        ## close the stdout
        if sys.stdout != sys.__stdout__:
            sys.stdout.close()
        if sys.stderr != sys.__stderr__:
            sys.stderr.close()

if __name__ == "__main__":
    main(sys.argv[1:])
