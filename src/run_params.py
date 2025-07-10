import argparse
from pprint import pprint
import ast

def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_num', type=int, default=0, help="model number")
    parser.add_argument('--prompts', type=arg_as_list, default=[], help="prompts")

    args = parser.parse_args()
    print('Called with args:')
    pprint(vars(args), indent=2)

    return args