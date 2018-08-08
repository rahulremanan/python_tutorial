# !/usr/bin/python3.6
# -*- coding: utf-8 -*-
"""
Python script that enables to run underlying linux environment commands as an
input list.
Created on Sat Jul 21 22:48:11 2018

@author: Dr. Rahul Remanan, Moad Computer (https://www.moad.computer)
@support: info@moad.computer

"""
import argparse
import subprocess
import sys


def string_to_bool(val):
    """
    A function that checks if an user argument is boolean or not.

    Example usage:

            import argsparse

            a = argparse.ArgumentParser()
            a.add_argument("--some_bool_arg",
               help = "Specify a boolean argument ...",
               dest = "some_bool_arg",
               required=False,
               default=[True],
               nargs=1,
               type = string_to_bool)

            args = a.parse_args()
            args = get_user_options()
    """
    if val.lower() in ('yes', 'true', 't', 'y', '1', 'yeah', 'yup'):
        return True
    elif val.lower() in ('no', 'false', 'f', 'n', '0', 'none', 'nope'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected ...')
    pass


def execute_in_shell(command=None, verbose=False):
    """
    This is a function that executes shell scripts from within python.

    Example usage:
    execute_in_shell(command = ['ls ./some/folder/',
                                'ls ./some/folder/  -1 | wc -l'],
                     verbose = True )

    This command returns dictionary with elements: Output and Error.

    Output records the console output,
    Error records the console error messages.

    :param command: takes a list of shell commands
    :param verbose: takes a boolean value to set verbose level
    :return: Dictionary with two elements Output and Error
    """

    error = []
    output = []

    if isinstance(command, list):
        for i in range(len(command)):
            try:
                process = subprocess.Popen(command[i], shell=True,
                                           stdout=subprocess.PIPE)
                process.wait()
                out, err = process.communicate()
                error.append(err)
                output.append(out)

                if error[0] is not None:
                    print("ERROR: running command {}".format(command[i]))
                    print(err)
                    exit(1)

                if verbose:
                    print('Success running shell command: {}'
                          ''.format(command[i]))
            except Exception as e:
                print('Failed running shell command: {}'.format(command[i]))
                if verbose:
                    print(type(e))
                    print(e.args)
                    print(e)

    else:
        print('The argument command takes a list input ...')
    return {'Output': output, 'Error': error}


def get_user_options():
    a = argparse.ArgumentParser()
    help_str = "Specify if the commands should be passed to linux shell ..."
    a.add_argument("--execute_shell",
                   help=help_str,
                   dest="execute_shell",
                   required=True,
                   type=string_to_bool,
                   nargs=1)

    help_str = "Specify list of commands to be passed onto the linux shell ..."
    a.add_argument("--command",
                   help=help_str,
                   dest="command",
                   required=True,
                   nargs=1)

    a.add_argument("--verbose",
                   help="Specify verbose level ...",
                   dest="verbose",
                   required=True,
                   type=string_to_bool,
                   nargs=1)

    arguments = a.parse_args()
    return arguments


if __name__ == "__main__":
    args = get_user_options()
    execute_shell = args.execute_shell[0]
    if execute_shell:
        print("Shell session initiated ...")
        print(execute_in_shell(command=[str(args.command[0])],
                               verbose=args.verbose[0]))
    else:
        print("Nothing to do here ...")
        print("Try setting the --execute_in_shell flag to True ...")
        print("For more help, run with -h flag ...")
        sys.exit(1)
