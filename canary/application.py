# Imports
import sys
import os
from canaryTest import CanaryBratAnalysis
from canaryLib import canaryLocal
# Cosmetic Imports for Terminal Application (Dev mode)
from pyfiglet import Figlet
from clint.arguments import Args
from clint.textui import puts, colored, indent

def menu():
    # Banner
    f = Figlet(font='standard')
    with indent(6):
        puts(colored.cyan(f.renderText('Canary')))

    # Menu
    puts(colored.cyan('---------------------------------------------------'))
    with indent(5):
        puts(colored.cyan('1: Find Arguments.......................'))
        puts(colored.cyan('2: Testing (Canary vs Manual Analysis)..'))
        puts(colored.cyan('3: String Extraction (BRAT).............'))
        puts(colored.cyan('4: Help/Library Definitions.............'))
    puts(colored.cyan('---------------------------------------------------'))

    

def input():
    loop = True

    while loop:
        # Opens Menu
        menu()
        # Gets User Input
        input = raw_input('Select Option: ')
        puts(colored.cyan('---------------------------------------------------'))

        if input=='1':
            print("CANARY DEV BUILD")
            puts(colored.cyan('---------------------------------------------------'))
            loop=False
            # User Input (For local File)
            input = raw_input("Please enter the file you with to Mine from the Corpus Folder e.g. essay001.txt: ")
            # Paths to file (input/fixed path)
            test_input = os.path.isfile('.././corpus/' + input)
            puts(colored.cyan('---------------------------------------------------'))
            # Testing to see if the file exists
            if test_input:
                path = ('.././corpus/' + input)
                canaryLocal(path)
                puts(colored.cyan('---------------------------------------------------'))
            else:
                print "File does not exist in the directory 'corpus', please try again!"
                input = raw_input("Please enter the file you with to Mine from the Corpus Folder e.g. essay001.txt: ")
                puts(colored.cyan('---------------------------------------------------'))
        elif input=='2':
            loop=False 
            # Run individual test to compare results from Canary vs Manual Analysis
            CanaryBratAnalysis()
        elif input=='3':
            loop=False
            print 'MOVE CODE FROM OTHER CANARY INTO HERE WITH THE STRINGS OF ARGS'
        elif input=='4':
            print 'SEE CANARY.AM'
            puts(colored.cyan('---------------------------------------------------'))
            loop=False
        else:
            input = raw_input('Invalid Input, Select Option: ')
            

# Main
if __name__ == "__main__":
    # Launch Menu
    input()