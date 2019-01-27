# Imports
import sys
import os
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
        puts(colored.cyan('2: Testing..............................'))
        puts(colored.cyan('3: Help/Library Definitions.............'))
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
            print 'RUN CANARY'
            puts(colored.cyan('---------------------------------------------------'))
            loop=False
        elif input=='2':
            print 'TESTING...'
            puts(colored.cyan('---------------------------------------------------'))
            loop=False
        elif input=='3':
            print 'SEE CANARY.AM'
            puts(colored.cyan('---------------------------------------------------'))
            loop=False
        else:
            input = raw_input('Invalid Input, Select Option: ')
            

# Main
if __name__ == "__main__":
    # Launch Menu
    input()