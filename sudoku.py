# Sodoku Solver
# using CSP and MRV
# version 2
# uninformed default order vs CSP using MRV
# blind search iterates through each square one by one, checking rows, columns, and nearby squares.
# the CSP search uses two classes,.one to construct the problem and one to load the board and domains.
# the backtracking search function backtracks through the board using an MRV heuristic to prune possible decisions
# written by ComputerOnFire
from tkinter.filedialog import askopenfilename  # file browser gui
import re  # regular expressions. helps with parsing the text file.
# helps with iterating through blind search, and with printing the sudoku board for output.
import itertools
import copy
import sys
import textwrap  # pretty printing
from functools import reduce
# set the size of each grid square (3 for 3x3, 4 for 4x4)
GridSize = list(range(4))
# hexadecimal values, stored as a string
DOMAINS = '0123456789ABCDEF'


def merge(seqs):  # helper function that assmbles the sudoku board from individual cells
    return sum(seqs, [])


assigns = 0
# default variable selector agent. brute force uninformed search for sudoku.
# Simple left to right, top to bottom sudoku solver. uninformed, no heuristics. Takes input as a single line.


def checkBlock(x, y):  # checks squares in the same 4x4 block.
    return (x//64 == y//64 and x % 16//4 == y % 16//4)


def checkRow(x, y):  # checks squares in the same row
    return (x//16 == y//16)


def checkColumn(x, y):  # checks squares in the same column
    return (x-y) % 16 == 0


def blindSolve(board):
  global assigns
  # board.rstrip('\n')
  x = board.find('-')
  if x == -1:
    print('Uninformed Total Cell Assignments: ' + str(assigns))
    print()
    print('\n'.join(textwrap.wrap(' '.join(board), width=32)))
    pass
    input('Press enter to exit...')
    sys.exit()

  invalid = set()  # creates an empty set to hold the already placed numbers
  for y in range(256):  # 16x16. total size of the board
    if checkRow(x, y) or checkColumn(x, y) or checkBlock(x, y):
      invalid.add(board[y])

  for v in DOMAINS:
    if v not in invalid:  # checks if the value is in the invalid domains
      assigns = assigns + 1
      blindSolve(board[:x]+v+board[x+1:])


# changes the variable on the sudoku board to meet constraints.
fail = None
NextCell = itertools.count().__next__


# helper functions for default order search

# iterative solver. no backtracking or MRV heuristic.

def size(x):  # returns the length of a given list
    return len(x)


def firstValue(iterable, default=None):
    # finds the first value in an iterable list.
    return next(iter(iterable), default)


def firstEmpty(filled, Board):
    # default variable order solving. left to right, top to bottom.
    return firstValue([x for x in Board.variables if x not in filled])


BoxGrid = [[[[NextCell() for x in GridSize] for y in GridSize]
            for bx in GridSize] for by in GridSize]


# sudoku CSP class. functions for manipulating the sudoku board, constraints, domain, neighbors. everything except the search algorithm itself.


class Sudoku():
    def __init__(self, variables, domains, neighbors, constraints):
        # initates the CSP for the sudoku problem

        self.variables = list(domains.keys())
        self.domains = domains
        self.neighbors = neighbors
        self.constraints = constraints
        # used for uninformed default order search.
        self.iteration = constraints
        self.nassigns = 0
        self.availableCells = {
            v: list(self.domains[v]) for v in self.variables}

    def assign(self, cell, value, assignment):
        # assigns a value to a square.
        assignment[cell] = value
        # inciments the number of assignments used. used for calculating effeciency vs other algos.
        self.nassigns += 1

    def unassign(self, x, assignment):
        # unassigns a value from a square. used when backtracking.
        if x in assignment:
            del assignment[x]

    def conflicts(self, x, value, assignment):
        # returns the number of conflicts the variable has with other values
        def detect(y):
            return (y in assignment
                    and not self.constraints(x, value, y, assignment[y]))
        return count(detect(v) for v in self.neighbors[x])

    def output(self, assignment):
        def printBox(box): return [' '.join(
            map(printCell, row)) for row in box]

        def printCell(cell): return str(assignment.get(cell, '-'))
        def abut(lines1, lines2): return list(
            map(' '.join, list(zip(lines1, lines2))))
        print('\n'.join(  # makes the grid visible and easy to discern.
            '\n'.join(reduce(
                abut, map(printBox, brow))) for brow in self.cellGrid))

    def success(self, board):
        # checks if every empty square has been filled with every constraint met.
        current = dict(board)
        # print(current)
        return (size(current) == size(self.variables)
                and all(self.conflicts(x, current[x], current) == 0
                        for x in self.variables))

    def result(self, state, act):
        # executes an action, returns the result of said action.
        (cell, value) = act
        return state + ((cell, value),)

    def inferences(self, x, value):
        # finds inferences for  x = value
        removed = [(x, a) for a in self.availableCells[x] if a != value]
        self.availableCells[x] = [value]
        return removed

    def remove(self, x, value, removed):
        # prunes a path from available cells.
        self.availableCells[x].remove(value)
        if removed is not None:
            removed.append((x, value))

    def choices(self, x):
        # returns all valid variable options that have not been ruled out by the constraints.
        return (self.availableCells or self.domains)[x]

    def currentState(self):
        #returns the current state of the assignment.
        return {v: self.availableCells[v][0]
                for v in self.variables if 1 == len(self.availableCells[v])}

    def backStep(self, removed):
        #revert an action
        for B, b in removed:
            self.availableCells[B].append(b)


# returns true when inferences are NOT found.


def checkInference(sudoku, x, value, assignment, removed):
    return True


Rows = merge([list(map(merge, zip(*brow))) for brow in BoxGrid])


def differenceConstraint(A, a, B, b):
    #used to check if there is a difference between two neighbor values. required for sudoku problems.
    return a != b


Columns = list(zip(*Rows))


def pruneValues(sudoku, x, value, assignment, removed):
    #prunes possible neighbor values when the variable is not equal to remaining constraint values.
    for B in sudoku.neighbors[x]:
        if B not in assignment:
            for b in sudoku.availableCells[B][:]:
                if not sudoku.constraints(x, value, B, b):
                    sudoku.remove(B, b, removed)
            if not sudoku.availableCells[B]:
                return False
    return True


Boxes = merge([list(map(merge, brow)) for brow in BoxGrid])


def defaultOrder(x, filled, Board):
    #returns the values without ordering. default as given by the file input.
    return Board.choices(x)
# framework for setting up a board, loaded in as a string from the text file.


NeighborCells = {v: set() for v in merge(Rows)}
for x in map(set, Boxes + Rows + Columns):
    for v in x:
        NeighborCells[v].update(x - {v})


def count(seq):
    #counts the number of items that are true. used for checking values that follow constraints
    return sum(map(bool, seq))


def checkValidSquares(sudoku, x, assignment):
    if sudoku.availableCells:
        return len(sudoku.availableCells[x])
    else:
        return count(sudoku.conflicts(x, value, assignment) == 0
                     for value in sudoku.domains[x])


def findMin(seq, key=lambda x: x):
    #finds a the minimum value in a sequence of values.
    return min((seq), key=key)


def MRV(assignment, sudoku):
    #MRV heuristic.
    return findMin(
        [v for v in sudoku.variables if v not in assignment],
        key=lambda x: checkValidSquares(sudoku, x, assignment))


#backtracking search algorithm for solving CSPs, Using the above MRV. taken from page 215.


def backtrackSearch(sudoku,
                    chooseEmptyCell=firstEmpty,
                    domainOrder=defaultOrder,
                    inference=checkInference):
    def backtrack(assignment):  # main recursive backtracking function
        #print(assignment)
        if len(assignment) == len(sudoku.variables):
            return assignment
        x = chooseEmptyCell(assignment, sudoku)
        for value in domainOrder(x, assignment, sudoku):
            if sudoku.conflicts(x, value, assignment) == 0:
                sudoku.assign(x, value, assignment)
                removed = sudoku.inferences(x, value)
                if inference(sudoku, x, value, assignment, removed):
                    result = backtrack(assignment)
                    if result is not None:
                        return result
                sudoku.backStep(removed)
        sudoku.unassign(x, assignment)
        return None
    result = backtrack({})
    assert result is None or sudoku.success(result)
    return result


# constructs a sudoku board from a given text file (grid). uses a domain of 0-9,-F
class Board(Sudoku):

    def __init__(self, grid):
        # finds all characters from the textfile that are numbers/letters or -, this ignores newline characters
        self.squares = iter(re.findall(r'\w|\-', grid))
        # pulls rows, cols etc from the global variables, these are configured for a 16x16 (4x4) board.
        self.rows = Rows
        self.cols = Columns
        self.neighbors = NeighborCells
        self.GridSize = GridSize
        self.cell = NextCell
        self.cellGrid = BoxGrid
        self.boxes = Boxes
        self.domains = {x: [ch] if ch in DOMAINS else DOMAINS  # merges the valid characters loaded from the text file with the established row layout.
                        for x, ch in zip(merge(self.rows), self.squares)}
        Sudoku.__init__(self, None, self.domains,  # initalizes the CSP class with the problem domain and neighbor rules
                        self.neighbors, differenceConstraint)


# main function that runs when the program is executed.
if __name__ == '__main__':
    sudokuFile = askopenfilename()  # opens a GUI menu to choose the file
    # opens the file using the path chosen by the user
    sudokuFile = open(sudokuFile, 'r')
    sudoku1 = sudokuFile.read()  # reads the text file into memory
    sudokuFile.close()  # closes the open file
    sudoku2 = ''.join(line.rstrip()
                      for line in sudoku1)  # stripping newlines for blindSolve
    s2 = Board(sudoku2)
    s2.output(s2.currentState())
    print()
    print()
    print('Solving...')
    #solved using the CSP with MRV and backtracking agent.
    backtrackSearch(s2, chooseEmptyCell=MRV,
                    inference=pruneValues) is not fail
    #print(assigns)
    print('Backtracking/MRV Total cell assignments: ' + str(s2.nassigns))
    print()
    s2.output(s2.currentState())
    print()
    s1 = Board(sudoku1)
    s3 = copy.deepcopy(s1)
    #s1.output(s1.currentState())
    #default variable selector agent.
    print()
    print()
    print('Solving...')
    blindSolve(sudoku2)
    print()
    print()
    #s1.output(s1.currentState())
    #print('Total cell assignments: ' + str(assigns))
    print()
    # keeps the window open if not run in a command line
    input("Press enter to exit...")
    #blindSolve(sudoku2)
    #print(s2.output(s2.currentState()))
    #print(s2.nassigns)
