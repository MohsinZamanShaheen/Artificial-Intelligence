import piece
import numpy as np


class Board():
    """
    A class to represent a chess board.

    ...

    Attributes:
    -----------
    board : list[list[Piece]]
        represents a chess board
        
    turn : bool
        True if white's turn

    white_ghost_piece : tup
        The coordinates of a white ghost piece representing a takeable pawn for en passant

    black_ghost_piece : tup
        The coordinates of a black ghost piece representing a takeable pawn for en passant

    Methods:
    --------
    print_board() -> None
        Prints the current configuration of the board

    move(start:tup, to:tup) -> None
        Moves the piece at `start` to `to` if possible. Otherwise, does nothing.
        
    """

    def __init__(self, initState=None, xinit=True):
        """
        Initializes the modified board.
        """
        self.listNames = ['P', 'R', 'H', 'B', 'Q', 'K', 'P', 'R', 'H', 'B', 'Q', 'K']

        self.listSuccessorStates = []
        self.listNextStates = []

        self.board = []

        self.currentStateW = []
        self.currentStateB = []

        self.listVisitedStates = []

        self.board = []

        # Modified Board set-up (3x4)
        for i in range(3):
            self.board.append([None] * 4)

        self.board[1][1] = "X"

        # assign pieces
        if xinit:
            # White
            self.board[2][0] = piece.King(True)

            #Black
            self.board[0][0] = piece.King(False)

        # assign pieces
        else:
            self.currentState = initState

            # assign pieces
            for i in range(3):
                for j in range(4):
                    # White
                    if initState[i][j] == 6:  # 6 represents the King
                        self.board[i][j] = piece.King(True)
                        # store current state (Whites)
                        self.currentStateW.append([i, j, int(initState[i][j])])

                    if initState[i][j] == 12:
                        self.board[i][j] = piece.King(False)
                        self.currentStateB.append([i, j, int(initState[i][j])])

    def isSameState(self, a, b):

        isSameState1 = True
        # a and b are lists
        for k in range(len(a)):

            if a[k] not in b:
                isSameState1 = False

        isSameState2 = True
        # a and b are lists
        for k in range(len(b)):

            if b[k] not in a:
                isSameState2 = False

        isSameState = isSameState1 and isSameState2
        return isSameState


    def getListNextStatesW(self, mypieces):

        """
        Gets the list of next possible states given the currentStateW
        for each kind of piece
        
        """

        self.listNextStates = []

        # print("mypieces",mypieces)
        # print("len ",len(mypieces))
        for j in range(len(mypieces)):

            self.listSuccessorStates = []

            mypiece = mypieces[j]
            listOtherPieces = mypieces.copy()

            listOtherPieces.remove(mypiece)

            listPotentialNextStates = []

            #print("White: ->>>: ", str(self.board[mypiece[0]][mypiece[1]]))

            if (str(self.board[mypiece[0]][mypiece[1]]) == 'K'):

                #      print(" mypiece at  ",mypiece[0],mypiece[1])
                listPotentialNextStates = [[mypiece[0] + 1, mypiece[1], 6], \
                                           [mypiece[0] + 1, mypiece[1] - 1, 6], [mypiece[0], mypiece[1] - 1, 6], \
                                           [mypiece[0] - 1, mypiece[1] - 1, 6], \
                                           [mypiece[0] - 1, mypiece[1], 6], [mypiece[0] - 1, mypiece[1] + 1, 6], \
                                           [mypiece[0], mypiece[1] + 1, 6], [mypiece[0] + 1, mypiece[1] + 1, 6]]
                # check they are empty
                for k in range(len(listPotentialNextStates)):
                    aa = listPotentialNextStates[k]
                    if aa[0] > -1 and aa[0] < 8 and aa[1] > -1 and aa[1] < 8 and listPotentialNextStates[
                        k] not in listOtherPieces:
                        #and listPotentialNextStates[k] not in self.currentStateB:

                        if self.board[aa[0]][aa[1]] == None or not self.board[aa[0]][aa[1]].color:
                            self.listSuccessorStates.append([aa[0], aa[1], aa[2]])


            elif (str(self.board[mypiece[0]][mypiece[1]]) == 'P'):

                #       print(" mypiece at  ",mypiece[0],mypiece[1])
                listPotentialNextStates = [[mypiece[0], mypiece[1], 1], [mypiece[0] + 1, mypiece[1], 1]]
                # check they are empty
                for k in range(len(listPotentialNextStates)):

                    aa = listPotentialNextStates[k]
                    if aa[0] > -1 and aa[0] < 8 and aa[1] > -1 and aa[1] < 8 and listPotentialNextStates[
                        k] not in listOtherPieces:

                        if self.board[aa[0]][aa[1]] == None:
                            self.listSuccessorStates.append([aa[0], aa[1], aa[2]])


            elif (str(self.board[mypiece[0]][mypiece[1]]) == 'R'):

                #         print(" mypiece at  ",mypiece[0],mypiece[1])
                listPotentialNextStates = []

                ix = mypiece[0]
                iy = mypiece[1]

                while (ix > 0):
                    ix = ix - 1
                    #Ocupem la posició si és negra
                    if self.board[ix][iy] != None:
                        if not self.board[ix][iy].color:
                            listPotentialNextStates.append([ix, iy, 2])
                        break

                    elif self.board[ix][iy] == None :
                        listPotentialNextStates.append([ix, iy, 2])

                ix = mypiece[0]
                iy = mypiece[1]
                while (ix < 7):
                    ix = ix + 1
                    if self.board[ix][iy] != None:
                        if not self.board[ix][iy].color:
                            listPotentialNextStates.append([ix, iy, 2])
                        break

                    elif self.board[ix][iy] == None:
                        listPotentialNextStates.append([ix, iy, 2])

                ix = mypiece[0]
                iy = mypiece[1]
                while (iy > 0):
                    iy = iy - 1
                    if self.board[ix][iy] != None:
                        if not self.board[ix][iy].color:
                            listPotentialNextStates.append([ix, iy, 2])
                        break

                    elif self.board[ix][iy] == None:
                        listPotentialNextStates.append([ix, iy, 2])

                ix = mypiece[0]
                iy = mypiece[1]
                while (iy < 7):
                    iy = iy + 1
                    if self.board[ix][iy] != None:
                        if not self.board[ix][iy].color:
                            listPotentialNextStates.append([ix, iy, 2])
                        break

                    elif self.board[ix][iy] == None:
                        listPotentialNextStates.append([ix, iy, 2])

                        # check positions are not occupied - so far cannot kill pieces
                listPotentialNextStates
                for k in range(len(listPotentialNextStates)):

                    pos = listPotentialNextStates[k].copy()
                    pos[2] = 12
                    overlapping = False
                    """
                    if pos in self.currentStateB:
                        overlapping = True
                    """
                    if listPotentialNextStates[k] not in listOtherPieces and listPotentialNextStates[
                        k] and not overlapping:
                        self.listSuccessorStates.append(listPotentialNextStates[k])


            elif (str(self.board[mypiece[0]][mypiece[1]]) == 'H'):

                #         print(" mypiece at  ",mypiece[0]," ",mypiece[1]," ",3)
                listPotentialNextStates = []

                ix = mypiece[0]
                iy = mypiece[1]

                nextS = [ix + 1, iy + 2, 3]
                if nextS[0] > -1 and nextS[0] < 8 and nextS[1] > -1 and nextS[1] < 8:
                    self.listPotentialNextStates.append(nextS)
                nextS = [ix + 2, iy + 1, 3]
                if nextS[0] > -1 and nextS[0] < 8 and nextS[1] > -1 and nextS[1] < 8:
                    self.listPotentialNextStates.append(nextS)

                nextS = [ix + 1, iy - 2, 3]
                if nextS[0] > -1 and nextS[0] < 8 and nextS[1] > -1 and nextS[1] < 8:
                    self.listPotentialNextStates.append(nextS)
                nextS = [ix + 2, iy - 1, 3]
                if nextS[0] > -1 and nextS[0] < 8 and nextS[1] > -1 and nextS[1] < 8:
                    self.listPotentialNextStates.append(nextS)

                nextS = [ix - 2, iy - 1, 3]
                if nextS[0] > -1 and nextS[0] < 8 and nextS[1] > -1 and nextS[1] < 8:
                    self.listPotentialNextStates.append(nextS)
                nextS = [ix - 1, iy - 2, 3]
                if nextS[0] > -1 and nextS[0] < 8 and nextS[1] > -1 and nextS[1] < 8:
                    self.listPotentialNextStates.append(nextS)

                nextS = [ix - 1, iy + 2, 3]
                if nextS[0] > -1 and nextS[0] < 8 and nextS[1] > -1 and nextS[1] < 8:
                    self.listPotentialNextStates.append(nextS)

                nextS = [ix - 2, iy + 1, 3]
                if nextS[0] > -1 and nextS[0] < 8 and nextS[1] > -1 and nextS[1] < 8:
                    self.listPotentialNextStates.append(nextS)

                # check positions are not occupied
                for k in range(len(listPotentialNextStates)):

                    if listPotentialNextStates[k] not in listOtherPieces:
                        self.listSuccessorStates.append(listPotentialNextStates[k])



            elif (str(self.board[mypiece[0]][mypiece[1]]) == 'B'):

                #         print(" mypiece at  ",mypiece[0],mypiece[1], 4)
                listPotentialNextStates = []

                ix = mypiece[0]
                iy = mypiece[1]

                while (ix > 0 and iy > 0):
                    ix = ix - 1
                    iy = iy - 1
                    if self.board[ix][iy] != None:
                        listPotentialNextStates.append([ix, iy, 4])
                        break

                    elif self.board[ix][iy] == None:
                        listPotentialNextStates.append([ix, iy, 4])

                ix = mypiece[0]
                iy = mypiece[1]
                while (ix < 7 and iy > 0):
                    ix = ix + 1
                    iy = iy + 1
                    if self.board[ix][iy] != None:
                        listPotentialNextStates.append([ix, iy, 4])
                        break

                    elif self.board[ix][iy] == None:
                        listPotentialNextStates.append([ix, iy, 4])

                ix = mypiece[0]
                iy = mypiece[1]
                while (ix > 0 and iy < 7):
                    ix = ix - 1
                    iy = iy + 1
                    if self.board[ix][iy] != None:
                        listPotentialNextStates.append([ix, iy, 4])
                        break

                    elif self.board[ix][iy] == None:
                        listPotentialNextStates.append([ix, iy, 4])

                ix = mypiece[0]
                iy = mypiece[1]
                while (ix < 7 and iy < 7):
                    ix = ix + 1
                    iy = iy + 1
                    if self.board[ix][iy] != None:
                        listPotentialNextStates.append([ix, iy, 4])
                        break

                    elif self.board[ix][iy] == None:
                        listPotentialNextStates.append([ix, iy, 4])

                self.listSuccessorStates = listPotentialNextStates

            elif (str(self.board[mypiece[0]][mypiece[1]]) == 'Q'):

                #       print(" mypiece at  ",mypiece[0],mypiece[1])
                listPotentialNextStates = []

                # bishop wise
                ix = mypiece[0]
                iy = mypiece[1]

                while (ix > 0 and iy > 0):
                    ix = ix - 1
                    iy = iy - 1

                    if self.board[ix][iy] != None:
                        listPotentialNextStates.append([ix, iy, 5])
                        break

                    elif self.board[ix][iy] == None:
                        listPotentialNextStates.append([ix, iy, 5])

                ix = mypiece[0]
                iy = mypiece[1]
                while (ix < 7 and iy > 0):
                    ix = ix + 1
                    iy = iy + 1
                    if self.board[ix][iy] != None:
                        listPotentialNextStates.append([ix, iy, 5])
                        break

                    elif self.board[ix][iy] == None:
                        listPotentialNextStates.append([ix, iy, 5])

                ix = mypiece[0]
                iy = mypiece[1]
                while (ix > 0 and iy < 7):
                    ix = ix - 1
                    iy = iy + 1
                    if self.board[ix][iy] != None:
                        listPotentialNextStates.append([ix, iy, 5])
                        break

                    elif self.board[ix][iy] == None:
                        listPotentialNextStates.append([ix, iy, 5])

                ix = mypiece[0]
                iy = mypiece[1]
                while (ix < 7 and iy < 7):
                    ix = ix + 1
                    iy = iy + 1
                    if self.board[ix][iy] != None:
                        listPotentialNextStates.append([ix, iy, 5])
                        break

                    elif self.board[ix][iy] == None:
                        listPotentialNextStates.append([ix, iy, 5])

                        # Rook-like
                ix = mypiece[0]
                iy = mypiece[1]

                while (ix > 0):
                    ix = ix - 1
                    if self.board[ix][iy] != None:
                        listPotentialNextStates.append([ix, iy, 5])
                        break

                    elif self.board[ix][iy] == None:
                        listPotentialNextStates.append([ix, iy, 5])

                ix = mypiece[0]
                iy = mypiece[1]
                while (ix < 7):
                    ix = ix + 1
                    if self.board[ix][iy] != None:
                        listPotentialNextStates.append([ix, iy, 5])
                        break

                    elif self.board[ix][iy] == None:
                        listPotentialNextStates.append([ix, iy, 5])

                ix = mypiece[0]
                iy = mypiece[1]
                while (iy > 0):
                    iy = iy - 1
                    if self.board[ix][iy] != None:
                        listPotentialNextStates.append([ix, iy, 5])
                        break

                    elif self.board[ix][iy] == None:
                        listPotentialNextStates.append([ix, iy, 5])

                ix = mypiece[0]
                iy = mypiece[1]
                while (iy < 7):
                    iy = iy + 1
                    if self.board[ix][iy] != None:
                        listPotentialNextStates.append([ix, iy, 5])
                        break

                    elif self.board[ix][iy] == None:
                        listPotentialNextStates.append([ix, iy, 5])

                        # check positions are not occupied
                for k in range(len(listPotentialNextStates)):

                    if listPotentialNextStates[k] not in listOtherPieces:
                        self.listSuccessorStates.append(listPotentialNextStates[k])

                        # add other state pieces
            for k in range(len(self.listSuccessorStates)):
                self.listNextStates.append([self.listSuccessorStates[k]] + listOtherPieces)

        # for duplicates
        newList = self.listNextStates.copy
        newListNP = np.array(newList)

        # print("list nexts",self.listNextStates)

    def getListNextStatesB(self, mypieces):
        """
        Gets the list of next possible states given the currentStateW for the black king.
        """
        self.listNextStates = []

        for j in range(len(mypieces)):
            self.listSuccessorStates = []

            mypiece = mypieces[j]
            listOtherPieces = mypieces.copy()
            listOtherPieces.remove(mypiece)

            if (self.board[mypiece[0]][mypiece[1]].name == 'K'):
                listPotentialNextStates = [
                    [mypiece[0] + 1, mypiece[1], 12],
                    [mypiece[0] + 1, mypiece[1] - 1, 12],
                    [mypiece[0], mypiece[1] - 1, 12],
                    [mypiece[0] - 1, mypiece[1] - 1, 12],
                    [mypiece[0] - 1, mypiece[1], 12],
                    [mypiece[0] - 1, mypiece[1] + 1, 12],
                    [mypiece[0], mypiece[1] + 1, 12],
                    [mypiece[0] + 1, mypiece[1] + 1, 12]
                ]

                for k in range(len(listPotentialNextStates)):
                    aa = listPotentialNextStates[k]
                    if (
                            0 <= aa[0] < 3  # Check valid row index
                            and 0 <= aa[1] < 4  # Check valid column index
                            and listPotentialNextStates[k] not in listOtherPieces
                    ):
                        if self.board[aa[0]][aa[1]] is None or self.board[aa[0]][
                            aa[1]] == '':  # Check for None or empty string
                            self.listSuccessorStates.append([aa[0], aa[1], aa[2]])

            # Add other state pieces
            for k in range(len(self.listSuccessorStates)):
                self.listNextStates.append([self.listSuccessorStates[k]] + listOtherPieces)

    def print_board(self):
        """
        Prints the current state of the modified board.
        """

        buffer = ""
        for i in range(17):
            buffer += "*"
        print(buffer)
        for i in range(len(self.board)):
            tmp_str = "|"
            for j in self.board[i]:
                if j == None:
                    tmp_str += "   |"
                else:
                    tmp_str += f" {str(j)} |" if j != "X" else " X |"
            print(tmp_str)
        buffer = ""
        for i in range(17):
            buffer += "*"
        print(buffer)
