import numpy as np
from typing import List
from copy import deepcopy
from game import Move
import time




MOVESET = (Move.TOP, Move.BOTTOM, Move.RIGHT, Move.LEFT)


def print_state(game_state: tuple, optional=None) -> None:
    """
    Prints the Quixo board in a given state
    """
    string = ""
    for i in range(5):
        row = ""
        for j in range(5):
            index = j + 5 * i
            if game_state[index] == 0:
                row += "X "
            elif game_state[index] == 1:
                row += "O "
            else:
                row += "• "

        if i == 2 and not optional == None:
            row += "\t" + str(optional)
        string += row + "\n"
    print(string)


def print_states(game_state_list: List[tuple]):
    string = ""
    for i in range(5):
        row = ""
        for game_state in game_state_list:
            for j in range(5):
                if game_state[j + 5 * i] == 0:
                    row += "X "
                elif game_state[j + 5 * i] == 1:
                    row += "O "
                else:
                    row += "• "
            row += "\t\t"

        string += row + "\n"
    print(string)


def simple_slide(state: np.array, index: int, slide: Move):
    from_pos = (index // 5, index % 5)
    state = state.reshape(5,5)
    Moves = set(MOVESET)
    
    if from_pos[0] == 0:
        Moves.remove(Move.TOP)
    elif from_pos[0] == 4:
        Moves.remove(Move.BOTTOM)
    if from_pos[1] == 0:
        Moves.remove(Move.LEFT)
    elif from_pos[1] == 4:
        Moves.remove(Move.RIGHT)

    if not slide in Moves:
        return False
    # take the piece
    piece = state[from_pos]
    # if the player wants to slide it to the left
    if slide == Move.LEFT:
        # for each column starting from the column of the piece and moving to the left
        for i in range(from_pos[1], 0, -1):
            # copy the value contained in the same row and the previous column
            state[(from_pos[0], i)] = state[(from_pos[0], i - 1)]
        # move the piece to the left
        state[(from_pos[0], 0)] = piece
    # if the player wants to slide it to the right
    elif slide == Move.RIGHT:
        # for each column starting from the column of the piece and moving to the right
        for i in range(from_pos[1], state.shape[1] - 1, 1):
            # copy the value contained in the same row and the following column
            state[(from_pos[0], i)] = state[(from_pos[0], i + 1)]
        # move the piece to the right
        state[(from_pos[0], state.shape[1] - 1)] = piece
    # if the player wants to slide it upward
    elif slide == Move.TOP:
        # for each row starting from the row of the piece and going upward
        for i in range(from_pos[0], 0, -1):
            # copy the value contained in the same column and the previous row
            state[(i, from_pos[1])] = state[(i - 1, from_pos[1])]
        # move the piece up
        state[(0, from_pos[1])] = piece
    # if the player wants to slide it downward
    elif slide == Move.BOTTOM:
        # for each row starting from the row of the piece and going downward
        for i in range(from_pos[0], state.shape[0] - 1, 1):
            # copy the value contained in the same column and the following row
            state[(i, from_pos[1])] = state[(i + 1, from_pos[1])]
        # move the piece down
        state[(state.shape[0] - 1, from_pos[1])] = piece
        state = state.reshape((25))
    return True


def check_winner(state: tuple) -> int:
    """Check the winner. Returns the player ID of the winner if any, otherwise returns -1"""
    # for each row
    board = np.array(state).reshape((5, 5))

    for x in range(board.shape[0]):
        # if a player has completed an entire row
        if board[x, 0] != -1 and all(board[x, :] == board[x, 0]):
            # return the relative id
            return board[x, 0]
    # for each column
    for y in range(board.shape[1]):
        # if a player has completed an entire column
        if board[0, y] != -1 and all(board[:, y] == board[0, y]):
            # return the relative id
            return board[0, y]
    # if a player has completed the principal diagonal
    if board[0, 0] != -1 and all(
        [board[x, x] for x in range(board.shape[0])] == board[0, 0]
    ):
        # return the relative id
        return board[0, 0]
    # if a player has completed the secondary diagonal
    if board[0, -1] != -1 and all(
        [board[x, -(x + 1)] for x in range(board.shape[0])] == board[0, -1]
    ):
        # return the relative id
        return board[0, -1]
    return -1
INDEX_PERMUTATION_ROTATION = (20, 15, 10, 5, 0, 21, 16, 11, 6, 1, 22, 17, 12, 7, 2, 23, 18, 13, 8, 3, 24, 19, 14, 9, 4)

def rotate(game_state: tuple):
    """
    Function that returns the rotation of the game state like in the example
    """
    #  0  1  2  3  4    20 15 10  5  0
    #  5  6  7  8  9    21 16 11  6  1
    # 10 11 12 13 14    22 17 12  7  2
    # 15 16 17 18 19    23 18 13  8  3
    # 20 21 22 23 24    24 19 14  9  4

    # Senso orario
    return tuple(game_state[i] for i in INDEX_PERMUTATION_ROTATION)

INDEX_PERMUTATION_SYMMETRY = (4, 3, 2, 1, 0, 9, 8, 7, 6, 5, 14, 13, 12, 11, 10, 19, 18, 17, 16, 15, 24, 23, 22, 21, 20)

def symmetry(game_state: tuple):
    """
    Function that returns the vertical symmetry of the game state like the example
    """
    #  0  1  2  3  4      4  3  2  1  0
    #  5  6  7  8  9      9  8  7  6  5 
    # 10 11 12 13 14     14 13 12 11 10
    # 15 16 17 18 19     19 18 17 16 15 
    # 20 21 22 23 24     24 23 22 21 20

    return tuple(game_state[i] for i in INDEX_PERMUTATION_SYMMETRY)




def evaluation0(state: tuple, first_player):
    frontier = (0, 1, 2, 3, 4, 5, 9, 10, 14, 15, 19, 20, 24)
    sum_0 = 0
    sum_1 = 0
    for i in frontier:
        if state[i] == 0:
            sum_0 = sum_0 + 10
        elif state[i] == 1:
            sum_1 = sum_1 - 10
    if state[12] == 0:
        sum_0 = sum_0 + 5
    elif state[12] == 1:
        sum_1 = sum_1 - 5

    for i in (6, 8, 16, 18):
        if state[i] == 0:
            sum_0 = sum_0 + 3 
        elif state[i] == 1:
            sum_1 = sum_1 - 3        
    for i in (7, 11, 13, 17):
        if state[i] == 0:
            sum_0 = sum_0 + 2
        elif state[i] == 1:
            sum_1 = sum_1 - 2  
    if check_winner == first_player:
        sum_0 = sum_0 + 100
    else:
        sum_1 = sum_1 - 100                      
    if first_player:
        return sum_0            
    else:
        return sum_1
    

def evaluation1(state: tuple, first_player):
    sum_0 = 0
    sum_1 = 0
   
    for i in ( 1, 2, 3, 21, 22, 23):
        if state[i] == 0 and state[i-1] == 0:
            sum_0 = sum_0 + 10
        elif state[i] == 0 and state[i+1] == 0:
            sum_0 = sum_0 + 10

    for i in ( 1, 2, 3, 21, 22, 23):
        if state[i] == 1 and state[i-1] == 1:
            sum_1 = sum_1 - 10
        elif state[i] == 1 and state[i+1] == 1:
            sum_1 = sum_1 - 10 

    for i in ( 5, 10, 15, 9, 14, 19):
        if state[i] == 0 and state[i-5] == 0:
            sum_0 = sum_0 + 10
        elif state[i] == 0 and state[i+5] == 0:
            sum_0 = sum_0 + 10    

    for i in ( 5, 10, 15, 9, 14, 19):
        if state[i] == 1 and state[i-5] == 1:
            sum_1 = sum_1 - 10
        elif state[i] == 1 and state[i+5] == 1:
            sum_1 = sum_1 - 10     

    if state[12] == 0:
        sum_0 = sum_0 + 8
    elif state[12] == 1:
        sum_1 = sum_1 - 8

    for i in (1, 2, 3, 5, 7, 9, 10, 11, 13, 14, 15, 17, 19, 21, 22, 23):
        if state[i] == 0:
            sum_0 = sum_0 + 2
        elif state[i] == 1:
            sum_1 = sum_1 - 2
    for i in (0, 4, 20, 24, 6, 8, 16, 18):
        if state[i] == 0:
            sum_0 = sum_0 + 3 
        elif state[i] == 1:
            sum_1 = sum_1 - 3 

    if not first_player:
        return sum_0+sum_1            
    else:
        return sum_0+sum_1   


def evaluation2(state: tuple, first_player):
    sum_0 = 0
    sum_1 = 0
   
    for i in ( 1, 2, 3, 21, 22, 23):
        if state[i] == 0 and state[i-1] == 0:
            sum_0 = sum_0 + 10
        elif state[i] == 0 and state[i+1] == 0:
            sum_0 = sum_0 + 10

    for i in ( 1, 2, 3, 21, 22, 23):
        if state[i] == 1 and state[i-1] == 1:
            sum_1 = sum_1 - 10
        elif state[i] == 1 and state[i+1] == 1:
            sum_1 = sum_1 - 10 

    for i in ( 5, 10, 15, 9, 14, 19):
        if state[i] == 0 and state[i-5] == 0:
            sum_0 = sum_0 + 10
        elif state[i] == 0 and state[i+5] == 0:
            sum_0 = sum_0 + 10    

    for i in ( 5, 10, 15, 9, 14, 19):
        if state[i] == 1 and state[i-5] == 1:
            sum_1 = sum_1 - 10
        elif state[i] == 1 and state[i+5] == 1:
            sum_1 = sum_1 - 10     

    if state[12] == 0:
        sum_0 = sum_0 + 8
    elif state[12] == 1:
        sum_1 = sum_1 - 8

    for i in (1, 2, 3, 5, 7, 9, 10, 11, 13, 14, 15, 17, 19, 21, 22, 23):
        if state[i] == 0:
            sum_0 = sum_0 + 2
        elif state[i] == 1:
            sum_1 = sum_1 - 2
    for i in (0, 4, 20, 24, 6, 8, 16, 18):
        if state[i] == 0:
            sum_0 = sum_0 + 3 
        elif state[i] == 1:
            sum_1 = sum_1 - 3 

    for i in (6, 8, 12, 16, 18):
        if state[i]==0: 
            if state[i-1]==0:
               sum_0 = sum_0 + 1
            if state[i-5-1]==0:
                sum_0 = sum_0 + 1
            if state[i-5]==0:   
                sum_0 = sum_0 + 1 
            if state[i-5+1]==0:
                sum_0 = sum_0 + 1
            if state[i+1]==0:   
                sum_0 = sum_0 + 1     
            if state[i+5+1]==0:
                sum_0 = sum_0 + 1
            if state[i+5]==0:
                sum_0 = sum_0 + 1
            if state[i+5-1]==0: 
                sum_0 = sum_0 + 1
        if state[i]==1: 
            if state[i-1]==1:
                sum_1 = sum_1 - 1
            if state[i-5-1]==1:
                sum_1 = sum_1 - 1
            if state[i-5]==1:   
                sum_1 = sum_1 - 1 
            if state[i-5+1]==1:
                sum_1 = sum_1 - 1
            if state[i+1]==1:   
                sum_1 = sum_1 - 1    
            if state[i+5+1]==1:
                sum_1 = sum_1 - 1
            if state[i+5]==1:
                sum_1 = sum_1 - 1
            if state[i+5-1]==1: 
                sum_1 = sum_1 - 1                                   
    
    for i in (7, 11, 13, 17):
        if state[i]==0: 
            if state[i-1]==0:
               sum_0 = sum_0 + 1
            if state[i-5]==0:   
                sum_0 = sum_0 + 1 
            if state[i+1]==0:   
                sum_0 = sum_0 + 1     
            if state[i+5]==0:
                sum_0 = sum_0 + 1
        
        if state[i]==1: 
            if state[i-1]==1:
                sum_1 = sum_1 - 1
            if state[i-5]==1:   
                sum_1 = sum_1 - 1 
            if state[i+1]==1:   
                sum_1 = sum_1 - 1    
            if state[i+5]==1:
                sum_1 = sum_1 - 1  

    if not first_player:
        return sum_0+sum_1            
    else:
        return sum_0+sum_1    

def alpha_beta_pruning1(state, depth, player, alpha=-np.inf, beta=np.inf):
    terminal = check_winner(state)
    if terminal == 0:
        return 10000, (0,0), 0
    elif terminal == 1:
        return -10000, (0,0), 0
    
    if depth == 0:
        return evaluation1(state, player), 0, 0
    
    childs = feasible_moves(state, player)

    best_pos = 0
    best_slide = 0
    if player == 0:
        max_eval = -np.inf
        for child, i, slide in childs:
            eval, _, _ = alpha_beta_pruning1(np.array(child), depth-1, not player, alpha, beta)
            if max_eval < eval:
                max_eval = eval
                best_pos = i
                best_slide = slide
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval, best_pos, best_slide
    else:
        min_eval = np.inf
        for child, i, slide in childs:
            eval, _, _ = alpha_beta_pruning1(np.array(child), depth-1, not player, alpha, beta)
            if min_eval > eval:
                min_eval = eval
                best_pos = i
                best_slide = slide
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval, best_pos, best_slide    

def alpha_beta_pruning2(state, depth, player, alpha=-np.inf, beta=np.inf):
    terminal = check_winner(state)
    if terminal == 0:
        return 10000, (0,0), 0
    elif terminal == 1:
        return -10000, (0,0), 0
    
    if depth == 0:
        return evaluation2(state, player), 0, 0
    
    childs = feasible_moves(state, player)

    best_pos = 0
    best_slide = 0
    if player == 0:
        max_eval = -np.inf
        for child, i, slide in childs:
            eval, _, _ = alpha_beta_pruning2(np.array(child), depth-1, not player, alpha, beta)
            if max_eval < eval:
                max_eval = eval
                best_pos = i
                best_slide = slide
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval, best_pos, best_slide
    else:
        min_eval = np.inf
        for child, i, slide in childs:
            eval, _, _ = alpha_beta_pruning2(np.array(child), depth-1, not player, alpha, beta)
            if min_eval > eval:
                min_eval = eval
                best_pos = i
                best_slide = slide
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval, best_pos, best_slide    



BORDER = (0, 1, 2, 3, 4, 5, 9, 10, 14, 15, 19, 20, 21, 22, 23, 24)

def feasible_moves(state, player: int):
    childs = list()
    set_childs = set()
    
    for i in BORDER:
        
        if state[i] == -1 or state[i] == player:
            for slide in MOVESET:
                state1 = deepcopy(state)
                state1[i] = player
                if simple_slide(state1, i, slide):
                    child = tuple(state1)
                    if not child in set_childs:
                        childs.append((child, i, slide))
                        set_childs = set_childs.union({
                        child,
                        rotate(child),
                        rotate(rotate(child)),
                        rotate(rotate(rotate(child))),
                        symmetry(child),
                        symmetry(rotate(child)),
                        symmetry(rotate(rotate(child))),
                        rotate(symmetry(child))
                    })
            
    return childs                   





if __name__ == "__main__":
    start = time.time()
    s = np.ones((25))*-1
    s1 = np.array([0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, -1, 0, 0, 0, 0, 1, 1, 1, 0])
    print(alpha_beta_pruning2(s1, 2, True))
    print(time.time()-start)

   
    
