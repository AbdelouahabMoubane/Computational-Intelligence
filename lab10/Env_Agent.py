#Copyright(c) 2023 Abdelouahab Moubane <abdelmub@gmail.com> https://github.com/AbdelouahabMoubane
#Copyright(c) 2023 Edoardo Vay  <vay.edoardo@gmail.com> <https://github.com/Edoxy>



import numpy as np
import random
from dataclasses import dataclass
from typing import List, Tuple, Callable, Dict
from enum import Enum
from tqdm.auto import tqdm
from IPython.display import clear_output
from itertools import combinations
from collections import defaultdict


class StateType(Enum):
    WIN = 2 # means the X player wins
    LOSS = -2 # means the O player wins
    DRAW = 1
    IN_PROGRESS = 0


@dataclass
class ttt_node:
    """
    DataClass defining a node in the Tic-tac-toe Tree structure: all the possible positions are encoded in a tuple
    """
    state: tuple # This tuple encodes the position: -1 for a free space, 0 for X and 1 for O
    childs: Dict[tuple, any] # Dictionary with the position reachable from this position: keys are the tuple encoding the position
    state_type: StateType = StateType.DRAW # State type of the node
    """
    IMPORTANT: state_type is referred always at the player that begin the game (the player that plays with the 'X'), so 
    if StateType = StateType.WIN the first player wins
    if StateType = StateType.LOSS the first player wins
    """


def print_state(game_state: tuple, optional=None) -> None:
    """
    Prints the Tic Tac Toe board in a given state
    """
    string = ""
    for i in range(3):
        row = ""
        for j in range(3):
            index = j + 3 * i
            if game_state[index] == 0:
                row += "X "
            elif game_state[index] == 1:
                row += "O "
            else:
                row += "• "

        if i == 1 and not optional == None:
            row += "\t" + str(optional)
        string += row + "\n"
    print(string)


def print_2_states(game_state1: tuple, game_state2: tuple):

    string = ""
    for i in range(3):
        row = ""
        for j in range(3):
            if game_state1[j + 3 * i] == 0:
                row += "X "
            elif game_state1[j + 3 * i] == 1:
                row += "O "
            else:
                row += "• "

        row += "\t\t"
        for j in range(3):
            if game_state2[j + 3 * i] == 0:
                row += "X "
            elif game_state2[j + 3 * i] == 1:
                row += "O "
            else:
                row += "• "
        string += row + "\n"
    print(string)


def check_same_state(game_state1, game_state2, n_rotations=0, n_symmetry=0):
    """
    Recursive function that checks whether two states are equal except for rotations and symmetries
    """
    if game_state1 == game_state2:
        return True
    elif n_rotations == 3:
        return False
    elif n_symmetry == 2:
        return False
    else:
        game_state2_rotated = rotate(game_state2)
        game_state2_symmetric = symmetry(game_state2)
        ROTATION = check_same_state(
            game_state1,
            game_state2_rotated,
            n_rotations=n_rotations + 1,
            n_symmetry=n_symmetry,
        )
        if ROTATION:
            return True
        SYMMETRY = check_same_state(
            game_state1,
            game_state2_symmetric,
            n_rotations=n_rotations,
            n_symmetry=n_symmetry + 1,
        )
        return ROTATION or SYMMETRY
        

def state_in_list(game_state, list_game_states):
    """
    Function that checks that the game state is in a list of game states, except for rotations and symmetries
    """
    result = False
    for game in list_game_states:
        result = result or check_same_state(game, game_state)

    return result


def ply(index: int, state: List[int], player_id: int) -> bool:
    """
    Function that executes a move on a state 
    """
    if state[index] == -1:
        is_valid = True
        state[index] = player_id
    else:
        is_valid = False
    return is_valid


def is_terminal(game_state: tuple, player_id: bool) -> bool:
    """
    Function that checks if in a game state, there is a winner, so the game is over 
    """
    VALUES = [4, 9, 2, 3, 5, 7, 8, 1, 6]
    score = [VALUES[i] for i in range(9) if game_state[i] == player_id]
    n_moves = len(score)
    
    if n_moves < 3:
        return False
    
    else:
        return any(sum(c) == 15 for c in combinations(score, 3))


def rotate(game_state: tuple):
    """
    Function that returns the rotation of the game state like in the example
    """
    # 1 2 3     7 4 1
    # 4 5 6 ->  8 5 2
    # 7 8 9     9 6 3
    # Senso orario
    INDEX_PERMUTATION = np.array([7, 4, 1, 8, 5, 2, 9, 6, 3]) - 1
    return tuple(game_state[INDEX_PERMUTATION[i]] for i in range(9))


def symmetry(game_state: tuple):
    """
    Function that returns the vertical symmetry of the game state like the example
    """
    # 1 2 3     3 2 1
    # 4 5 6     6 5 4
    # 7 8 9     9 8 7
    # verticale
    INDEX_PERMUTATION = np.array([3, 2, 1, 6, 5, 4, 9, 8, 7]) - 1
    return tuple(game_state[INDEX_PERMUTATION[i]] for i in range(9))


def random_game(root: ttt_node):
    """
    Function that gives the strategy of a random player
    """
    while not root.childs == dict():
        root_state = random.choice(list(root.childs.keys()))
        root = root.childs[root_state]
        print_state(root.state)
        print(root.state_type)


def create_tree(root_node: ttt_node, depth=0):
    """
    The function recursively builds a tree of possible moves,
    exploring all game options up to a terminal state (win, lose, or draw). 
    For each valid move, it creates a child node representing the next state of the game 
    and continues until all available game possibilities are explored.
    """
    current_player = (depth) % 2
    if is_terminal(root_node.state, 1 - current_player):
        if 1 - current_player == 0:
            root_node.state_type = StateType.WIN
        else:
            root_node.state_type = StateType.LOSS
        return
    for i in range(9):
        tmp_game = list(root_node.state)
        if ply(i, tmp_game, depth % 2) and not state_in_list(
            tmp_game, root_node.childs.keys()
        ):
            root_node.state_type = StateType.IN_PROGRESS
            child = tuple(tmp_game)
            root_node.childs[child] = ttt_node(child, dict())
            create_tree(root_node.childs[child], depth=depth + 1)
    return


class TicTacToe_Env:
    def __init__(self, init_node: ttt_node) -> None:
        self.root = init_node
        self.CurrentState = init_node
        self.payoff_win = 150
        self.payoff_loss = -100
        self.payoff_draw = -50
        self.player_id = 1

        self.Feasible_Actions = list(self.CurrentState.childs.keys())

    def game_reset(self, agent=None):
        """
        Function manages the advancing to the next game state, 
        considering the agent (if any) or randomly choosing a move from the available possibilities.
        Next, update information about the current player and possible actions before resetting the current player identifier.
        """

        if self.player_id == 1:
            self.CurrentState = self.root
        else:
            if agent == None:
                self.CurrentState = self.root.childs[
                    random.choice(list(self.root.childs.keys()))
                ]
            else:
                self.CurrentState = self.root.childs[agent.policy(self.root)]

        self.player_id = 1 - self.player_id
        self.Feasible_Actions = list(self.CurrentState.childs.keys())
        return self.player_id

    def get_FeasibleAction(self):
        return list(self.CurrentState.childs.keys())

    def TakeAction(self, action: tuple, agent=None) -> Tuple[float, ttt_node, bool]:
        FLAG = True
        self.CurrentState = self.CurrentState.childs[action]

        state = self.CurrentState.state_type

        if state == StateType.WIN or state == StateType.LOSS:
            payoff = self.payoff_win
        elif state == StateType.DRAW:
            payoff = self.payoff_draw
        else:
            if agent == None:
                Action = random.choice(self.get_FeasibleAction())
            else:
                Action = agent.policy(self.CurrentState)

            self.CurrentState = self.CurrentState.childs[Action]
            state = self.CurrentState.state_type
            if state == StateType.WIN or state == StateType.LOSS:
                payoff = self.payoff_loss
            elif state == StateType.DRAW:
                payoff = self.payoff_draw
            else:
                payoff = 0
                FLAG = False

        return payoff, self.CurrentState, FLAG


class Agent:
    def __init__(self, env: TicTacToe_Env) -> None:
        self.Env = env
        self.QFactor = defaultdict(float)
        return

    #implementation of Q-learnig

    def QLearning(self, discount, alpha, epsilon=0.2, n_step=100, agent="random"):

        """
        Implementation of

        Q(s,a) ← (1-alpha) * Q(s,a) + alpha * (instant_payoff + discount * maxQ(s',a'))

        alpha is the learning rate (0 < alpha ≤ 1), which indicates how much the agent must take new information 
        into account compared to pre-existing information

        istant_payoff is the reward obtained by performing the action a in state s

        discount factor (0 < discount ≤ 1), which determines the importance of future rewards compared to immediate ones

        s' it is the next state after performing the action a

        a' is a feasible action in state s'
        
        """
        Action = None
        n_moves = 0
        for _ in tqdm(range(n_step)):

            current_state = self.Env.CurrentState.state

            qf, Action = self.FindBest(self.Env.CurrentState)
            if np.random.rand() < epsilon and qf < discount**(2-n_moves)*self.Env.payoff_win:
                Action = random.choice(self.Env.get_FeasibleAction())

            if not agent == "random": #and np.random.rand() < epsilon:
                instant_payoff, newState, flag = self.Env.TakeAction(Action, agent)
            else:
                instant_payoff, newState, flag = self.Env.TakeAction(Action)
            n_moves += 1

            if flag:
                self.Env.game_reset()
                n_moves = 0
                qtilde = instant_payoff
            else:
                next_payoff, _ = self.FindBest(newState)
                qtilde = instant_payoff + discount * next_payoff

            self.QFactor[(current_state, Action)] = (
                alpha * qtilde
                + (1 - alpha) * self.QFactor[(current_state, Action)]
            )

            current_state = newState

        return

    def FindBest(self, State: ttt_node):
        """
        Funtion that finds the best q_factor beetween q_factors(i,a), 
        where 'i' is a next state and 'a' is a feasible action.
        
        """

        ActionList = list(State.childs.keys())
        qf = np.zeros((len(ActionList)))

        for i, Action in enumerate(ActionList):
            qf[i] = self.QFactor[(State.state, Action)]

        best_index = np.argmax(qf)

        contribution = qf[best_index]
        action = ActionList[best_index]
        return contribution, action

    def policy(self, State: ttt_node):
        _, Action = self.FindBest(State)
        return Action

    def play_game(self, flag_agent=False):
        if flag_agent:
            self.Env.game_reset(self)
        else:
            self.Env.game_reset()
        END = False
        while not END:
            _, Action = self.FindBest(self.Env.CurrentState)
            if flag_agent:
                _, NewState, END = self.Env.TakeAction(Action, self)
            else:
                _, NewState, END = self.Env.TakeAction(Action)
            print_2_states(Action, NewState.state)
        print(self.Env.CurrentState.state_type)

    def play_games(self, n_games, flag_agent=False):
        """
        Function that makes to play n games and calculate the rate of wins, of draws and of loses

        """
        n_wins = 0
        n_losses = 0
        n_draws = 0

        for _ in range(n_games):
            player_id = 0
            if flag_agent:
                player_id = self.Env.game_reset(self)
            else:
                player_id = self.Env.game_reset()

            END = False
            while not END:
                _, Action = self.FindBest(self.Env.CurrentState)
                if flag_agent:
                    _, _, END = self.Env.TakeAction(Action, self)
                else:
                    _, _, END = self.Env.TakeAction(Action)
            result = self.Env.CurrentState.state_type

            if (result == StateType.WIN and player_id == 0) or (
                result == StateType.LOSS and player_id == 1
            ):
                n_wins += 1
            elif result == StateType.DRAW:
                n_draws += 1
            else:
                n_losses += 1

        return n_wins / n_games, n_draws / n_games, n_losses / n_games

    def play_game_Human(self):

        """
        Function that allows the user to play against our agent
        """

        self.Env.game_reset(self)

        END = False
        while not END:
            print_state(self.Env.CurrentState.state)
            print("-YOUR OPTIONS-")
            Actions = self.Env.get_FeasibleAction()
            for i, s in enumerate(Actions):
                print_state(s, i)
            Action_index = int(input("Choose your move index: "))
            Action = Actions[Action_index]
            _, _, END = self.Env.TakeAction(Action, self)
            clear_output(wait=True)
        print_state(self.Env.CurrentState.state)
        print(self.Env.CurrentState.state_type)


if __name__ == "__main__":
    StartingPosition = ttt_node(tuple(-1 for _ in range(9)), dict())
    create_tree(StartingPosition)
    env = TicTacToe_Env(StartingPosition)
    discount = 0.9
    alpha = 1
    agente = Agent(env)
    env.game_reset()
    agente.QLearning(discount=discount, alpha=alpha, epsilon=0, n_step=1_000)
    for _ in tqdm(range(100)):
        env.game_reset()
        agente.QLearning(
            discount=discount, alpha=alpha, epsilon=0.7, n_step=1_000, agent=agente
        )

    WR, DR, LR = agente.play_games(100_000, flag_agent=True)
    print(f"WinRate: {WR*100:.4}% DrawRate: {DR*100:.4}% LossRate: {LR*100:.4}% ")
