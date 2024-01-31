import random
from game import Game, Move, Player
from Quixo import alpha_beta_pruning1, alpha_beta_pruning2

class RandomPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return from_pos, move

# both players are very good with depth >= 2
    
class MyPlayer1(Player):
    def __init__(self, depth) -> None:
        super().__init__()
        self.depth = depth

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        eval, pos, slide = alpha_beta_pruning1(game.get_board().reshape((25)), self.depth, game.get_current_player())
        print(eval)
        return ((pos % 5, pos // 5), slide)  

class MyPlayer2(Player):
    def __init__(self, depth) -> None:
        super().__init__()
        self.depth = depth

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        eval, pos, slide = alpha_beta_pruning2(game.get_board().reshape((25)), self.depth, game.get_current_player())
        print(eval)
        return ((pos % 5, pos // 5), slide)  



if __name__ == '__main__':
    g = Game()
    #g.print()
    player0 = MyPlayer1(3)
    player1 = RandomPlayer()
    winner = g.play(player0, player1)
    #g.print()
    print(f"Winner: Player {winner}")
