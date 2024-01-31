## Collaborations
The code was written in collaboration with Edoardo Vay 316737, except evaluation functions of the states and the implementation of alpha-beta pruning.

## Approach

The goal of the work is to create an agent that learns to play Quixo. We tried to use a similar approach to TicTacToe, but Quixo is a more complicated game and has many configurations, so we have a huge number of nodes. Furthermore, game states can repeat during a game, so we cannot use a tree to represent possible sequences of configurations.
therefore we chose to use another approach, alpha-beta pruning, with different depths, and we used some techniques and ideas to improve the code at a computational level and reduce the problem's dimensionality.

To reduce the dimensionality, like in TicTacToe, we have excluded equivalent configurations except for rotations and symmetries, to do this we used the dihedral group D_n, is the group of symmetries of a regular polygon with $n$ sides, in particular we used the one for the square D_4, it is a topic made during the course Istituzioni di algebra e geometria.

The strategy underlying evaluations of the state is to control the border as much as possible, in order to limit the opponent's plays. The evaluation function gave a higher rating for positions that are the intersection of multiple possible future solutions, in this case the central position is the more important. All this, however also taking into account the adjacent pieces.

The agent perform very well against the random player (50 games he plays first and 50 second), I have the following results for the different depths:
-depth = 1, the win percentage is variable between 95% and 100%;
-depth > 1, the win percentage is 100%, for all the runs I've done.
