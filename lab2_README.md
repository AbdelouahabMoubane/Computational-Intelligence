The initial logic of the policy based on a set of Floating Numbers was developed in collaboration with Edoardo Vay 316737.


My evolution strategy is based on NimSum, which is the xor between the binary quantities of the objects for each row. The idea is to evolve the population, where each individual is a vector of weights associated with all possible NimSums, given a given number of rows.

ES is a strategy function, where, given the vector of weights, the player stochastically chooses a NimSum based on the weight and plays to obtain it. ES1 is similar to ES, but the player deterministically chooses the NimSum with the maximum weight among the feasible actions.

offspring, is the function that, given an individual, creates the offspring, that is, the new population

Using the evolution strategy, I made the individuals evolve by making them play with the optimal strategy in the first approach, while in the second at each iteration I made the offspring clash with the parent.