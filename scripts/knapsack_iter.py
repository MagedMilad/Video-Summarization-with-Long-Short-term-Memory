# A Dynamic Programming based Python Program for 0-1 Knapsack problem
# Returns the maximum value that can be put in a knapsack of capacity W
import numpy as np

def knapSack(W, wt, val, n):
    K = [[0 for x in range(W+1)] for x in range(n+1)]

    # Build table K[][] in bottom up manner
    for i in range(n+1):
        for w in range(W+1):
            if i==0 or w==0:
                K[i][w] = 0
            elif wt[i-1] <= w:
                K[i][w] = max(val[i-1] + K[i-1][w-wt[i-1]],  K[i-1][w])
            else:
                K[i][w] = K[i-1][w]


    best = K[n][W]

    amount = np.zeros(n);
    a = best;
    j = n;
    Y = W;

    # j = j + 1;
    #
    # amount(j) = 1;
    # Y = Y - weights(j);
    # j = j - 1;
    # a = A(j + 1, Y + 1);

    while a > 0:
       while K[j][Y] == a:
           j = j - 1;

       j = j + 1;
       amount[j-1] = 1;
       Y = Y - wt[j-1];
       j = j - 1;
       a = K[j][Y];

    return amount

# weights = [1 ,1 ,1, 1 ,2 ,2 ,3];
# values  = [1 ,1 ,2 ,3, 1, 3 ,5];
# best = 13
# print(knapSack(7, weights, values, 7))