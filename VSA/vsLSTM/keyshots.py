import numpy as np
from cpd_auto import cpd_auto, estimate_vmax
from knapsack_iter import knapSack


def gen_data(n, m, d=1):
    """Generates data with change points
    n - number of samples
    m - number of change-points
    WARN: sigma is proportional to m
    Returns:
        X - data array (n X d)
        cps - change-points array, including 0 and n"""
    np.random.seed(1)
    # Select changes at some distance from the boundaries
    cps = np.random.permutation((n * 3 / 4) - 1)[0:m] + 1 + n / 8
    cps = np.sort(cps)
    cps = [0] + list(cps) + [n]
    mus = np.random.rand(m + 1, d) * (m / 2)  # make sigma = m/2
    X = np.zeros((n, d))
    for k in range(m + 1):
        X[cps[k]:cps[k + 1], :] = mus[k, :][np.newaxis, :] + np.random.rand(cps[k + 1] - cps[k], d)
    return (X, np.array(cps))


# def to_keyshots(values):
#     max_weight = int(0.15 * len(values))
#     # print ('{}==='.format(len(values) - 1))
#
#     K = np.dot(values, values.T)
#     vmax = estimate_vmax(K)
#     cps, scores = cpd_auto(K, len(values) - 1, vmax, lmin=1, lmax=max_weight)
#     cps = np.append([0], np.append(cps, [len(values)], 0), 0)
#     lists = [(values[cps[idx]:cps[idx + 1]], cps[idx]) for idx in range(len(cps) - 1)]
#     segments = [tuple([np.average(i[0]), len(i[0]), i[1]]) for i in lists]
#     # print segments
#     chosen = knapsack_.knapsack(segments, max_weight)[1]
#     keyshots = np.zeros(len(values))
#     for i in chosen:
#         keyshots[i[2]:i[1] + i[2]] = 1
#     return keyshots


# def to_keyshots_feature(x,y):
#     max_weight = int(0.15 * len(x))
#     # print ('{}==='.format(len(values) - 1))
#
#     K = np.dot(x, x.T)
#     vmax = estimate_vmax(K)
#     cps, scores = cpd_auto(K, len(x) - 1, vmax, lmin=1, lmax=max_weight)
#     cps = np.append([0], np.append(cps, [len(x)], 0), 0)
#     lists = [(y[cps[idx]:cps[idx + 1]], cps[idx]) for idx in range(len(cps) - 1)]
#     segments = [tuple([np.average(i[0]), len(i[0]), i[1]]) for i in lists]
#     # print segments
#     chosen = knapsack_.knapsack(segments, max_weight)[1]
#     # value, weight, start = zip(*segments)
#     # print value, weight, start
#     # chosen = knapsack.knapsack(value, weight).solve(max_weight)
#     # print chosen
#     keyshots = np.zeros(len(y))
#     # for i in chosen[1]:
#     #     keyshots[start[i]:start[i] + weight[i]] = 1
#     for i in chosen:
#         keyshots[i[2]:i[1] + i[2]] = 1
#     return keyshots

def to_keyshots_feature(x,y):
    max_weight = int(0.18 * len(x))
    # print ('{}==='.format(len(values) - 1))

    K = np.dot(x, x.T)
    vmax = estimate_vmax(K)
    cps, scores = cpd_auto(K, len(x) - 1, vmax, lmin=1, lmax=max_weight)
    cps = np.append([0], np.append(cps, [len(x)], 0), 0)
    lists = [(y[cps[idx]:cps[idx + 1]], cps[idx]) for idx in range(len(cps) - 1)]
    segments = [tuple([np.average(i[0]), len(i[0]), i[1]]) for i in lists]
    # print segments

    value, weight, start = zip(*segments)
    # print value, weight, start
    chosen = knapSack(max_weight,weight,value,len(weight))
    # chosen_2 = knapsack_.knapsack(segments, max_weight)

    # dd = [start[j] for m, j in enumerate(chosen[1])]
    # print np.setdiff1d((np.unique(dd)), (zip(*chosen_2[1])[2]))
    # print len(segments)
    # print (np.unique(dd))
    # print (zip(*chosen_2[1])[2])
    # print chosen
    # keyshots = np.zeros(len(y))
    # for i in chosen[1]:
    #     keyshots[start[int(i)]:start[int(i)] + weight[int(i)]] = 1

    # keyshots2 = np.zeros(len(y))
    #
    # for i in chosen_2[1]:
    #     keyshots2[i[2]:i[1] + i[2]] = 1
    #
    # print ('{} => {}'.format(np.sum(keyshots2),np.sum(int(j) for i,j in enumerate(chosen))))


    keyshots2 = np.zeros(len(y))
    chosen = [int(j) for i,j in enumerate(chosen)]

    for i,j in enumerate(chosen):
        if(j==1):
            keyshots2[start[int(i)]:start[int(i)] + weight[int(i)]] = 1


    return keyshots2
    # return [int(j) for i,j in enumerate(chosen)]


# if __name__ == "__main__":
#     n = 1000
#     m = 10
#     (X, cps_gt) = gen_data(n, m)
#     # print "Ground truth:", cps_gt
#     print to_keyshots(X).shape
