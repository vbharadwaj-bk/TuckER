import io
import sys
import numpy as np

import matplotlib.pyplot as plt


def plotFile(filename, plot=False):
    f = open(filename, "r")
    lines = f.readlines()
    lines = [str.rstrip(line) for line in lines]

    trainLossX = []
    trainLossY = []

    testHits10X = []
    testHits10Y = []

    endLoop = False
    for i in range(500):
        try:
            start = lines.index("{}".format(i+1))
            trainLossX.append(i+1)
            trainLossY.append(np.log(float(lines[start+2])))
            end = lines.index("{}".format(i+2))
            try:
                testIndex = lines.index("Test:", start)
                if testIndex <= end:
                    testHits10X.append(i+1)
                    testHits10Y.append(float(lines[testIndex+2].split(' ')[2]))
            except ValueError:
                pass
        except ValueError:
            try:
                testIndex = lines.index("Test:", start)
                testHits10X.append(i+1)
                testHits10Y.append(float(lines[testIndex+2].split(' ')[2]))
            except ValueError:
                pass
            break

    if plot:
        fig, ax1 = plt.subplots()
        color = 'tab:red'
        ax1.set_xlabel('Epoch #')
        ax1.set_ylabel('Log Training Loss (Binary Cross Entropy)', color=color)

        # Exclude first few datapoints to avoid throwing off the scale
        ax1.plot(trainLossX[3:], trainLossY[3:], color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('Hits@10 on Test Set', color=color)
        ax2.plot(testHits10X, testHits10Y, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()

    f.close()
    return testHits10X, testHits10Y


if __name__ == '__main__':
    data = [plotFile(fname) for fname in sys.argv[1:]]
    fig, ax = plt.subplots()
    ax.plot(data[0][0], data[0][1][:], label="Baseline TuckER (SVD)")

    ax.plot(data[1][0][:], data[1][1][:], label="Embedding TT-Rank 8")
    ax.plot(data[2][0][:], data[2][1][:], label="Embedding TT-Rank 64")
    ax.plot(data[3][0][:], data[3][1][:], label="Embedding TT-Rank 128")

    ax.set_xlabel("Epoch #")
    ax.set_ylabel("Hits@10 Test Performance")
    ax.legend()
    plt.show()

