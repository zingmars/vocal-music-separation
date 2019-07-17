# A script to generate a binary mask's visual representation
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys

parser = argparse.ArgumentParser(description="Create a visualisation for NN output")
parser.add_argument("file", type=str, help="Path to the outputs (saved numpy array, i.e. labels.out)")
parser.add_argument("type", type=str, default="binary", nargs='?', help="Type of graph to output (normal/binary)")
parser.add_argument("output", default="", nargs='?', type=str, help="Path to output file (png)")
args = parser.parse_args()

data = np.loadtxt(args.file)

fig = plt.figure()
plt.title('Binary mask')

if args.type == "normal":
    # import matplotlib as mpl
    # mpl.use('Qt5Agg') #Requires pyqt5

    x_range_start = 0
    x_range_end = 50
    y_range_start = 845
    y_range_end = 865

    newdata = data[x_range_start:x_range_end, y_range_start:y_range_end]
    plt.imshow(newdata, cmap="Greys", origin='lower')

    ax = plt.gca()
    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency')
    for i in range(0, np.shape(newdata)[0]):
        for j in range(0, np.shape(newdata)[1]):
            s = '%.2f' % newdata[i][j]
            ax.text(j, i, s, fontsize=5, ha='center', va='center')
elif args.type == "binary":
    processed = np.empty(np.shape(data))
    processed[data>0.45] = [1]
    processed[data<0.45] = [0]
    plt.imshow(processed, interpolation='nearest', cmap="Greys", origin='lower')
else:
    print("Invalid action - ", args.type)
    sys.exit(2)

if args.output is not "":
    plt.savefig(args.output, dpi=1000)
    print("Saved to ", args.output)
else:
    #plt.rcParams['figure.figsize'] = (1000, 1000)
    plt.tight_layout()
    plt.show()
