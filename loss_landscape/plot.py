"""Plot the contours and trajectory give the corresponding files"""

import argparse
import logging
import os

import numpy as np
from matplotlib import pyplot

# Import seaborn
import seaborn as sns

# Apply the default theme
sns.set_style("white")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-D", "--debug", action='store_true')
    parser.add_argument("--result_folder", "-r", required=True)
    parser.add_argument("--trajectory_file", required=False, default=None)
    parser.add_argument("--surface_file", required=False, default=None)
    parser.add_argument("--plot_prefix", required=True, help="prefix for the figure names")

    args = parser.parse_args()

    # set up logging
    os.makedirs(f"{args.result_folder}", exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger = logging.getLogger()
    if args.debug:
        logger.setLevel(logging.DEBUG)

    if args.surface_file:
        # create a contour plot
        data = np.load(f"{args.surface_file}")

        xcoords = data["xcoordinates"]
        ycoords = data["ycoordinates"]
        losses = data["losses"]
        acc = data["accuracies"]

        X, Y = np.meshgrid(xcoords, ycoords, indexing="ij")
        Z = losses
        fig = pyplot.figure()
        CS = pyplot.contour(X, Y, Z, cmap=sns.color_pallete("viridis", as_cmap=True), levels=np.linspace(np.min(Z), np.max(Z), 10))
        pyplot.clabel(CS, inline=1, fontsize=8)
        fig.savefig(
            f"{args.result_folder}/{args.plot_prefix}_surface_2d_contour", dpi=300,
            bbox_inches='tight'
        )
        pyplot.close()

    if args.trajectory_file:
        # create a 2D plot of trajectory
        data = np.load(f"{args.trajectory_file}")

        xcoords = data["xcoordinates"]
        ycoords = data["ycoordinates"]

        fig = pyplot.figure()
        pyplot.plot(xcoords, ycoords, linewidth=0.5, alpha=0.3)
        pyplot.scatter(xcoords, ycoords, marker='.', c=np.arange(len(xcoords)), cmap=sns.color_palette("mako", as_cmap=True))
        pyplot.colorbar()
        pyplot.tick_params('y', labelsize='x-large')
        pyplot.tick_params('x', labelsize='x-large')

        fig.savefig(
            f"{args.result_folder}/{args.plot_prefix}_trajectory_2d", dpi=300,
            bbox_inches='tight'
        )
        pyplot.close()

    if args.surface_file and args.trajectory_file:
        # create a contour plot
        data = np.load(f"{args.surface_file}")

        xcoords = data["xcoordinates"]
        ycoords = data["ycoordinates"]
        losses = data["losses"]
        acc = data["accuracies"]

        X, Y = np.meshgrid(xcoords, ycoords, indexing="ij")
        Z = losses
        fig = pyplot.figure()
        CS = pyplot.contour(X, Y, Z, cmap=sns.color_pallete("viridis", as_cmap=True), levels=np.linspace(np.min(Z), np.max(Z), 10))
        pyplot.clabel(CS, inline=1, fontsize=8)

        data = np.load(f"{args.trajectory_file}")

        xcoords = data["xcoordinates"]
        ycoords = data["ycoordinates"]
        pyplot.plot(xcoords, ycoords, linewidth=0.5, alpha=0.3)
        pyplot.colorbar()
        pyplot.scatter(xcoords, ycoords, marker='.', c=np.arange(len(xcoords)),cmap=sns.color_palette("rocket", as_cmap=True) )
        pyplot.tick_params('y', labelsize='x-large')
        pyplot.tick_params('x', labelsize='x-large')

        fig.savefig(
            f"{args.result_folder}/{args.plot_prefix}_trajectory+contour_2d", dpi=300,
            bbox_inches='tight'
        )
        pyplot.close()
