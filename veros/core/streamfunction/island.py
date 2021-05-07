import scipy.ndimage

from veros import veros_routine, logger
from veros.core import utilities
from veros.core.operators import numpy as np

# fall back to vanilla NumPy for some operations
import numpy as onp


def _compute_isleperim(kmt, enable_cyclic_x):
    kmt = utilities.enforce_boundaries(kmt, enable_cyclic_x)
    kmt = onp.asarray(kmt)

    structure = onp.ones((3, 3))  # merge diagonally connected land masses

    # find all land masses
    labelled, _ = scipy.ndimage.label(kmt == 0, structure=structure)

    # find and set perimeter
    land_masses = labelled > 0
    inner = scipy.ndimage.binary_dilation(land_masses, structure=structure)

    perimeter = onp.logical_xor(inner, land_masses)
    labelled[perimeter] = -1

    # match wrapping periodic land masses
    if enable_cyclic_x:
        west_slice = onp.array(labelled[2])
        east_slice = onp.array(labelled[-2])

        for west_label in onp.unique(west_slice[west_slice > 0]):
            east_labels = onp.unique(east_slice[west_slice == west_label])
            east_labels = east_labels[~onp.isin(east_labels, [west_label, -1])]
            if not east_labels.size:
                # already labelled correctly
                continue
            assert len(onp.unique(east_labels)) == 1, (west_label, east_labels)
            labelled[labelled == east_labels[0]] = west_label

    labelled = utilities.enforce_boundaries(labelled, enable_cyclic_x)
    labelled = onp.asarray(labelled)

    # label landmasses in a way that is consistent with pyom
    labels = onp.unique(labelled[labelled > 0])

    label_idx = {}
    for label in labels:
        # find index of first island cell, scanning west to east, north to south
        label_idx[label] = onp.argmax(labelled[:, ::-1].T == label)

    sorted_labels = list(sorted(labels, key=lambda i: label_idx[i]))

    # ensure labels are numbered consecutively
    relabelled = onp.array(labelled)
    for new_label, label in enumerate(sorted_labels, 1):
        if label == new_label:
            continue
        relabelled[labelled == label] = new_label

    return np.asarray(relabelled)


@veros_routine(dist_safe=False, local_variables=("kbot", "land_map"))
def isleperim(state):
    vs = state.variables
    settings = state.settings

    logger.debug(' Determining number of land masses')
    vs.land_map = _compute_isleperim(vs.kbot, settings.enable_cyclic_x)
    logger.debug(_ascii_map(vs.land_map))


def _ascii_map(boundary_map):
    def _get_char(c):
        if c == 0:
            return '.'
        if c < 0:
            return '#'
        return str(c % 10)

    boundary_map = onp.array(boundary_map)
    nx, ny = boundary_map.shape

    map_string = ''
    linewidth = 100
    iremain = nx
    istart = 0
    map_string += '\n'
    map_string += ' ' * (5 + min(linewidth, nx) // 2 - 13) + 'Land mass and perimeter'
    map_string += '\n'
    for _ in range(1, nx // linewidth + 2):
        iline = min(iremain, linewidth)
        iremain = iremain - iline
        if iline > 0:
            map_string += '\n'
            map_string += ''.join(['{:5d}'.format(istart + i + 1 - 2) for i in range(1, iline + 1, 5)])
            map_string += '\n'
            for j in range(ny - 1, -1, -1):
                map_string += '{:3d} '.format(j)
                map_string += ''.join([_get_char(boundary_map[istart + i - 2, j]) for i in range(2, iline + 2)])
                map_string += '\n'
            map_string += ''.join(['{:5d}'.format(istart + i + 1 - 2) for i in range(1, iline + 1, 5)])
            map_string += '\n'
            istart = istart + iline
    map_string += '\n'

    return map_string
