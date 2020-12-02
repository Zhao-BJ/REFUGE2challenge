import numpy as np
from skimage.feature import blob_log
from skimage.color import rgb2gray


def get_peak_coordinates(image, threshold=0.2):
    image_gray = rgb2gray(image)
    image_gray = np.pad(image_gray, (15, 15), 'constant')
    blobs = blob_log(image_gray, min_sigma=10, max_sigma=50, threshold=threshold)
    bb = blobs[:, :2].astype('int')
    if blobs.shape[0] < 2:
        new_blobs = np.copy(blobs)
        while new_blobs.shape[0] < 2:
            threshold = 0.8 * threshold
            if threshold < 0.001:
                print('Threshold too low! Passing...')
                break
            else:
                new_blobs = blob_log(image, min_sigma=10, max_sigma=50, threshold=threshold)
        blobs = new_blobs
        if blobs.shape[0] < 2:
            np.concatenate((blobs, [[256, 256, 0]]), axis=0)
    blobs = blobs - 15 # to account for to the initial padding
    blobs[np.where(blobs > 512)] = 0
    blobs[np.where(blobs < 0)] = 0
    blobs = blobs[:, :2].astype('int')
    bb2 = blobs[:, :2].astype('int')
    #if blobs.shape[0] > 2:
    #    sorted_indx = np.argsort(image[bb2[:, 0], bb2[:, 1]], axis=None)[::-1]
    #    print sorted_indx
    #    blobs = bb2[sorted_indx[:2]]
    return blobs


def determine_fovea(image, coords, neigh=3):
    """ Determines which peak corresponds to the OD and to the Fovea.
    input params:
        image: the RGB image
        coords: the coordinates of the two selected peak_coords
        neigh: the neighbourhood to consider for evaluation
    returns:
        od_coords: the coordinates of the peak selected as OD
        fov_coords: the coordinates of the peak selected as Fovea
    """
    # create a special case for the border, in case the peak is located close
    # to it, it must always have neighbours
    coords[np.where(coords < neigh)] = neigh
    coords[np.where(coords > (639-neigh))] = (639-neigh)

    coord_new1 = coords[0]

    # Calculate the mean intensity of each peak and its neighbohood
    # i1 = np.mean(image[:,:,1][coord_new1[0]-neigh:coord_new1[0]+neigh,
    #                             coord_new1[1]-neigh:coord_new1[1]+neigh])
    # i2 = np.mean(image[:,:,1][coord_new2[0]-neigh:coord_new2[0]+neigh,
    #                             coord_new2[1]-neigh:coord_new2[1]+neigh])
    #
    # # The OD is expected to have higher intensity
    # if i1 >= i2:
    #     od_coords = coord_new1
    #     fov_coords = coord_new2
    #
    # elif i1<i2:
    #     od_coords = coord_new2
    #     fov_coords = coord_new1
    # else:
    #     od_coords = (256,256)
    #     fov_coords = (256,256)
    fov_coords = coord_new1
    return fov_coords


def determine_od(image, coords, neigh=3):
    """ Determines which peak corresponds to the OD and to the Fovea.
    input params:
        image: the RGB image
        coords: the coordinates of the two selected peak_coords
        neigh: the neighbourhood to consider for evaluation
    returns:
        od_coords: the coordinates of the peak selected as OD
        fov_coords: the coordinates of the peak selected as Fovea
    """
    # create a special case for the border, in case the peak is located close to it, it must always have neighbours
    coords[np.where(coords < neigh)] = neigh
    coords[np.where(coords > (511-neigh))] = (511-neigh)

    coord_new1, coord_new2 = coords[0], coords[1]

    # Calculate the mean intensity of each peak and its neighbohood
    i1 = np.mean(image[:, :, 1][coord_new1[0]-neigh:coord_new1[0]+neigh, coord_new1[1]-neigh:coord_new1[1]+neigh])
    i2 = np.mean(image[:, :, 1][coord_new2[0]-neigh:coord_new2[0]+neigh, coord_new2[1]-neigh:coord_new2[1]+neigh])

    # The OD is expected to have higher intensity
    if i1 >= i2:
        od_coords = coord_new1
        fov_coords = coord_new2
    elif i1 < i2:
        od_coords = coord_new2
        fov_coords = coord_new1
    else:
        od_coords = (256, 256)
        fov_coords = (256, 256)
    return od_coords, fov_coords
