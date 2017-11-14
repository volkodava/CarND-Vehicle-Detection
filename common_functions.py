import io

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
from skimage.feature import hog


def validate_images_shape(fnames, expected_shape):
    result = []
    for fname in fnames:
        _, image = read_image(fname)
        assert expected_shape == image.shape
        result.append(image)

    return result


def read_image(fname):
    if fname is None:
        return fname, None

    result_image = mpimg.imread(fname)
    if fname.endswith(".png"):
        # data from .png images scaled 0 to 1 by mpimg
        result_image *= 255

    return fname, np.uint8(result_image)


def read_images(fnames):
    assert isinstance(fnames, (list, tuple, np.ndarray)), "Files must be list/tuple/ndarray"

    result = [read_image(fname) for fname in fnames]
    return zip(*result)


def show_images(images, labels, cols, figsize=(16, 8), title=None):
    assert len(images) == len(labels)

    rows = (len(images) / cols) + 1

    plt.figure(figsize=figsize)

    for idx, image in enumerate(images):
        plt.subplot(rows, cols, idx + 1)
        image = image.squeeze()
        if len(image.shape) == 2:
            plt.imshow(image, cmap="gray")
        else:
            plt.imshow(image)
        plt.title(labels[idx])
        plt.axis("off")

    if title is not None:
        plt.suptitle(title, fontsize=16)

    plt.tight_layout(pad=3.0)
    plt.show()


def plot_to_image():
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    buf.close()

    # close plot before return to stop from adding more information from outer scope
    plt.close()

    return np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 3)


# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, block_norm="L2-Hys",
                     transform_sqrt=True, vis=False, feature_vec=True):
    if vis:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  block_norm=block_norm, transform_sqrt=transform_sqrt,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       block_norm=block_norm, transform_sqrt=transform_sqrt,
                       visualise=vis, feature_vector=feature_vec)

        return features


# Define a function to compute binned color features
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features


# Define a function to compute color histogram features
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, color_space="RGB", spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256), orient=9,
                        pix_per_cell=8, cell_per_block=2,
                        hog_channel="0", block_norm="L2-Hys",
                        transform_sqrt=True, vis=False, feature_vec=True,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # 1) Define an empty list to receive features
    img_features = []
    # 2) Apply color conversion if other than "RGB"
    feature_image = convert_color(img, color_space)
    # 3) Compute spatial features if flag is set
    if spatial_feat:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # 4) Append features to list
        img_features.append(spatial_features)
    # 5) Compute histogram features if flag is set
    if hist_feat:
        hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
        # 6) Append features to list
        img_features.append(hist_features)
    # 7) Compute HOG features if flag is set
    if hog_feat:
        if hog_channel == "ALL":
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:, :, channel],
                                                     orient, pix_per_cell, cell_per_block, block_norm=block_norm,
                                                     transform_sqrt=transform_sqrt, vis=vis, feature_vec=feature_vec))
        else:
            hog_features = get_hog_features(feature_image[:, :, int(hog_channel)], orient,
                                            pix_per_cell, cell_per_block, block_norm=block_norm,
                                            transform_sqrt=transform_sqrt, vis=vis, feature_vec=feature_vec)
        # 8) Append features to list
        img_features.append(hog_features)

    # 9) Return concatenated array of features
    return np.concatenate(img_features)


def convert_color(img, color_space):
    feature_image = None
    if color_space != "RGB":
        if color_space == "HSV":
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == "LUV":
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == "HLS":
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == "YUV":
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == "YCrCb":
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        elif color_space == "LAB":
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    else:
        feature_image = np.copy(img)

    return feature_image


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space="RGB", spatial_size=(32, 32),
                     hist_bins=32, hist_range=(0, 256), orient=9,
                     pix_per_cell=8, cell_per_block=2,
                     hog_channel="0", block_norm="L2-Hys",
                     transform_sqrt=True, vis=False, feature_vec=True,
                     spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for image in imgs:
        img_features = single_img_features(image, color_space=color_space, spatial_size=spatial_size,
                                           hist_bins=hist_bins, hist_range=hist_range, orient=orient,
                                           pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                           hog_channel=hog_channel, block_norm=block_norm,
                                           transform_sqrt=transform_sqrt, vis=vis, feature_vec=feature_vec,
                                           spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)

        features.append(img_features)
    # Return list of feature vectors
    return features


# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_windows(image, windows, clf, scaler, sample_height, sample_width,
                   color_space="RGB", spatial_size=(32, 32), hist_bins=32,
                   hist_range=(0, 256), orient=9,
                   pix_per_cell=8, cell_per_block=2,
                   hog_channel="0", block_norm="L2-Hys",
                   transform_sqrt=True, vis=False, feature_vec=True,
                   spatial_feat=True, hist_feat=True, hog_feat=True):
    # 1) Create an empty list to receive positive detection windows
    on_windows = []
    # 2) Iterate over all windows in the list
    for window in windows:
        # 3) Extract the test window from original image
        resized_image = cv2.resize(image[window[0][1]:window[1][1], window[0][0]:window[1][0]],
                                   (sample_width, sample_height))
        # 4) Extract features for that window using single_img_features()
        features = single_img_features(resized_image, color_space=color_space,
                                       spatial_size=spatial_size,
                                       hist_bins=hist_bins, hist_range=hist_range,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, block_norm=block_norm,
                                       transform_sqrt=transform_sqrt, vis=vis, feature_vec=feature_vec,
                                       spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
        # 5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # 6) Predict using your classifier
        prediction = clf.predict(test_features)
        # 7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    # 8) Return windows for positive detections
    return on_windows


# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] is None:
        x_start_stop[0] = 0
    if x_start_stop[1] is None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] is None:
        y_start_stop[0] = 0
    if y_start_stop[1] is None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0] * (xy_overlap[0]))
    ny_buffer = np.int(xy_window[1] * (xy_overlap[1]))
    nx_windows = np.int((xspan - nx_buffer) / nx_pix_per_step)
    ny_windows = np.int((yspan - ny_buffer) / ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you"ll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]

            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=5, color_configs=None):
    # Make a copy of the image
    imcopy = np.copy(img)

    apply_color_config = False
    if color_configs is not None:
        apply_color_config = True

    # Iterate through the bounding boxes
    for bbox in bboxes:
        use_color = color
        if apply_color_config:
            # window_list.append(((startx, starty), (endx, endy)))
            x_diff = abs(bbox[0][0] - bbox[1][0])
            y_diff = abs(bbox[0][1] - bbox[1][1])

            assert x_diff == y_diff, "X:%s == Y:%s" % (x_diff, y_diff)

            use_color = color_configs[x_diff]

        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], use_color, thick)
    # Return the image copy with boxes drawn
    return imcopy


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes


def apply_heat_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def group_bboxes(labels, window_size_threshold=(32, 32)):
    result_bboxes = []
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        bbox_width = bbox[1][0] - bbox[0][0]
        bbox_height = bbox[1][1] - bbox[0][1]

        # Draw the box on the image
        if bbox_width >= window_size_threshold[0] \
                and bbox_height >= window_size_threshold[1]:
            result_bboxes.append(bbox)
    # Return the result bboxes
    return result_bboxes


def plot3d(pixels, colors_rgb,
           axis_labels=list("RGB"), axis_limits=((0, 255), (0, 255), (0, 255))):
    """Plot pixels in 3D."""

    # Create figure and 3D axes
    fig = plt.figure(figsize=(8, 8))
    ax = Axes3D(fig)

    # Set axis limits
    ax.set_xlim(*axis_limits[0])
    ax.set_ylim(*axis_limits[1])
    ax.set_zlim(*axis_limits[2])

    # Set axis labels and sizes
    ax.tick_params(axis="both", which="major", labelsize=14, pad=8)
    ax.set_xlabel(axis_labels[0], fontsize=16, labelpad=16)
    ax.set_ylabel(axis_labels[1], fontsize=16, labelpad=16)
    ax.set_zlabel(axis_labels[2], fontsize=16, labelpad=16)

    # Plot pixel values with colors given in colors_rgb
    ax.scatter(
        pixels[:, :, 0].ravel(),
        pixels[:, :, 1].ravel(),
        pixels[:, :, 2].ravel(),
        c=colors_rgb.reshape((-1, 3)), edgecolors="none")

    # return Axes3D object for further manipulation
    return ax
