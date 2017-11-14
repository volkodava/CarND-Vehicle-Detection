import glob
import os
import time
from collections import deque

import scipy.misc
from moviepy.video.io.VideoFileClip import VideoFileClip
from scipy.ndimage.measurements import label
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from common_functions import *

output_path = "output_images"

color_space = "HSV"  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb, LAB
orient = 32  # HOG orientations
pix_per_cell = 16  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = "ALL"  # Can be "0", "1", "2", or "ALL"
block_norm = "L2"  # Can be "L1", "L1-sqrt", "L2", "L2-Hys"
spatial_size = None  # Spatial binning dimensions
hist_bins = -1  # Number of histogram bins
hist_range = None  # Histogram range
spatial_feat = False  # Spatial features on or off
hist_feat = False  # Histogram features on or off
hog_feat = True  # HOG features on or off
transform_sqrt = True
vis = False
feature_vec = True
heatmap_threshold = 2
# retrain = False
retrain = False
xy_overlap = 0.75

# Video processing params
QUEUE_LENGTH = 5
# width, height
window_size_threshold = (32, 32)

# image region for slide windows
# http://htmlcolorcodes.com/color-names/
slide_window_config = [
    {
        "top_y": 0.53,
        "bottom_y": 0.9,
        "xy_window": 130,
        "color": (255, 0, 0)  # red
    }, {
        "top_y": 0.53,
        "bottom_y": 0.9,
        "xy_window": 120,
        "color": (0, 255, 0)  # green
    }, {
        "top_y": 0.53,
        "bottom_y": 0.9,
        "xy_window": 110,
        "color": (0, 0, 255)  # blue
    }, {
        "top_y": 0.53,
        "bottom_y": 0.9,
        "xy_window": 100,
        "color": (255, 20, 147)  # deep pink
    }, {
        "top_y": 0.53,
        "bottom_y": 0.9,
        "xy_window": 90,
        "color": (255, 165, 0)  # orange
    }, {
        "top_y": 0.53,
        "bottom_y": 0.9,
        "xy_window": 80,
        "color": (255, 255, 0)  # yellow
    }
]

# saved model
linear_svc_path = os.path.join("work", "linear_svc.pkl")
standard_scaler_path = os.path.join("work", "standard_scaler.pkl")

cars_path = os.path.join("work", "cars.pkl")
notcars_path = os.path.join("work", "notcars.pkl")


def cars_notcars_available(cars_path=cars_path, notcars_path=notcars_path):
    return os.path.exists(cars_path) \
           and os.path.exists(notcars_path)


def save_cars_notcars(cars, notcars,
                      cars_path=cars_path, notcars_path=notcars_path):
    joblib.dump(cars, cars_path)
    joblib.dump(notcars, notcars_path)


def load_cars_notcars(cars_path=cars_path, notcars_path=notcars_path):
    return joblib.load(cars_path), \
           joblib.load(notcars_path)


def trained_model_available(linear_svc_path=linear_svc_path,
                            standard_scaler_path=standard_scaler_path):
    return os.path.exists(linear_svc_path) \
           and os.path.exists(standard_scaler_path)


def save_trained_model(linear_svc, standard_scaler,
                       linear_svc_path=linear_svc_path,
                       standard_scaler_path=standard_scaler_path):
    joblib.dump(linear_svc, linear_svc_path)
    joblib.dump(standard_scaler, standard_scaler_path)


def load_trained_model(linear_svc_path=linear_svc_path,
                       standard_scaler_path=standard_scaler_path):
    return joblib.load(linear_svc_path), \
           joblib.load(standard_scaler_path)


def read_train_data(cars_search_pattern, notcars_search_pattern, sample_size=-1):
    cars = [path for path in glob.iglob(cars_search_pattern, recursive=True)]
    notcars = [path for path in glob.iglob(notcars_search_pattern, recursive=True)]

    cars = cars[0:sample_size]
    notcars = notcars[0:sample_size]

    _, car_sample = read_image(cars[0])
    height, width, depth = car_sample.shape

    cars_images = validate_images_shape(cars, (height, width, depth))
    notcars_images = validate_images_shape(notcars, (height, width, depth))

    return cars_images, notcars_images, (height, width, depth)


def rescale_to_0_1(image):
    if np.max(image) > 1:
        return np.float32(image) / 255

    return image


def train_classifier(cars, notcars,
                     color_space="RGB", spatial_size=(32, 32),
                     hist_bins=32, hist_range=(0, 256), orient=9,
                     pix_per_cell=8, cell_per_block=2,
                     hog_channel="0", block_norm="L2-Hys",
                     transform_sqrt=True, vis=False, feature_vec=True,
                     spatial_feat=True, hist_feat=True, hog_feat=True,
                     retrain=False, debug=True,
                     linear_svc_path=linear_svc_path,
                     standard_scaler_path=standard_scaler_path):
    if not retrain and trained_model_available(linear_svc_path, standard_scaler_path):
        print("Model loaded from backup")
        return (*load_trained_model(linear_svc_path, standard_scaler_path), -1, -1)

    car_features = extract_features(cars, color_space=color_space,
                                    spatial_size=spatial_size,
                                    hist_bins=hist_bins, hist_range=hist_range,
                                    orient=orient, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel=hog_channel, block_norm=block_norm,
                                    transform_sqrt=transform_sqrt, vis=vis, feature_vec=feature_vec,
                                    spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = extract_features(notcars, color_space=color_space,
                                       spatial_size=spatial_size,
                                       hist_bins=hist_bins, hist_range=hist_range,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, block_norm=block_norm,
                                       transform_sqrt=transform_sqrt, vis=vis, feature_vec=feature_vec,
                                       spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)

    X = np.vstack((car_features, notcar_features)).astype(np.float64)

    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)

    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=0)
    feature_vector_length = len(X_train[0])

    if debug:
        print("Using:", orient, "orientations", pix_per_cell,
              "pixels per cell and", cell_per_block, "cells per block")
        print("Feature vector length:", feature_vector_length)
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    # Check the score of the SVC
    score = round(svc.score(X_test, y_test), 4)
    if debug:
        print(round(t2 - t, 2), "Seconds to train SVC...")
        print("Test Accuracy of SVC = ", score)

    save_trained_model(svc, X_scaler, linear_svc_path, standard_scaler_path)

    return svc, X_scaler, score, feature_vector_length


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_car_windows(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
                     hist_bins, trg_color_space=cv2.COLOR_RGB2YCrCb):
    windows = []

    img_tosearch = image[ystart:ystop, :, :]
    ctrans_tosearch = cv2.cvtColor(img_tosearch, trg_color_space)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block ** 2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(
                np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)

                windows.append(((xbox_left, ytop_draw + ystart),
                                (xbox_left + win_draw, ytop_draw + win_draw + ystart)))

    return windows


def read_all_data(force=False, cars_search_pattern="work/vehicles/**/*.png",
                  notcars_search_pattern="work/non-vehicles/**/*.png",
                  cars_path=cars_path, notcars_path=notcars_path):
    if not force and cars_notcars_available(cars_path, notcars_path):
        cars, notcars = load_cars_notcars(cars_path, notcars_path)
        height, width, depth = cars[0].shape

        print("Cars/NotCars loaded from backup")
        return cars, notcars, (height, width, depth)

    cars, notcars, (height, width, depth) = read_train_data(cars_search_pattern=cars_search_pattern,
                                                            notcars_search_pattern=notcars_search_pattern)
    save_cars_notcars(cars, notcars, cars_path, notcars_path)
    return cars, notcars, (height, width, depth)


def group_windows(image, windows,
                  heatmap_threshold=heatmap_threshold,
                  window_size_threshold=window_size_threshold):
    heat = np.zeros_like(image[:, :, 0]).astype(np.float)
    # Add heat to each box in box list
    heat = add_heat(heat, windows)
    # Apply threshold to help remove false positives
    heat = apply_heat_threshold(heat, heatmap_threshold)
    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)
    # plot and convert to image heatmap
    plt.close()
    plt.imshow(heatmap, cmap="hot")
    plt.axis("off")
    plt.tight_layout()
    heatmap_img = plot_to_image()

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    grouped_bboxes = group_bboxes(labels, window_size_threshold)

    return heatmap_img, grouped_bboxes


def to_grayscale(image):
    return np.mean(image, axis=2)


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


def combine_images_horiz(a, b):
    assert len(a.shape) == 3, "Height, width, depth required"
    assert len(a.shape) == len(b.shape), "Shape of images must be equal"

    ha, wa, da = a.shape[:3]
    hb, wb, db = b.shape[:3]

    assert da == db, "Depth must be the same for both images"

    max_height = np.max([ha, hb])
    total_width = wa + wb
    new_img = np.zeros(shape=(max_height, total_width, da), dtype=np.uint8)

    new_img[:ha, :wa] = a
    new_img[:hb, wa:wa + wb] = b

    return new_img


def combine_images_vert(a, b):
    assert len(a.shape) == 3, "Height, width, depth required"
    assert len(a.shape) == len(b.shape), "Shape of images must be equal"

    ha, wa, da = a.shape[:3]
    hb, wb, db = b.shape[:3]

    assert da == db, "Depth must be the same for both images"

    total_height = ha + hb
    max_width = np.max([wa, wb])
    new_img = np.zeros(shape=(total_height, max_width, da), dtype=np.uint8)

    new_img[:ha, :wa] = a
    new_img[ha:ha + hb, :wb] = b

    return new_img


def combine_3_images(main, first, second):
    if main is None \
            or first is None \
            or second is None:
        return main

    height, width, depth = main.shape

    new_height = height // 2
    first_width = width // 2
    second_width = width - first_width

    result_image = np.zeros((height + new_height, width, depth), dtype=np.uint8)

    main_height_range = (0, height)
    main_width_range = (0, width)

    first_height_range = (height, height + new_height)
    first_width_range = (0, first_width)

    second_height_range = (height, height + new_height)
    second_width_range = (first_width, first_width + second_width)

    # main
    result_image[main_height_range[0]:main_height_range[1], main_width_range[0]:main_width_range[1], :] = main
    # first
    result_image[first_height_range[0]:first_height_range[1], first_width_range[0]:first_width_range[1], :] = \
        cv2.resize(first, (first_width, new_height))
    # second
    result_image[second_height_range[0]:second_height_range[1], second_width_range[0]:second_width_range[1], :] = \
        cv2.resize(second, (second_width, new_height))

    return result_image


def convert_hog(image, block_norm=block_norm, cell_per_block=cell_per_block, hog_channel=hog_channel, orient=orient,
                pix_per_cell=pix_per_cell, transform_sqrt=transform_sqrt):
    result_hog_image = None
    if hog_channel == "ALL":
        for channel in range(image.shape[2]):
            features, hog_image = get_hog_features(image[:, :, channel],
                                                   orient, pix_per_cell, cell_per_block, block_norm=block_norm,
                                                   transform_sqrt=transform_sqrt, vis=True,
                                                   feature_vec=False)
            if len(hog_image.shape) == 2:
                hog_image = np.expand_dims(hog_image, axis=2)
                hog_image = np.uint8(hog_image * 255)

            if result_hog_image is None:
                result_hog_image = hog_image
            else:
                result_hog_image = combine_images_horiz(result_hog_image, hog_image)
    else:
        features, result_hog_image = get_hog_features(image[:, :, int(hog_channel)], orient,
                                                      pix_per_cell, cell_per_block, block_norm=block_norm,
                                                      transform_sqrt=transform_sqrt, vis=True,
                                                      feature_vec=False)
    return result_hog_image.squeeze()


class LaneProcessor:
    def __init__(self, sample_height, sample_width, svc, X_scaler,
                 window_size_threshold, output_dir):
        self.sample_height = sample_height
        self.sample_width = sample_width

        self.svc = svc
        self.X_scaler = X_scaler
        self.count = 0

        self.color_configs = {}

        for config in slide_window_config:
            self.color_configs[config["xy_window"]] = config["color"]

        # shutil.rmtree(output_dir, ignore_errors=True)
        # os.makedirs(output_dir, exist_ok=True)

        self.window_size_threshold = window_size_threshold
        self.output_dir = output_dir

        self.grouped_windows = deque(maxlen=QUEUE_LENGTH)

    def aggregate(self, grouped_windows, values):
        if grouped_windows is not None:
            values.append(grouped_windows)

        if len(values) > 0:
            grouped_windows = []
            list(grouped_windows.extend(value) for value in values)

        return grouped_windows

    def process(self, image):
        height, width = image.shape[:2]

        all_slide_windows = []
        for config in slide_window_config:
            top_y = config["top_y"]
            bottom_y = config["bottom_y"]
            xy_window = config["xy_window"]

            y_start = int(height * top_y)
            y_stop = int(height * bottom_y)

            all_slide_windows = all_slide_windows + slide_window(image, x_start_stop=[None, None],
                                                                 y_start_stop=[y_start, y_stop],
                                                                 xy_window=(xy_window, xy_window),
                                                                 xy_overlap=(xy_overlap, xy_overlap))

        hot_windows = search_windows(image, all_slide_windows, self.svc, self.X_scaler,
                                     self.sample_height, self.sample_width, color_space,
                                     spatial_size, hist_bins, hist_range, orient,
                                     pix_per_cell, cell_per_block, hog_channel,
                                     block_norm, transform_sqrt, vis, feature_vec,
                                     spatial_feat, hist_feat, hog_feat)

        window_img = draw_boxes(image, hot_windows, color_configs=self.color_configs)

        heatmap_img, grouped_bboxes = group_windows(image, hot_windows,
                                                    heatmap_threshold,
                                                    self.window_size_threshold)

        aggregated_bboxes = self.aggregate(grouped_bboxes, self.grouped_windows)
        heatmap_img, grouped_bboxes = group_windows(image, aggregated_bboxes,
                                                    heatmap_threshold=0,
                                                    window_size_threshold=(0, 0))

        window_grouped_img = draw_boxes(image, grouped_bboxes)

        result_image = combine_3_images(window_grouped_img, window_img, heatmap_img)

        cv2.putText(result_image, "#%s" % self.count, (80, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), lineType=cv2.LINE_AA, thickness=2)

        # for debugging
        scipy.misc.toimage(image).save(os.path.join(self.output_dir, "%s_orig.png" % self.count))
        scipy.misc.toimage(result_image).save(os.path.join(self.output_dir, "%s_res.png" % self.count))

        self.count += 1

        return result_image


def tag_video(finput, foutput, sample_height, sample_width,
              linear_svc_path, standard_scaler_path,
              window_size_threshold=(32, 32),
              subclip_secs=None, output_dir="./work/debug_video"):
    detector = LaneProcessor(sample_height, sample_width,
                             linear_svc_path, standard_scaler_path,
                             window_size_threshold, output_dir)

    video_clip = VideoFileClip(finput)
    if subclip_secs is not None:
        video_clip = video_clip.subclip(*subclip_secs)

    out_clip = video_clip.fl_image(detector.process)
    out_clip.write_videofile(foutput, audio=False)


if __name__ == "__main__":
    test_images_fnames = [path for path in glob.iglob("test_images/*.jpg", recursive=True)]
    image_paths, images = read_images(test_images_fnames)

    images_to_show = images
    labels_to_show = image_paths
    # show_images(images_to_show, labels=labels_to_show, cols=len(images_to_show) // 2,
    #             title="Input")

    # tag_video("project_video.mp4", os.path.join(output_path, "out_project_video.mp4"), subclip_secs=(38, 42))

    # Read in cars and notcars
    cars, notcars, (sample_height, sample_width, sample_depth) = read_all_data()
    # cars = cars[:10]
    # notcars = notcars[:10]

    print("Train samples loaded.")

    # csvsql --query "select * from model_orient__pix_per_cell_params order by score desc, feature_vector_length" work/models/model_orient__pix_per_cell_params.csv | csvlook
    #
    # orient = [8, 12, 16, 24, 32]  # HOG orientations
    # pix_per_cell = [8, 12, 16, 24, 32]  # HOG pixels per cell
    #
    # models_dir = os.path.join("work", "models")
    # shutil.rmtree(models_dir, ignore_errors=True)
    # os.makedirs(models_dir, exist_ok=True)
    #
    # result_file = os.path.join(models_dir, "model_orient__pix_per_cell_params.csv")
    # open(result_file, "w").close()
    #
    # with open(result_file, "a") as f:
    #     writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
    #     writer.writerow(
    #         (
    #             "score", "orient", "pix_per_cell", "feature_vector_length",
    #             "linear_svc_file", "standard_scaler_file",
    #             "error"
    #         )
    #     )
    #
    #     for orient_param in orient:
    #         for pix_per_cell_param in pix_per_cell:
    #             name = "orient_%s__pix_per_cell_%s" % (orient_param, pix_per_cell_param)
    #
    #             linear_svc_file = "linear_svc_%s.pkl" % name
    #             standard_scaler_file = "standard_scaler_%s.pkl" % name
    #
    #             linear_svc_path = os.path.join(models_dir, linear_svc_file)
    #             standard_scaler_path = os.path.join(models_dir, standard_scaler_file)
    #
    #             score = -1
    #             feature_vector_length = -1
    #             error = ""
    #             try:
    #                 svc, X_scaler, score, feature_vector_length = train_classifier(rescale_to_0_1(cars),
    #                                                                                rescale_to_0_1(notcars),
    #                                                                                color_space=color_space,
    #                                                                                spatial_size=spatial_size,
    #                                                                                hist_bins=hist_bins,
    #                                                                                hist_range=hist_range,
    #                                                                                orient=orient_param,
    #                                                                                pix_per_cell=pix_per_cell_param,
    #                                                                                cell_per_block=cell_per_block,
    #                                                                                hog_channel=hog_channel,
    #                                                                                block_norm=block_norm,
    #                                                                                transform_sqrt=transform_sqrt,
    #                                                                                vis=vis,
    #                                                                                feature_vec=feature_vec,
    #                                                                                spatial_feat=spatial_feat,
    #                                                                                hist_feat=hist_feat,
    #                                                                                hog_feat=hog_feat,
    #                                                                                retrain=retrain,
    #                                                                                linear_svc_path=linear_svc_path,
    #                                                                                standard_scaler_path=standard_scaler_path)
    #             except Exception as exc:
    #                 error = str(exc)
    #
    #             writer.writerow(
    #                 (
    #                     score, orient_param, pix_per_cell_param, feature_vector_length,
    #                     linear_svc_file, standard_scaler_file,
    #                     error
    #                 )
    #             )
    #             f.flush()

    # tag_video("project_video.mp4", os.path.join(output_path, "out_project_video.mp4"), subclip_secs=(38, 42))
    # self.linear_svc_path = "../work/models/linear_svc_orient_%s__pix_per_cell_%s.pkl" % (orient, pix_per_cell)
    # self.standard_scaler_path = "../work/models/standard_scaler_orient_%s__pix_per_cell_%s.pkl" % (orient, pix_per_cell)

    # linear_svc_path = "work/models/linear_svc_orient_%s__pix_per_cell_%s.pkl" % (orient, pix_per_cell)
    # standard_scaler_path = "work/models/standard_scaler_orient_%s__pix_per_cell_%s.pkl" % (orient, pix_per_cell)
    #
    # svc, X_scaler = load_trained_model(linear_svc_path, standard_scaler_path)

    linear_svc_path = "output_images/linear_svc.pkl"
    standard_scaler_path = "output_images/standard_scaler.pkl"

    svc, X_scaler, score, feature_vector_length = train_classifier(rescale_to_0_1(cars),
                                                                   rescale_to_0_1(notcars),
                                                                   color_space=color_space,
                                                                   spatial_size=spatial_size,
                                                                   hist_bins=hist_bins,
                                                                   hist_range=hist_range,
                                                                   orient=orient,
                                                                   pix_per_cell=pix_per_cell,
                                                                   cell_per_block=cell_per_block,
                                                                   hog_channel=hog_channel,
                                                                   block_norm=block_norm,
                                                                   transform_sqrt=transform_sqrt,
                                                                   vis=vis,
                                                                   feature_vec=feature_vec,
                                                                   spatial_feat=spatial_feat,
                                                                   hist_feat=hist_feat,
                                                                   hog_feat=hog_feat,
                                                                   retrain=retrain,
                                                                   linear_svc_path=linear_svc_path,
                                                                   standard_scaler_path=standard_scaler_path)

    output_file = "out_project_video_tmp.mp4"

    tag_video("project_video.mp4", os.path.join(output_path, output_file), sample_height, sample_width,
              svc, X_scaler, window_size_threshold, output_dir="./output_images")

    # tag_video("project_video.mp4", os.path.join(output_path, output_file), sample_height, sample_width,
    #           svc, X_scaler, subclip_secs=(10, 11))
