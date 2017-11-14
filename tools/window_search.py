import shutil

import scipy.misc
from tqdm import tqdm

from experiments import *

# csvsql --query "select expected, actual, input_size, result_size, window_size, filepath from windows" work/windows.csv | csvlook

screenshots = [
    {
        "location": "../test_images/test1.jpg",
        "cars_num": 2,
        "coords": [((820, 420), (940, 480)), ((1070, 410), (1260, 510))]
    },
    {
        "location": "../test_images/test001.jpg",
        "cars_num": 1,
        "coords": [((1120, 390), (1270, 520))]
    },
    {
        "location": "../test_images/test2.jpg",
        "cars_num": 0,
        "coords": []
    },
    {
        "location": "../test_images/test002.jpg",
        "cars_num": 2,
        "coords": [((990, 400), (1270, 580)), ((870, 410), (950, 450))]
    },
    {
        "location": "../test_images/test3.jpg",
        "cars_num": 1,
        "coords": [((870, 420), (960, 460))]
    },
    {
        "location": "../test_images/test003.jpg",
        "cars_num": 2,
        "coords": [((770, 420), (860, 470)), ((1220, 420), (1279, 480))]
    },
    {
        "location": "../test_images/test4.jpg",
        "cars_num": 2,
        "coords": [((820, 420), (940, 480)), ((1050, 410), (1240, 490))]
    },
    {
        "location": "../test_images/test5.jpg",
        "cars_num": 2,
        "coords": [((810, 420), (940, 470)), ((1120, 410), (1279, 500))]
    },
    {
        "location": "../test_images/test6.jpg",
        "cars_num": 2,
        "coords": [((810, 420), (940, 480)), ((1020, 420), (1200, 480))]
    }
]

top_y = 0.53
bottom_y = 0.9
heatmap_threshold = 0

xy_window_min = 32
xy_window_max = 256

xy_overlap = 0.75

cars, notcars, (sample_height, sample_width, sample_depth) = read_all_data(cars_path="../work/cars.pkl",
                                                                           notcars_path="../work/notcars.pkl")

orient = 16  # HOG orientations
pix_per_cell = 32  # HOG pixels per cell

linear_svc_path = "../work/models/linear_svc_orient_%s__pix_per_cell_%s.pkl" % (orient, pix_per_cell)
standard_scaler_path = "../work/models/standard_scaler_orient_%s__pix_per_cell_%s.pkl" % (orient, pix_per_cell)

svc, X_scaler = load_trained_model(linear_svc_path, standard_scaler_path)

log_dir = "../work/window_log"
shutil.rmtree(log_dir, ignore_errors=True)
os.makedirs(log_dir, exist_ok=True)


def find_cars(image, xy_window=xy_window_min, xy_overlap=xy_overlap, debug=False):
    height, width = image.shape[:2]
    y_start = int(height * top_y)
    y_stop = int(height * bottom_y)

    slide_windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[y_start, y_stop],
                                 xy_window=(xy_window, xy_window),
                                 xy_overlap=(xy_overlap, xy_overlap))

    hot_windows = search_windows(image, slide_windows, svc, X_scaler,
                                 sample_height, sample_width, color_space,
                                 spatial_size, hist_bins, hist_range, orient,
                                 pix_per_cell, cell_per_block, hog_channel,
                                 block_norm, transform_sqrt, vis, feature_vec, spatial_feat, hist_feat, hog_feat)

    heat = np.zeros_like(image[:, :, 0]).astype(np.float)
    # Add heat to each box in box list
    heat = add_heat(heat, hot_windows)
    # Apply threshold to help remove false positives
    heat = apply_heat_threshold(heat, heatmap_threshold)
    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)
    # Find final boxes from heatmap using label function
    labels = label(heatmap)

    found_cars = []
    input_windows = []
    result_windows = []
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        found_cars.append(bbox)
        input_windows.append(xy_window)
        result_windows.append([abs(bbox[0][0] - bbox[1][0]), abs(bbox[0][1] - bbox[1][1])])

    if debug:
        image_boxes = draw_boxes(image, found_cars)
        plt.imshow(image_boxes)
        plt.show()

    return input_windows, result_windows, found_cars


if __name__ == "__main__":
    result_file = os.path.join(log_dir, "windows.csv")
    open(result_file, "w").close()

    with open(result_file, "a") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(
            (
                "expected", "actual",
                "input_size", "result_size",
                "window_size", "location",
                "input_coords", "result_coords",
                "filename",
                "error"
            )
        )

        inc = 3
        num_of_inc = int((xy_window_max - xy_window_min) / inc)

        num_of_experiments = num_of_inc * len(screenshots)

        with tqdm(total=num_of_experiments) as pbar:
            for screenshot in screenshots:
                xy_window = xy_window_min
                location = screenshot["location"]
                expected_car_num = screenshot["cars_num"]
                expected_coords = screenshot["coords"]

                original_filename = os.path.basename(location)
                # splitted_original_filename = original_filename.split(".")
                # basename = splitted_original_filename[:-1]
                # file_extension = splitted_original_filename[-1]

                input_size = []
                for bbox in expected_coords:
                    input_size.append([abs(bbox[0][0] - bbox[1][0]), abs(bbox[0][1] - bbox[1][1])])

                fname, image = read_image(location)

                for idx in range(num_of_inc):
                    actual_car_num = 0
                    result_size = ""
                    window_size = str([xy_window, xy_window])
                    result_coords = ""
                    input_coords = str(expected_coords)
                    error = ""
                    filename = "%s_%s" % (idx, original_filename)
                    filepath = os.path.join(log_dir, filename)
                    found_cars = None

                    try:
                        input_windows, result_windows, found_cars = find_cars(image, xy_window=xy_window)

                        actual_car_num = len(found_cars)
                        result_size = str(result_windows)
                        result_coords = str(found_cars)
                    except Exception as exc:
                        error = str(exc)

                    if actual_car_num > 0:
                        debug_image = draw_boxes(image, found_cars)
                        scipy.misc.toimage(debug_image).save(filepath)

                    if actual_car_num > 0 or error:
                        writer.writerow(
                            (
                                expected_car_num, actual_car_num,
                                input_size, result_size,
                                window_size, location,
                                input_coords, result_coords,
                                filename,
                                error
                            )
                        )
                        f.flush()

                    xy_window += inc

                    pbar.update(1)
