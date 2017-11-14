from skimage.viewer import CollectionViewer
from skimage.viewer.plugins import Plugin
from skimage.viewer.widgets import CheckBox, Slider

from experiments import *

orient = 16  # HOG orientations
pix_per_cell = 32  # HOG pixels per cell

# ROI polygon coefficients
top_y = 0.53
bottom_y = 0.9
heatmap_threshold = 2

xy_window_red = 130
xy_overlap_red = 0.75

xy_window_green = 120
xy_overlap_green = 0.75

xy_window_blue = 110
xy_overlap_blue = 0.75

xy_window_yellow = 100
xy_overlap_yellow = 0.75

xy_window_red_color = (255, 0, 0)
xy_window_green_color = (0, 255, 0)
xy_window_blue_color = (0, 0, 255)
xy_window_yellow_color = (255, 255, 0)


class ModelTestUi:
    def __init__(self, search_pattern):
        self.plugin = Plugin(image_filter=self.image_filter, dock="right")

        self.show_origin_checkbox = CheckBox("show_orig", value=False, alignment="left")
        self.use_first_window_config_checkbox = CheckBox("use_first_window_config", value=True, alignment="left")

        self.top_y_slider = Slider('top_y', 0, 1, value=top_y)
        self.bottom_y_slider = Slider('bottom_y', 0, 1, value=bottom_y)
        self.heatmap_threshold_slider = Slider('heatmap_threshold', 0, 10, value=heatmap_threshold, value_type='int')

        self.xy_window_red_slider = Slider('xy_window_red', 0, 256, value=xy_window_red, value_type='int')
        self.xy_overlap_red_slider = Slider('xy_overlap_red', 0, 1, value=xy_overlap_red)

        self.xy_window_green_slider = Slider('xy_window_green', 0, 256, value=xy_window_green, value_type='int')
        self.xy_overlap_green_slider = Slider('xy_overlap_green', 0, 1, value=xy_overlap_green)

        self.xy_window_blue_slider = Slider('xy_window_blue', 0, 256, value=xy_window_blue, value_type='int')
        self.xy_overlap_blue_slider = Slider('xy_overlap_blue', 0, 1, value=xy_overlap_blue)

        self.xy_window_yellow_slider = Slider('xy_window_yellow', 0, 256, value=xy_window_yellow, value_type='int')
        self.xy_overlap_yellow_slider = Slider('xy_overlap_yellow', 0, 1, value=xy_overlap_yellow)

        self.plugin += self.show_origin_checkbox
        self.plugin += self.use_first_window_config_checkbox

        self.plugin += self.top_y_slider
        self.plugin += self.bottom_y_slider
        self.plugin += self.heatmap_threshold_slider

        self.plugin += self.xy_window_red_slider
        self.plugin += self.xy_overlap_red_slider

        self.plugin += self.xy_window_green_slider
        self.plugin += self.xy_overlap_green_slider

        self.plugin += self.xy_window_blue_slider
        self.plugin += self.xy_overlap_blue_slider

        self.plugin += self.xy_window_yellow_slider
        self.plugin += self.xy_overlap_yellow_slider

        self.cars, self.notcars = load_cars_notcars(cars_path="../work/cars.pkl", notcars_path="../work/notcars.pkl")
        self.sample_height, self.sample_width, self.sample_depth = self.cars[0].shape

        linear_svc_path = "../work/models/linear_svc_orient_%s__pix_per_cell_%s.pkl" % (orient, pix_per_cell)
        standard_scaler_path = "../work/models/standard_scaler_orient_%s__pix_per_cell_%s.pkl" % (orient, pix_per_cell)

        self.svc, self.X_scaler = load_trained_model(linear_svc_path, standard_scaler_path)

        fnames = [path for path in glob.iglob(search_pattern, recursive=True)]
        _, self.images = read_images(fnames)
        self.viewer = CollectionViewer(self.images)

        self.viewer.connect_event("key_press_event", self.on_press)
        self.viewer += self.plugin
        print("Done")

    def image_filter(self, image, *args, **kwargs):
        image = np.copy(image)

        show_orig = kwargs["show_orig"]
        use_first_window_config = kwargs["use_first_window_config"]

        top_y = kwargs["top_y"]
        bottom_y = kwargs["bottom_y"]
        heatmap_threshold = kwargs["heatmap_threshold"]

        xy_window_red = kwargs["xy_window_red"]
        xy_overlap_red = kwargs["xy_overlap_red"]

        xy_window_green = kwargs["xy_window_green"]
        xy_overlap_green = kwargs["xy_overlap_green"]

        xy_window_blue = kwargs["xy_window_blue"]
        xy_overlap_blue = kwargs["xy_overlap_blue"]

        xy_window_yellow = kwargs["xy_window_yellow"]
        xy_overlap_yellow = kwargs["xy_overlap_yellow"]

        if show_orig:
            return image

        height, width = image.shape[:2]
        y_start = int(height * top_y)
        y_stop = int(height * bottom_y)

        slide_windows_red = slide_window(image, x_start_stop=[None, None], y_start_stop=[y_start, y_stop],
                                         xy_window=(xy_window_red, xy_window_red),
                                         xy_overlap=(xy_overlap_red, xy_overlap_red))

        all_slide_windows = slide_windows_red

        if not use_first_window_config:
            slide_windows_green = slide_window(image, x_start_stop=[None, None], y_start_stop=[y_start, y_stop],
                                               xy_window=(xy_window_green, xy_window_green),
                                               xy_overlap=(xy_overlap_green, xy_overlap_green))

            slide_windows_blue = slide_window(image, x_start_stop=[None, None], y_start_stop=[y_start, y_stop],
                                              xy_window=(xy_window_blue, xy_window_blue),
                                              xy_overlap=(xy_overlap_blue, xy_overlap_blue))

            slide_windows_yellow = slide_window(image, x_start_stop=[None, None], y_start_stop=[y_start, y_stop],
                                                xy_window=(xy_window_yellow, xy_window_yellow),
                                                xy_overlap=(xy_overlap_yellow, xy_overlap_yellow))

            all_slide_windows = all_slide_windows + slide_windows_green
            all_slide_windows = all_slide_windows + slide_windows_blue
            all_slide_windows = all_slide_windows + slide_windows_yellow

        hot_windows = search_windows(image, all_slide_windows, self.svc, self.X_scaler,
                                     self.sample_height, self.sample_width, color_space,
                                     spatial_size, hist_bins, hist_range, orient,
                                     pix_per_cell, cell_per_block, hog_channel,
                                     block_norm, transform_sqrt, vis, feature_vec, spatial_feat, hist_feat, hog_feat)

        image_boxes = draw_boxes(image, all_slide_windows)
        window_img = draw_boxes(image, hot_windows, color_configs={
            xy_window_red: xy_window_red_color,
            xy_window_green: xy_window_green_color,
            xy_window_blue: xy_window_blue_color,
            xy_window_yellow: xy_window_yellow_color
        })

        heatmap_img, grouped_bboxes = group_windows(image, hot_windows,
                                                    heatmap_threshold)
        window_grouped_img = draw_boxes(image, grouped_bboxes)

        # draw red start/stop position on image
        cv2.line(image, (0, y_start), (width, y_start), (255, 0, 0), 2)
        cv2.line(image, (0, y_stop), (width, y_stop), (255, 0, 0), 2)

        result = combine_images_horiz(image, image_boxes)
        result = combine_images_horiz(result, window_img)
        result = combine_images_horiz(result, heatmap_img)
        result = combine_images_horiz(result, window_grouped_img)

        return result

    def on_press(self, event):
        if event.key == "ctrl+r":
            self.on_reset()
        elif event.key == "ctrl+p":
            self.on_print()

    def on_print(self, args=None):
        print("""
top_y = {}
bottom_y = {}
heatmap_threshold = {}

xy_window_red = {}
xy_overlap_red = {}

xy_window_green = {}
xy_overlap_green = {}

xy_window_blue = {}
xy_overlap_blue = {}

xy_window_yellow = {}
xy_overlap_yellow = {}
""".format(round(self.top_y_slider.val, 3), round(self.bottom_y_slider.val, 3), self.heatmap_threshold_slider.val,
           self.xy_window_red_slider.val, round(self.xy_overlap_red_slider.val, 3),
           self.xy_window_green_slider.val, round(self.xy_overlap_green_slider.val, 3),
           self.xy_window_blue_slider.val, round(self.xy_overlap_blue_slider.val, 3),
           self.xy_window_yellow_slider.val, round(self.xy_overlap_yellow_slider.val, 3),
           ))

    def on_reset(self, args=None):
        print("Reset")

        self.update_val(self.top_y_slider, top_y)
        self.update_val(self.bottom_y_slider, bottom_y)
        self.update_val(self.heatmap_threshold_slider, heatmap_threshold)
        self.update_val(self.xy_window_red_slider, xy_window_red)
        self.update_val(self.xy_overlap_red_slider, xy_overlap_red)
        self.update_val(self.xy_window_green_slider, xy_window_green)
        self.update_val(self.xy_overlap_green_slider, xy_overlap_green)
        self.update_val(self.xy_window_blue_slider, xy_window_blue)
        self.update_val(self.xy_overlap_blue_slider, xy_overlap_blue)
        self.update_val(self.xy_window_yellow_slider, xy_window_yellow)
        self.update_val(self.xy_overlap_yellow_slider, xy_overlap_yellow)

        self.plugin.filter_image()

    def show(self):
        self.viewer.show()

    def update_val(self, comp, newval):
        comp.val = newval
        comp.editbox.setText("%s" % newval)

        return newval


if __name__ == "__main__":
    ModelTestUi("../test_images/test*.jpg").show()
