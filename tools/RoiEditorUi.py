from skimage.viewer import CollectionViewer
from skimage.viewer.plugins import Plugin
from skimage.viewer.widgets import CheckBox, Slider

from experiments import *

# ROI polygon coefficients
# to find big cars
top_y = 0.53
bottom_y = 0.9
xy_window = 192
xy_overlap = 0.75


# to find small cars
# top_y = 0.53
# bottom_y = 0.9
# xy_window = 89
# xy_overlap = 0.5


class RoiEditorUi:
    def __init__(self, search_pattern):
        self.plugin = Plugin(image_filter=self.image_filter, dock="bottom")

        self.show_origin_checkbox = CheckBox("show_orig", value=False, alignment="left")

        self.top_y_slider = Slider('top_y', 0, 1, value=top_y)
        self.bottom_y_slider = Slider('bottom_y', 0, 1, value=bottom_y)
        self.xy_window_slider = Slider('xy_window', 0, 512, value=xy_window, value_type='int')
        self.xy_overlap_slider = Slider('xy_overlap', 0, 1, value=xy_overlap)

        self.plugin += self.show_origin_checkbox
        self.plugin += self.top_y_slider
        self.plugin += self.bottom_y_slider
        self.plugin += self.xy_window_slider
        self.plugin += self.xy_overlap_slider

        fnames = [path for path in glob.iglob(search_pattern, recursive=True)]
        self.fnames, self.images = read_images(fnames)
        self.viewer = CollectionViewer(self.images)

        self.viewer.connect_event("key_press_event", self.on_press)
        self.viewer += self.plugin
        print("Done")

    def image_filter(self, image, *args, **kwargs):
        image = np.copy(image)

        show_orig = kwargs["show_orig"]

        top_y = kwargs["top_y"]
        bottom_y = kwargs["bottom_y"]
        xy_window = kwargs["xy_window"]
        xy_overlap = kwargs["xy_overlap"]

        car_index = self.viewer.slider.val
        cv2.putText(image, self.fnames[car_index], (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)

        if show_orig:
            return image

        height, width = image.shape[:2]
        y_start = int(height * top_y)
        y_stop = int(height * bottom_y)

        slide_windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[y_start, y_stop],
                                     xy_window=(xy_window, xy_window),
                                     xy_overlap=(xy_overlap, xy_overlap))
        image = draw_boxes(image, slide_windows)

        # draw red start/stop position on image
        cv2.line(image, (0, y_start), (width, y_start), (255, 0, 0), 2)
        cv2.line(image, (0, y_stop), (width, y_stop), (255, 0, 0), 2)

        return image

    def on_press(self, event):
        if event.key == "ctrl+r":
            self.on_reset()
        elif event.key == "ctrl+p":
            self.on_print()

    def on_print(self, args=None):
        print("""
top_y = {}
bottom_y = {}
xy_window = {}
xy_overlap = {}
    """.format(round(self.top_y_slider.val, 3), round(self.bottom_y_slider.val, 3),
               self.xy_window_slider.val, round(self.xy_overlap_slider.val, 3)
               ))

    def on_reset(self, args=None):
        print("Reset")

        self.update_val(self.top_y_slider, top_y)
        self.update_val(self.bottom_y_slider, bottom_y)
        self.update_val(self.xy_window_slider, xy_window)
        self.update_val(self.xy_overlap_slider, xy_overlap)

        self.plugin.filter_image()

    def show(self):
        self.viewer.show()

    def update_val(self, comp, newval):
        comp.val = newval
        comp.editbox.setText("%s" % newval)

        return newval


if __name__ == "__main__":
    RoiEditorUi("../test_images/test*.jpg").show()
