from skimage.viewer import CollectionViewer
from skimage.viewer.plugins import Plugin
from skimage.viewer.widgets import CheckBox, ComboBox, Slider

from experiments import *

color_space_index = 0
orient = 9
pix_per_cell = 8
cell_per_block = 2
block_norm_index = 0
transform_sqrt = True
spatial_size = 32
hist_bins = 32
hist_range_start = 0
hist_range_end = 256
hog_channel_index = 0


class FeatureEditorUi:
    def __init__(self, image_scale=(512, 512), size=20):
        self.image_scale = image_scale
        self.size = size

        self.plugin = Plugin(image_filter=self.image_filter, dock="right")

        self.configurations = ["Color Space", "HOG", "Spatial", "Hist", "Extract All"]
        self.color_spaces = ["YCrCb", "LAB", "HSV", "LUV", "YUV", "HLS", "RGB"]
        self.image_collection = ["cars", "notcars"]
        self.block_norms = ["L1", "L1-sqrt", "L2", "L2-Hys"]
        self.hog_channel = ["ALL", "0", "1", "2"]

        self.image_collection_combobox = ComboBox("cars_notcars", self.image_collection)
        self.show_origin_checkbox = CheckBox("show_orig", value=False, alignment="left")
        self.configuration_combobox = ComboBox("configuration", self.configurations)

        self.colorspace_combobox = ComboBox("color_space", self.color_spaces)
        self.orient_slider = Slider("orient", 1, 50, value=orient, value_type="int")
        self.pix_per_cell_slider = Slider("pix_per_cell", 1, 256, value=pix_per_cell, value_type="int")
        self.cell_per_block_slider = Slider("cell_per_block", 1, 256, value=cell_per_block, value_type="int")
        self.block_norm_combobox = ComboBox("block_norm", self.block_norms)
        self.transform_sqrt_checkbox = CheckBox("transform_sqrt", value=transform_sqrt, alignment="left")
        self.spatial_size_slider = Slider("spatial_size", 1, 256, value=spatial_size, value_type="int")
        self.hist_bins_slider = Slider("hist_bins", 1, 256, value=hist_bins, value_type="int")
        self.hist_range_start_slider = Slider("hist_range_start", 0, 256, value=hist_range_start, value_type="int")
        self.hist_range_end_slider = Slider("hist_range_end", 0, 256, value=hist_range_end, value_type="int")
        self.hog_channel_combobox = ComboBox("hog_channel", self.hog_channel)

        self.plugin += self.image_collection_combobox
        self.plugin += self.show_origin_checkbox
        self.plugin += self.configuration_combobox
        self.plugin += self.colorspace_combobox
        self.plugin += self.orient_slider
        self.plugin += self.pix_per_cell_slider
        self.plugin += self.cell_per_block_slider
        self.plugin += self.block_norm_combobox
        self.plugin += self.transform_sqrt_checkbox
        self.plugin += self.spatial_size_slider
        self.plugin += self.hist_bins_slider
        self.plugin += self.hist_range_start_slider
        self.plugin += self.hist_range_end_slider
        self.plugin += self.hog_channel_combobox

        self.cars_images, self.notcars_images = load_cars_notcars(cars_path="../work/cars.pkl",
                                                                  notcars_path="../work/notcars.pkl")
        self.sample_height, self.sample_width, self.sample_depth = self.cars_images[0].shape

        self.rnd_cars_indices = np.random.choice(len(self.cars_images), size=self.size, replace=False)
        self.rnd_notcars_indices = np.random.choice(len(self.notcars_images), size=self.size, replace=False)

        self.cars_images = np.uint8(self.cars_images)[self.rnd_cars_indices]
        self.notcars_images = np.uint8(self.notcars_images)[self.rnd_notcars_indices]

        self.cars_selected = True
        self.viewer = CollectionViewer(self.cars_images)

        self.viewer.connect_event("key_press_event", self.on_press)
        self.viewer += self.plugin
        print("Done")

    def image_filter(self, image, *args, **kwargs):
        image = np.copy(image)

        show_orig = kwargs["show_orig"]
        cars_notcars = kwargs["cars_notcars"]
        color_space = kwargs["color_space"]
        configuration = kwargs["configuration"]
        orient = kwargs["orient"]
        pix_per_cell = kwargs["pix_per_cell"]
        cell_per_block = kwargs["cell_per_block"]
        block_norm = kwargs["block_norm"]
        transform_sqrt = kwargs["transform_sqrt"]
        spatial_size = kwargs["spatial_size"]
        hist_bins = kwargs["hist_bins"]
        hist_range_start = kwargs["hist_range_start"]
        hist_range_end = kwargs["hist_range_end"]
        hog_channel = kwargs["hog_channel"]

        car_index = self.viewer.slider.val

        if cars_notcars == "cars" and not self.cars_selected:
            self.viewer.image_collection = self.cars_images
            self.viewer.update_index(None, 0)
            self.cars_selected = True
        elif cars_notcars == "notcars" and self.cars_selected:
            self.viewer.image_collection = self.notcars_images
            self.viewer.update_index(None, 0)
            self.cars_selected = False

        if show_orig:
            return image

        target_color_space = cv2.COLOR_RGB2HSV

        if color_space == "RGB":
            target_color_space = None
        elif color_space == "YCrCb":
            target_color_space = cv2.COLOR_RGB2YCrCb
        elif color_space == "LAB":
            target_color_space = cv2.COLOR_RGB2LAB
        elif color_space == "LUV":
            target_color_space = cv2.COLOR_RGB2LUV
        elif color_space == "YUV":
            target_color_space = cv2.COLOR_RGB2YUV
        elif color_space == "HLS":
            target_color_space = cv2.COLOR_RGB2HLS

        converted_image = image
        converted_car_image = self.cars_images[car_index]
        converted_notcars_image = self.notcars_images[car_index]

        if target_color_space is None:
            # image already in RGB
            pass
        else:
            converted_image = cv2.cvtColor(image, target_color_space)
            converted_car_image = cv2.cvtColor(converted_car_image, target_color_space)
            converted_notcars_image = cv2.cvtColor(converted_notcars_image, target_color_space)

        if configuration == "Color Space":
            return combine_images_vert(converted_car_image, converted_notcars_image)

        if configuration == "HOG":
            hog_car_image = convert_hog(converted_car_image, block_norm, cell_per_block, hog_channel, orient,
                                        pix_per_cell, transform_sqrt)
            hog_notcar_image = convert_hog(converted_notcars_image, block_norm, cell_per_block, hog_channel, orient,
                                           pix_per_cell, transform_sqrt)

            hog_car_image = np.expand_dims(hog_car_image, axis=2)
            hog_notcar_image = np.expand_dims(hog_notcar_image, axis=2)

            return combine_images_vert(hog_car_image, hog_notcar_image).squeeze()

        if configuration == "Spatial":
            spatial_car_image = self.show_spatial(converted_car_image, spatial_size)
            spatial_notcar_image = self.show_spatial(converted_notcars_image, spatial_size)

            return combine_images_vert(spatial_car_image, spatial_notcar_image)

        if configuration == "Hist":
            hist_car_image = self.show_hist(converted_car_image, hist_bins, hist_range_end, hist_range_start,
                                            "Car")
            hist_notcar_image = self.show_hist(converted_notcars_image, hist_bins, hist_range_end, hist_range_start,
                                               "NotCar")

            return combine_images_vert(hist_car_image, hist_notcar_image)

        if configuration == "Extract All":
            return self.show_extract_all(car_index, block_norm, cars_notcars, cell_per_block, color_space,
                                         converted_image, hist_bins, hist_range_end, hist_range_start, hog_channel,
                                         orient, pix_per_cell, spatial_size, transform_sqrt)

        return converted_image

    def show_extract_all(self, car_index, block_norm, cars_notcars, cell_per_block, color_space, converted_image,
                         hist_bins,
                         hist_range_end, hist_range_start, hog_channel, orient, pix_per_cell, spatial_size,
                         transform_sqrt):
        print("Run extract all features")
        cars_images, notcars_images = self.cars_images, self.notcars_images
        # cars_images, notcars_images, _ = read_all_data()
        # cars_images = cars_images[:self.size]
        # notcars_images = notcars_images[:self.size]
        car_features = extract_features(rescale_to_0_1(cars_images), color_space=color_space,
                                        spatial_size=(spatial_size, spatial_size),
                                        hist_bins=hist_bins, hist_range=(hist_range_start, hist_range_end),
                                        orient=orient, pix_per_cell=pix_per_cell,
                                        cell_per_block=cell_per_block,
                                        hog_channel=hog_channel, block_norm=block_norm,
                                        transform_sqrt=transform_sqrt, vis=False, feature_vec=True,
                                        spatial_feat=True, hist_feat=True, hog_feat=True)
        print("Car features extracted")
        notcar_features = extract_features(rescale_to_0_1(notcars_images), color_space=color_space,
                                           spatial_size=(spatial_size, spatial_size),
                                           hist_bins=hist_bins, hist_range=(hist_range_start, hist_range_end),
                                           orient=orient, pix_per_cell=pix_per_cell,
                                           cell_per_block=cell_per_block,
                                           hog_channel=hog_channel, block_norm=block_norm,
                                           transform_sqrt=transform_sqrt, vis=False, feature_vec=True,
                                           spatial_feat=True, hist_feat=True, hog_feat=True)
        print("NotCar features extracted")
        if len(car_features) > 0 and cars_notcars == "cars":
            X = np.vstack((car_features, notcar_features)).astype(np.float64)

            # Fit a per-column scaler
            X_scaler = StandardScaler().fit(X)

            # Apply the scaler to X
            scaled_X = X_scaler.transform(X)

            # Plot an example of raw and scaled features
            plt.close()
            fig = plt.figure(figsize=(12, 4))
            plt.subplot(131)
            plt.imshow(cars_images[car_index])
            plt.title('Original Image')
            plt.subplot(132)
            plt.plot(X[car_index].squeeze())
            plt.title('Raw Features')
            plt.subplot(133)
            plt.plot(scaled_X[car_index].squeeze())
            plt.title('Normalized Features')
            fig.tight_layout()
            all_features = plot_to_image()

            return all_features
        else:
            print("Features NOT found!!!")
            return converted_image

    def show_hist(self, converted_image, hist_bins, hist_range_end, hist_range_start, title=""):
        hist_features = color_hist(converted_image, nbins=hist_bins, bins_range=(hist_range_start, hist_range_end))
        plt.close()
        plt.plot(hist_features.squeeze())
        plt.title(title)
        plt.ylim([0, np.max(hist_features)])
        hist_image = plot_to_image()
        print("Hist Number of features: ", len(hist_features))
        return hist_image

    def show_spatial(self, converted_image, spatial_size):
        spatial_features = bin_spatial(converted_image, size=(spatial_size, spatial_size))
        plt.close()
        plt.plot(spatial_features.squeeze())
        plt.ylim([0, np.max(spatial_features)])
        spatial_image = plot_to_image()
        print("Spatial Number of features: ", len(spatial_features))
        return spatial_image

    def on_press(self, event):
        if event.key == "ctrl+r":
            self.on_reset()
        elif event.key == "ctrl+p":
            self.on_print()

    def on_print(self, args=None):
        print("""
color_space_index = {}
orient = {}
pix_per_cell = {}
cell_per_block = {}
block_norm_index = {}
transform_sqrt = {}
spatial_size = {}
hist_bins = {}
hist_range_start = {}
hist_range_end = {}
hog_channel_index = {}
""".format(self.colorspace_combobox.index, self.orient_slider.val, self.pix_per_cell_slider.val,
           self.cell_per_block_slider.val, self.block_norm_combobox.index, self.transform_sqrt_checkbox.val,
           self.spatial_size_slider.val, self.hist_bins_slider.val, self.hist_range_start_slider.val,
           self.hist_range_end_slider.val, self.hog_channel_combobox.index
           ))

    def on_reset(self, args=None):
        print("Reset")

        self.update_combobox(self.colorspace_combobox, color_space_index)
        self.update_val(self.orient_slider, orient)
        self.update_val(self.pix_per_cell_slider, pix_per_cell)
        self.update_val(self.cell_per_block_slider, cell_per_block)
        self.update_combobox(self.block_norm_combobox, block_norm_index)
        self.update_checkbox(self.transform_sqrt_checkbox, transform_sqrt)
        self.update_val(self.spatial_size_slider, spatial_size)
        self.update_val(self.hist_bins_slider, hist_bins)
        self.update_val(self.hist_range_start_slider, hist_range_start)
        self.update_val(self.hist_range_end_slider, hist_range_end)
        self.update_combobox(self.hog_channel_combobox, hog_channel_index)

        self.plugin.filter_image()

    def show(self):
        self.viewer.show()

    def update_checkbox(self, comp, newval):
        comp.val = newval
        return newval

    def update_combobox(self, comp, index):
        comp.index = index
        return index

    def update_val(self, comp, newval):
        comp.val = newval
        comp.editbox.setText("%s" % newval)

        return newval


if __name__ == "__main__":
    FeatureEditorUi().show()
