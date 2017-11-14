import csv

from tqdm import tqdm

from experiments import *

# csvsql --query "select * from __results order by score desc" work/__results.csv | csvlook
# csvsql --query "select * from __results where score > 0.98 order by score desc" work/__results.csv | csvlook
# csvsql --query "select * from __results order by score desc limit 10" work/__results.csv | csvlook

first_val = 5
second_val = 8

sample_size = 1000

cars, notcars, (sample_height, sample_width, sample_depth) = read_all_data()

selected_cars_indices = np.random.choice(len(cars), size=sample_size, replace=False)
selected_notcars_indices = np.random.choice(len(notcars), size=sample_size, replace=False)

cars = np.array(cars)[selected_cars_indices]
notcars = np.array(notcars)[selected_notcars_indices]


def sequence(n, ainit=0, binit=1):
    result = []
    a, b = ainit, binit
    while b < n:
        result.append(b)
        a, b = b, a + b
    return result


config_ycrcb = {
    "label": "ycrcb",
    "color_space": "YCrCb",
    "orient": sequence(96, first_val, second_val),
    "pix_per_cell": sequence(34, first_val, second_val),
    "cell_per_block": 2,
    "hog_channel": "ALL",
    "block_norm": ["L1", "L1-sqrt", "L2", "L2-Hys"],
    "transform_sqrt": [True, False],
    "spatial_size": None,
    "hist_bins": None,
    "hist_range": None,
    "spatial_feat": False,
    "hist_feat": False,
    "hog_feat": True,
    "vis": False,
    "feature_vec": True,
    "heatmap_threshold": 1,
    "retrain": True
}

fnames = [path for path in glob.iglob("test_images/test*.jpg", recursive=True)]
_, images = read_images(fnames)


def run_test(config):
    print("----------------------", flush=True)
    print(config["label"], flush=True)
    print("----------------------", flush=True)

    print("Cars size:", len(cars), flush=True)
    print("Notcars size:", len(notcars), flush=True)

    orients = config.get("orient")
    pix_per_cells = config.get("pix_per_cell")
    block_norms = config.get("block_norm")
    transform_sqrts = config.get("transform_sqrt")

    experiments_num = len(orients) * len(pix_per_cells) * len(block_norms) * len(transform_sqrts)
    print("Experiments Num:", experiments_num, flush=True)

    result_file = os.path.join("work", "__results.csv")
    open(result_file, "w").close()
    with open(result_file, "a") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(("score", "orient", "pix_per_cell", "block_norm", "transform_sqrt", "error"))

        with tqdm(total=experiments_num) as pbar:
            for orient in orients:
                for pix_per_cell in pix_per_cells:
                    for block_norm in block_norms:
                        for transform_sqrt in transform_sqrts:
                            key = "%s_orient_%s_pix_per_cell_%s_block_norm_%s_transform_sqrt_%s" % \
                                  (config["label"], orient, pix_per_cell, block_norm, transform_sqrt)

                            score = 0
                            error = ""
                            try:
                                svc, X_scaler, score = train_classifier(rescale_to_0_1(cars), rescale_to_0_1(notcars),
                                                                        color_space=config.get("color_space"),
                                                                        spatial_size=config.get("spatial_size"),
                                                                        hist_bins=config.get("hist_bins"),
                                                                        hist_range=config.get("hist_range"),
                                                                        orient=orient, pix_per_cell=pix_per_cell,
                                                                        cell_per_block=config.get("cell_per_block"),
                                                                        hog_channel=config.get("hog_channel"),
                                                                        block_norm=block_norm,
                                                                        transform_sqrt=transform_sqrt,
                                                                        vis=config.get("vis"),
                                                                        feature_vec=config.get("feature_vec"),
                                                                        spatial_feat=config.get("spatial_feat"),
                                                                        hist_feat=config.get("hist_feat"),
                                                                        hog_feat=config.get("hog_feat"),
                                                                        retrain=config.get("retrain"), debug=False)

                                linear_svc_path = os.path.join("work", "__linear_svc_%s.pkl" % key)
                                standard_scaler_path = os.path.join("work", "__standard_scaler_%s.pkl" % key)
                                save_trained_model(svc, X_scaler, linear_svc_path, standard_scaler_path)
                            except Exception as exc:
                                error = str(exc)

                            writer.writerow((float(score), int(orient), int(pix_per_cell),
                                             str(block_norm), int(transform_sqrt), str(error)))
                            f.flush()

                            pbar.update(1)


if __name__ == "__main__":
    run_test(config_ycrcb)
