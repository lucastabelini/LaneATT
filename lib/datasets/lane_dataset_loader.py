class LaneDatasetLoader:
    def get_img_heigth(self, path):
        """Returns the image's height in pixels"""
        raise NotImplementedError()

    def get_img_width(self, path):
        """Returns the image's width in pixels"""
        raise NotImplementedError()

    def get_metrics(self, lanes, idx):
        """Returns dataset's metrics for a prediction `lanes`

        A tuple `(fp, fn, matches, accs)` should be returned, where `fp` and `fn` indicate the number of false-positives
        and false-negatives, respectively, matches` is a list with a boolean value for each
        prediction in `lanes` indicating if the prediction is a true positive and `accs` is a metric indicating the
        quality of each prediction (e.g., the IoU with an annotation)

        If the metrics can't be computed, placeholder values should be returned.
        """
        raise NotImplementedError()

    def load_annotations(self):
        """Loads all annotations from the dataset

        Should return a list where each item is a dictionary with keys `path` and `lanes`, where `path` is the path to
        the image and `lanes` is a list of lanes, represented by a list of points for example:

        return [{
            'path': 'example/path.png' # path to the image
            'lanes': [[10, 20], [20, 25]]
        }]
        """
        raise NotImplementedError()

    def eval_predictions(self, predictions, output_basedir):
        """Should return a dictionary with each metric's results
        Example:
        return {
            'F1': 0.9
            'Acc': 0.95
        }
        """
        raise NotImplementedError()

    def __getitem__(self, idx):
        """Should return the annotation with index idx"""
        raise NotImplementedError()

    def __len__(self):
        """Should return the number of samples in the dataset"""
        raise NotImplementedError()
