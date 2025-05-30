# sys libs
import os

import PIL.Image
# other libs
import numpy as np
import torch
import torch.utils.data


def pil_loader(path):
    with open(path, 'rb') as f:
        img = PIL.Image.open(f)
        return img.convert('RGB')

#frm https://github.com/zxhuang1698/interpretability-by-parts/blob/master/src/cub200/cub200.py
class CUB200(torch.utils.data.Dataset):
    """
    CUB200 dataset.

    Variables
    ----------
        _root, str: Root directory of the dataset.
        _train, bool: Load train/test data.
        _transform, callable: A function/transform that takes in a PIL.Image
            and transforms it.
        _train_data, list of np.array.
        _train_labels, list of int.
        _train_parts, list np.array.
        _train_boxes, list np.array.
        _test_data, list of np.array.
        _test_labels, list of int.
        _test_parts, list np.array.
        _test_boxes, list np.array.
    """
    def __init__(self, root, train=True, transform=None, resize=448, req_label=None, three_classes=False, mask_type='gt', crop_to_bbox=False):
        """
        Load the dataset.

        Args
        ----------
        root: str
            Root directory of the dataset.
        train: bool
            train/test data split.
        transform: callable
            A function/transform that takes in a PIL.Image and transforms it.
        resize: int
            Length of the shortest of edge of the resized image. Used for transforming landmarks and bounding boxes.

        """
        self._root = root
        self._train = train
        self._transform = transform
        self.loader = pil_loader
        self.newsize = resize
        self.req_label = req_label
        self.three_classes = three_classes
        self.mask_type = mask_type
        self.crop_to_bbox = crop_to_bbox
        # 15 key points provided by CUB
        self.num_kps = 15

        if not os.path.isdir(root):
            os.mkdir(root)
        # Load all data into memory for best IO efficiency. This might take a while
        if self._train:
            self._train_data, self._train_labels, self._train_parts, self._train_boxes = self._get_file_list(train=True)
            # assert (len(self._train_data) == 5994
            #         and len(self._train_labels) == 5994)
        else:
            self._test_data, self._test_labels, self._test_parts, self._test_boxes = self._get_file_list(train=False)
            # assert (len(self._test_data) == 5794
            #         and len(self._test_labels) == 5794)

    def __getitem__(self, index):
        """
        Retrieve data samples.

        Args
        ----------
        index: int
            Index of the sample.

        Returns
        ----------
        image: torch.FloatTensor, [3, H, W]
            Image of the given index.
        target: int
            Label of the given index.
        parts: torch.FloatTensor, [15, 4]
            Landmark annotations.
        boxes: torch.FloatTensor, [5, ]
            Bounding box annotations.
        """
        # load the variables according to the current index and split
        if self._train:
            image_path = self._train_data[index]
            target = self._train_labels[index]
            parts = self._train_parts[index]
            boxes = self._train_boxes[index]

        else:
            image_path = self._test_data[index]
            target = self._test_labels[index]
            parts = self._test_parts[index]
            boxes = self._test_boxes[index]

        # load the image
        image = self.loader(image_path)
        if self.mask_type == 'gt':
            mask = self.loader(image_path.replace('images', 'segmentations').replace('.jpg', '.png'))
        elif self.mask_type == 'unsup':
            mask = self.loader(image_path.replace('/images', 'pseudolabels/').replace('.jpg', '.png'))
        else:
            mask = self.loader(image_path.replace('/images', 'supervisedlabels3/' if self.three_classes else 'supervisedlabels/').replace('.jpg', '.png'))
        image = np.array(image)
        mask = np.array(mask.convert('L')) if self.mask_type == 'sup' else  (np.array(mask.convert('L')) == 255).astype(np.uint8)

        # numpy arrays to pytorch tensors
        parts = torch.from_numpy(parts).float()
        boxes = torch.from_numpy(boxes).float()

        if self.crop_to_bbox:
            image = image[boxes[2].int().item():boxes[2].int().item()+boxes[4].int().item(), boxes[1].int().item():boxes[1].int().item()+boxes[3].int().item()]
            mask = mask[boxes[2].int().item():boxes[2].int().item()+boxes[4].int().item(), boxes[1].int().item():boxes[1].int().item()+boxes[3].int().item()]
            # transform 15 landmarks according to the new shape
            # each landmark has a 4-element annotation: <landmark_id, column, row, existence>
            for j in range(self.num_kps):

                # step in only when the current landmark exists
                if abs(parts[j][-1]) > 1e-5:
                    # calculate the new location according to the new shape
                    parts[j][-3] = max(parts[j][-3] - boxes[1], 0)
                    parts[j][-2] = max(parts[j][-2] - boxes[2], 0)

        # calculate the resize factor
        # if original image height is larger than width, the real resize factor is based on width
        if image.shape[0] >= image.shape[1]:
            factor = self.newsize / image.shape[1]
        else:
            factor = self.newsize / image.shape[0]

        # transform 15 landmarks according to the new shape
        # each landmark has a 4-element annotation: <landmark_id, column, row, existence>
        for j in range(self.num_kps):

            # step in only when the current landmark exists
            if abs(parts[j][-1]) > 1e-5:
                # calculate the new location according to the new shape
                parts[j][-3] = parts[j][-3] * factor
                parts[j][-2] = parts[j][-2] * factor

        # rescale the annotation of bounding boxes
        # the annotation format of the bounding boxes are <image_id, col of top-left corner, row of top-left corner, width, height>
        boxes[1:] *= factor

        # convert the image into a PIL image for transformation
        image = PIL.Image.fromarray(image)
        mask = PIL.Image.fromarray(mask)

        # apply transformation
        if self._transform is not None:
            if isinstance(self._transform, list):
                image = self._transform[0](image)
                mask = self._transform[1](mask)
                return image, mask, target, parts, boxes, image_path
            else:
                image = self._transform(image)

        return image, target, parts, boxes, image_path

    def __len__(self):
        """Return the length of the dataset."""
        if self._train:
            return len(self._train_data)
        return len(self._test_data)

    def _get_file_list(self, train=True):
        """Prepare the data for train/test split and save onto disk."""
        # load the list into numpy arrays
        image_path = self._root + '/images/'
        id2name = np.genfromtxt(self._root + '/images.txt', dtype=str)
        id2train = np.genfromtxt(self._root + '/train_test_split.txt', dtype=int)
        id2part = np.genfromtxt(self._root + '/parts/part_locs.txt', dtype=float)
        id2box = np.genfromtxt(self._root + '/bounding_boxes.txt', dtype=float)

        # creat empty lists
        train_data = []
        train_labels = []
        train_parts = []
        train_boxes = []
        test_data = []
        test_labels = []
        test_parts = []
        test_boxes = []
        # iterating all samples in the whole dataset
        for id_ in range(id2name.shape[0]):

            # Label starts with 1
            label = int(id2name[id_, 1][:3])
            if self.three_classes:
                if not train:
                    if label != self.req_label:
                        continue
                if train:
                    if label not in [1, 2, 3]:
                        continue
            # load each variable
            image = os.path.join(image_path, id2name[id_, 1])
            parts = id2part[id_*self.num_kps : id_*self.num_kps+self.num_kps][:, 1:]
            boxes = id2box[id_]
            # training split
            if id2train[id_, 1] == 1:
                train_data.append(image)
                train_labels.append(label)
                train_parts.append(parts)
                train_boxes.append(boxes)
            # testing split
            else:
                test_data.append(image)
                test_labels.append(label)
                test_parts.append(parts)
                test_boxes.append(boxes)

        # return accoring to different splits
        if train == True:
            return train_data, train_labels, train_parts, train_boxes
        else:
            return test_data, test_labels, test_parts, test_boxes
