import os
import cv2
import paddle
import paddle.vision.transforms as T


class ImageDataset(paddle.io.Dataset):
    def __init__(self, root_dir, class_dir):
        self.class_dir = class_dir
        self.root_dir = root_dir
        self.data = []
        self.transforms = T.Compose([
            T.Resize(224)
        ])

        with open('Sample name indices for the same domain/{}.txt'.format(class_dir)) as f:
            for line in f.readlines():
                line = line.strip()
                label = int(line.split("-")[0])
                label = label - 1
                if len(line) > 0:
                    self.data.append([line, label])

    def __getitem__(self, item):
        img_name, label = self.data[item]
        img_item_path = os.path.join(self.root_dir, img_name)
        img = cv2.imread(img_item_path)
        img = self.transforms(img)
        img = img.transpose(2, 0, 1)
        img = img.astype('float32')
        label = int(label)

        return img, label

    def __len__(self):
        return len(self.data)