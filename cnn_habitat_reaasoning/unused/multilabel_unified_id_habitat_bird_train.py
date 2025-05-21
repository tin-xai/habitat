# %%
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict

import torch
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid
from sklearn.model_selection import train_test_split
from torchvision import models

import timm
from timm.loss import LabelSmoothingCrossEntropy

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns

import os
import json
import sys
from tqdm import tqdm
import random
import time
import copy
from datetime import datetime

# %% config
class CFG:
    seed = 42
    dataset = 'cub' # cub, nabirds, inat21
    model_name = 'resnet101' #resnet50, resnet101, efficientnet_b6, densenet121, tf_efficientnetv2_b0
    pretrained = True
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

    # data params
    dataset2num_classes = {'cub': 200, 'nabirds': 555, 'inat21':1486}
    bird_num_classes = dataset2num_classes[dataset]
    habitat_num_classes = 200
    alpha = 0

    dataset2path = {
        'cub': '/home/tin/datasets/cub',
        'nabirds': '/home/tin/datasets/nabirds/',
        'inat21': '/home/tin/datasets/inaturalist2021_onlybird/'
    }

    # cutmix
    cutmix = False
    cutmix_beta = 1.

    #hyper params
    batch_size = 64
    lr = 1e-5 # 1e-3
    image_size = 224
    epochs = 10

    # explaination
    explaination = False

    # inat21
    inat21_df_path = 'inat21_onlybirds.csv'
    write_inat_to_df = not os.path.exists(inat21_df_path)

    # focal loss
    fl_alpha = 1.0  # alpha of focal_loss
    fl_gamma = 2.0  # gamma of focal_loss
    class_weights = []

    # save folder
    save_folder    = f'./results/{dataset}_multilabel_unified_{model_name}_{str(datetime.now().strftime("%m_%d_%Y-%H:%M:%S"))}/'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

# %%
def set_seed(seed=None, cudnn_deterministic=True):
    if seed is None:
        seed = 42

    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = False

set_seed(seed=CFG.seed)

# %%
def Augment(train = False):
    if train:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter()]), p=0.1),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.RandomErasing(p=0.2, value='random')
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    
    return transform
    
class Unified_Inat21_Dataset(Dataset):
    def __init__(self, dataroot, df, transform=None, mode='train'):
        self.df = df
        self.df = self.df[self.df['Mode'] == mode]
        self.mode = mode
        self.transform = transform
        
        # cluster
        class_cluster_filepath = f"/home/tin/projects/reasoning/plain_clip/class_inat21_clusters_1486.json"
        f = open(class_cluster_filepath, 'r')
        idx2cluster = json.load(f)
        
        folderclasses = os.listdir(f"{dataroot}/bird_train/")
        folderclass2class = {}

        for folder_name in folderclasses:
                name_parts = folder_name.split('_')
                name = name_parts[-2] + ' ' + name_parts[-1]
                
                folderclass2class[folder_name] = name

        class2folderclass = {v:k for k,v in folderclass2class.items()}

        self.habitat_img_2_class = {}
        for idx, classes in idx2cluster.items():
            for cls in classes:
                # convert cls to folder class
                folderclass = class2folderclass[cls]
                habitat_image_paths = os.listdir(f"{dataroot}/inat21_inpaint_all/{folderclass}")
                for img_path in habitat_image_paths:
                    self.habitat_img_2_class[img_path] = int(idx) - 1

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        orig_image_path, label, mode = self.df.iloc[index].to_list()
        inpaint_image_path = orig_image_path.replace("bird_train", "inat21_inpaint_all")

        label = int(label)
        image = Image.open(orig_image_path).convert("RGB")
        inpaint_image = Image.open(inpaint_image_path).convert("RGB")
        label2 = self.habitat_img_2_class[inpaint_image_path.split('/')[-1]]

        if self.transform is not None:
            image = self.transform(image)
            inpaint_image = self.transform(inpaint_image)
        
        return (torch.cat((image, inpaint_image), 0), label, label2)

class ImageFolderWithTwoPaths(datasets.ImageFolder):
    def __init__(self, root1, root2, transform=None, target_transform=None):
        super(ImageFolderWithTwoPaths, self).__init__(root1, transform, target_transform)
        self.root2 = root2
        multilabel_filepath = f"/home/tin/projects/reasoning/plain_clip/{CFG.dataset}_multilabel_{CFG.habitat_num_classes}.json"
        f = open(multilabel_filepath, 'r')
        self.multilabel_graph = json.load(f)
        
        # folderclasses = os.listdir(root1)
        # folderclass2class = {}
        # if CFG.dataset == 'cub':
        #     for cls in folderclasses:
        #         name = cls.split('.')[1]
                
        #         if len(name.split('_')) > 2:
        #             name_parts = name.split('_')
        #             if len(name.split('_')) == 3:
        #                 name = name_parts[0] + '-' + name_parts[1] + ' ' + name_parts[2]
        #             else:
        #                 name = name_parts[0] + '-' + name_parts[1] + '-' + name_parts[2] + ' ' + name_parts[3]
        #         else:
        #             name = name.replace('_', ' ')
        #         folderclass2class[cls] = name
        # else:
        #     nabirds_idx2class_file = '/home/tin/projects/reasoning/scraping/nabird_data/nabird_classes.txt'
        #     nabirds_idx2class = {}
        #     idx2class = pd.read_table(nabirds_idx2class_file, header=None)
        #     idx2class['id'] = idx2class[0].apply(lambda s: int(s.split(' ')[0]))
        #     idx2class['label_name'] = idx2class[0].apply(lambda s: ' '.join(s.split(' ')[1:]))
        #     for id, label_name in zip(idx2class['id'], idx2class['label_name']):
        #         nabirds_idx2class[id] = label_name
        #     for cls_idx in folderclasses:
        #         name = nabirds_idx2class[int(cls_idx)]
        #         folderclass2class[cls_idx] = name


    def __getitem__(self, index):
        
        path, label = self.samples[index]
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        path2 = self.root2 +"/" + path.split("/")[-2] + "/" + path.split("/")[-1]
        img2 = self.loader(path2)
        # label2 = self.multilabel_graph[str(label+1)]
        # label2 = [l-1 for l in label2]
        # label2_mat = np.zeros((200, 1))
        # label2_mat[label2] = 1.
        label2 = label
        if self.transform is not None:
            img2 = self.transform(img2)

        return (torch.cat((img, img2), 0), label, label2)
        # return img, label
    
def get_data_loaders(dataset, batch_size):
    """
    Get the train, val, test dataloader
    """
    if dataset in ['cub', 'nabirds']:
        inpaint_train_img_folder = 'CUB_inpaint_all_train/' if dataset == 'cub' else 'train_inpaint/'
        orig_train_img_folder = 'CUB/train/' if dataset == 'cub' else 'train/'
        inpaint_test_img_folder = 'CUB_inpaint_all_test/' if dataset == 'cub' else 'test_inpaint/'
        orig_test_img_folder = 'CUB/test/' if dataset == 'cub' else 'test/'
        
        inpaint_train_data_dir = f"{CFG.dataset2path[dataset]}/{inpaint_train_img_folder}"
        orig_train_data_dir = f"{CFG.dataset2path[dataset]}/{orig_train_img_folder}"
        inpaint_test_data_dir = f"{CFG.dataset2path[dataset]}/{inpaint_test_img_folder}"
        orig_test_data_dir = f"{CFG.dataset2path[dataset]}/{orig_test_img_folder}"

        train_data = ImageFolderWithTwoPaths(root1=orig_train_data_dir, root2=inpaint_train_data_dir, transform=Augment(train=True))
        test_data = ImageFolderWithTwoPaths(root1=orig_test_data_dir, root2=inpaint_test_data_dir, transform=Augment(train=False))
        val_data = test_data

        train_data_len = len(train_data)
        valid_data_len = len(val_data)
        test_data_len = len(test_data)
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)

        classes = train_data.classes
        bird_classes = classes
        habitat_classes = ['a' for a in range(CFG.habitat_num_classes)]
        # bird_classes, habitat_classes = train_data.bird_classes, train_data.habitat_classes
        return (train_loader, val_loader, test_loader, train_data_len, valid_data_len, test_data_len, bird_classes, habitat_classes)
    
    elif dataset == 'inat21':
        orig_data_dir = CFG.dataset2path[dataset] + '/bird_train'
        inpaint_data_dir = CFG.dataset2path[dataset] + '/inat21_inpaint_all'

        def compute_class_weights():
            label_folders = os.listdir(orig_data_dir)
            for i, cls in enumerate(label_folders):
                folder_path = f"{orig_data_dir}/{cls}"
                num_samples = len(os.listdir(folder_path))
                CFG.class_weights.append(num_samples)
            CFG.class_weights =[num_sample/sum(CFG.class_weights) for i, num_sample in enumerate(CFG.class_weights)] 
        
        compute_class_weights()

        train_data_percent = 0.8
        test_data_percent = 0.1
        valid_data_percent = 0.1

        # Get the list of image file paths and their corresponding labels
        data = []
        label2idx = {}
        for i, label in enumerate(os.listdir(orig_data_dir)):
            label2idx[label] = i
        for label in os.listdir(orig_data_dir):
            label_folder = os.path.join(orig_data_dir, label)
            if os.path.isdir(label_folder):
                for filename in os.listdir(label_folder):
                    image_path = os.path.join(label_folder, filename)
                    data.append((image_path, label2idx[label]))
        
        y = [idx for path, idx in data]
        y =  np.array(y)
        train_data, test_data = train_test_split(data, test_size=test_data_percent, stratify=y, random_state=CFG.seed)
        train_data, val_data = train_test_split(train_data, test_size=valid_data_percent, random_state=CFG.seed)

        if CFG.write_inat_to_df:
            # write to dataframe
            train_df = pd.DataFrame(train_data, columns=["Path", "Label"])
            train_df["Mode"] = ["train" for _ in range(len(train_data))]
            val_df = pd.DataFrame(val_data, columns=["Path", "Label"])
            val_df["Mode"] = ["val" for _ in range(len(val_data))]
            test_df = pd.DataFrame(test_data, columns=["Path", "Label"])
            test_df["Mode"] = ["test" for _ in range(len(test_data))]
            df = pd.concat([train_df, val_df, test_df], ignore_index=True)
            df.to_csv(CFG.inat21_df_path, index=False)
        else:
            df = pd.read_csv(CFG.inat21_df_path)
        # generate subset based on indices
        train_dataset = Unified_Inat21_Dataset(CFG.dataset2path[dataset], df, transform=Augment(train=True), mode='train')
        test_dataset = Unified_Inat21_Dataset(CFG.dataset2path[dataset], df, transform=Augment(train=False), mode='test')
        val_dataset = Unified_Inat21_Dataset(CFG.dataset2path[dataset], df, transform=Augment(train=False), mode='val')

        train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=4)

        classes = os.listdir(orig_data_dir)
        bird_classes = classes
        habitat_classes = ['a' for a in range(1486)]

        train_data_len, valid_data_len, test_data_len = len(train_dataset), len(val_dataset), len(test_dataset)

    return (train_loader, val_loader, test_loader, train_data_len, valid_data_len, test_data_len, bird_classes, habitat_classes)#, classes)
# %%
(train_loader, val_loader, test_loader, train_data_len, valid_data_len, test_data_len, bird_classes, habitat_classes) = get_data_loaders(CFG.dataset, CFG.batch_size)
# %%
visualize_loader = False
if CFG.dataset == 'inat21' and visualize_loader:
    class_counts = defaultdict(int)

    for _, labels in train_loader:
        for label in labels:
            label = label.item()
            class_counts[label] += 1
    print("Counting finished !!!")

    num_bins = 5  
    num_classes = len(class_counts)
    print("Num classes: ", num_classes)

    bins = defaultdict(int)
    other_bin_count = 0
    bin_id = 1
    for i, (class_label, count) in enumerate(class_counts.items()):
        if i < (num_classes/num_bins)*bin_id:
            bins[bin_id] += count
        else:
            bin_id += 1
    
    for bin_id, count in bins.items():
        print(f"Bin {bin_id}: {count} samples")

    # Plot the reduced class statistics
    plt.bar(bins.keys(), bins.values())
    plt.xlabel("Class Label")
    plt.ylabel("Sample Count")
    plt.title("Reduced Class Statistics")
    plt.show()
# %%
dataloaders = {
    "train":train_loader,
    "val": val_loader,
    "test": test_loader
}
dataset_sizes = {
    "train":train_data_len,
    "val": valid_data_len,
    "test": test_data_len
}
dataset_sizes

# %%
class MultiTaskModel(nn.Module):
    def __init__(self, num_classes_task1=CFG.bird_num_classes, num_classes_task2=CFG.habitat_num_classes):
        super(MultiTaskModel, self).__init__()

        self.backbone = timm.create_model('resnet101', in_chans=6, num_classes=0, pretrained=True)
        num_features = self.backbone.num_features
        
        self.branch1 = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes_task1)
        )
        
        self.branch2 = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes_task2)
        )
        
    def forward(self, x):
        features = self.backbone(x)

        output_task1 = self.branch1(features)
        output_task2 = self.branch2(features)

        return output_task1, output_task2

from FeatureExtractors import ResNet_AvgPool_classifier, Bottleneck

class MultiTaskModel_2(nn.Module):
    def __init__(self, num_classes_task1=CFG.bird_num_classes, num_classes_task2=CFG.habitat_num_classes):
        super(MultiTaskModel_2, self).__init__()

        self.backbone = ResNet_AvgPool_classifier(Bottleneck, [3, 4, 6, 4])
        my_model_state_dict = torch.load('/home/tin/projects/reasoning/cnn_habitat_reaasoning/visual_correspondence_XAI/ResNet50/CUB_iNaturalist_17/Forzen_Method1-iNaturalist_avgpool_200way1_85.83_Manuscript.pth')
        self.backbone.load_state_dict(my_model_state_dict, strict=True)

        # Freeze backbone (for training only)
        for param in list(self.backbone.parameters())[:-2]:
            param.requires_grad = False

        num_features = 200#self.backbone.num_features
        
        # self.branch1 = nn.Sequential(
        #     nn.Linear(num_features, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, num_classes_task1)
        # )
        
        self.branch2 = nn.Sequential(
            nn.Linear(num_features, 100),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(100, num_classes_task2)
        )
        
    def forward(self, x):
        features1 = self.backbone(x[:,:3,:,:])
        features2 = self.backbone(x[:,3:,:,:])

        output_task1 = features1#self.branch1(features1)
        output_task2 = self.branch2(features2)

        return output_task1, output_task2

# %%
class GatedFusion(nn.Module):
    
    def __init__(self, dim1=200, dim2=200):
        
        super(GatedFusion, self).__init__()
        
        self.gate_1 = nn.Linear(dim1+dim2, dim2)
        self.gate_2 = nn.Linear(dim1+dim2, dim2)
        
        self.layer_norm = nn.LayerNorm(dim2)
   
    def forward(self, ftrs1, ftrs2):
        ftrs2 = ftrs2.squeeze()
    
        ftrs1_weight = F.sigmoid(self.gate_1(torch.cat((ftrs1, ftrs2), dim=1)))
        ftrs2_weight = F.sigmoid(self.gate_2(torch.cat((ftrs1, ftrs2), dim=1)))
        
        return self.layer_norm(
            ftrs1 * ftrs1_weight + ftrs2 * ftrs2_weight
        )
    
class MultiTaskModel_3(nn.Module):
    def __init__(self, num_classes_task1=CFG.bird_num_classes, num_classes_task2=CFG.habitat_num_classes):
        super(MultiTaskModel_3, self).__init__()

        self.backbone1 = ResNet_AvgPool_classifier(Bottleneck, [3, 4, 6, 4])
        my_model_state_dict = torch.load('/home/tin/projects/reasoning/cnn_habitat_reaasoning/visual_correspondence_XAI/ResNet50/CUB_iNaturalist_17/Forzen_Method1-iNaturalist_avgpool_200way1_85.83_Manuscript.pth')
        self.backbone1.load_state_dict(my_model_state_dict, strict=True)

        # Freeze backbone (for training only)
        for param in list(self.backbone1.parameters())[:-2]:
            param.requires_grad = False

        self.backbone2 = timm.create_model('resnet152', in_chans=3, num_classes=0, pretrained=True)
        num_features = self.backbone2.num_features
        
        self.branch1 = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, 200)
        )

        self.gated_fusion = GatedFusion(dim1=200, dim2=200)
        self.sigm = nn.Sigmoid()


    def forward(self, x):
        features1 = self.backbone1(x[:,:3,:,:])
        features2 = self.backbone2(x[:,3:,:,:])
        features2 = self.branch1(features2)

        output_task1 = features1#self.branch1(features1)
        output_task2 = self.gated_fusion(features1, features2)

        return output_task1, output_task2
# %%
# cut mix rand bbox
def rand_bbox(size, lam, to_tensor=True):
    W = size[-2]
    H = size[-1]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    #uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    if to_tensor:
        bbx1 = torch.tensor(bbx1)
        bby1 = torch.tensor(bby1)
        bbx2 = torch.tensor(bbx2)
        bby2 = torch.tensor(bby2)

    return bbx1, bby1, bbx2, bby2

def cutmix_same_class(images, labels, alpha):
    batch_size = len(images)

    images = images.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    num_classes = len(np.unique(labels))
    
    indices_by_class = [np.where(labels == c)[0] for c in range(num_classes)]
    class_indices = [c_indices for c_indices in indices_by_class if len(c_indices) > 1]
    class_indices = [np.random.permutation(c_indices) for c_indices in class_indices]

    lam = np.random.beta(alpha, alpha)
    cut_rat = np.sqrt(1.0 - lam)

    image_h, image_w, _ = images.shape[1:]  # Assuming image shape in (height, width, channels)

    mixed_images = images.copy()
    mixed_labels = labels.copy()

    for c_indices in class_indices:
        shuffled_indices = np.roll(c_indices, random.randint(1, len(c_indices) - 1))
        indices_pairs = zip(c_indices, shuffled_indices)

        for idx1, idx2 in indices_pairs:
            image1 = images[idx1]
            image2 = images[idx2]

            cx = np.random.randint(0, image_w)
            cy = np.random.randint(0, image_h)

            bbx1 = np.clip(int(cx - image_w * cut_rat / 2), 0, image_w)
            bby1 = np.clip(int(cy - image_h * cut_rat / 2), 0, image_h)
            bbx2 = np.clip(int(cx + image_w * cut_rat / 2), 0, image_w)
            bby2 = np.clip(int(cy + image_h * cut_rat / 2), 0, image_h)

            mixed_images[idx1, bby1:bby2, bbx1:bbx2, :] = image2[bby1:bby2, bbx1:bbx2, :]

            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (image_h * image_w))
            mixed_labels[idx1] = lam * labels[idx1] + (1.0 - lam) * labels[idx2]

    return torch.tensor(mixed_images), torch.tensor(mixed_labels)

# %%
def show_batch_cutmix_images(dataloader):
    for images,labels in dataloader:
        images = images.to(CFG.device)
        labels = labels.to(CFG.device)
        images, labels = cutmix_same_class(images, labels, CFG.cutmix_beta)

        fig,ax = plt.subplots(figsize = (16,12))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images,nrow=16).permute(1,2,0))
        break

# %%
def train(trainloader, validloader, optimizer, criterion1, criterion2, scheduler, model, num_epochs = 10):
    
    best_acc = 0.
    for epoch in range(num_epochs):
        print("")
        model.train()
        train_loss, train_bird_acc = train_epoch(trainloader, model, criterion1, criterion2, optimizer)
        print(f"Epoch {epoch}/{num_epochs}, Train Loss: {train_loss:.3f}, Train Bird Acc: {train_bird_acc:.3f}") #, LR: {scheduler.get_lr()}")
        
        with torch.no_grad():    
            valid_loss, valid_bird_acc = evaluate_epoch(validloader, criterion1, criterion2, model)     
            print(f"Epoch {epoch}/{num_epochs}, Valid Loss: {valid_loss:.3f}, Valid Bird Acc: {valid_bird_acc:.3f}")
            # save model
            if best_acc <= valid_bird_acc:
                print("Saving...")
                best_acc = valid_bird_acc
                torch.save(model.state_dict(), f"{CFG.save_folder}/{epoch}-{best_acc:.3f}-cutmix_{CFG.cutmix}.pth")
        
            scheduler.step()
    
    return model

# %%
from sklearn.metrics import precision_score, recall_score, f1_score

def calculate_metrics(pred, target, threshold=0.5):
    pred = np.array(pred > threshold, dtype=float)
    return {'micro/precision': precision_score(y_true=target, y_pred=pred, average='micro'),
            'micro/recall': recall_score(y_true=target, y_pred=pred, average='micro'),
            'micro/f1': f1_score(y_true=target, y_pred=pred, average='micro'),
            'macro/precision': precision_score(y_true=target, y_pred=pred, average='macro'),
            'macro/recall': recall_score(y_true=target, y_pred=pred, average='macro'),
            'macro/f1': f1_score(y_true=target, y_pred=pred, average='macro'),
            'samples/precision': precision_score(y_true=target, y_pred=pred, average='samples'),
            'samples/recall': recall_score(y_true=target, y_pred=pred, average='samples'),
            'samples/f1': f1_score(y_true=target, y_pred=pred, average='samples'),
            }
# %%
def train_epoch(trainloader, model, criterion1, criterion2, optimizer):
    model.train()
    losses = []
    bird_accs = []
    
    habitat_results = []
    habitat_targets = []
    
    for inputs, bird_labels, habitat_labels in tqdm(trainloader):
        inputs = inputs.to(CFG.device)

        bird_labels = bird_labels.to(CFG.device)
        habitat_labels = habitat_labels.to(CFG.device)

        # zero the parameter gradients
        optimizer.zero_grad()
        bird_outputs, habitat_outputs = model(inputs)

        habitat_results.extend(habitat_outputs.detach().cpu().numpy())
        habitat_targets.extend(habitat_labels.squeeze().detach().cpu().numpy())

        _, bird_preds = torch.max(bird_outputs, 1)

        loss1 = criterion1(bird_outputs, bird_labels) 
        # loss2 = criterion2(habitat_outputs.float().squeeze(), habitat_labels.float().squeeze())
        loss2 = criterion2(habitat_outputs.squeeze(), habitat_labels.squeeze())
        loss = CFG.alpha*loss1+(1-CFG.alpha)*loss2

        loss.backward()
        optimizer.step()

        # statistics
        losses.append(loss.item())
        bird_accs.append((torch.sum(bird_preds == bird_labels.data)/CFG.batch_size).detach().cpu().numpy())

    # result = calculate_metrics(np.array(habitat_results), np.array(habitat_targets))
    # print(result)
    return np.mean(losses), np.mean(bird_accs)

# %%
def evaluate_epoch(validloader, criterion1, criterion2, model):
    model.eval()
    losses = []
    bird_accs = []

    habitat_results = []
    habitat_targets = []
    for inputs, bird_labels, habitat_labels in tqdm(validloader):
        inputs = inputs.to(CFG.device)
        
        bird_outputs, habitat_outputs = model(inputs)

        bird_outputs = bird_outputs.detach().cpu() 
        habitat_outputs = habitat_outputs.detach().cpu() 
        
        habitat_results.extend(habitat_outputs.numpy())
        habitat_targets.extend(habitat_labels.squeeze().cpu().numpy())

        _, bird_preds = torch.max(bird_outputs, 1)

        criterion1 = criterion1.to('cpu')
        criterion2 = criterion2.to('cpu')
        loss1 = criterion1(bird_outputs, bird_labels) 
        # loss2 = criterion2(habitat_outputs, habitat_labels.float().squeeze())
        loss2 = criterion2(habitat_outputs, habitat_labels.squeeze())
        loss = CFG.alpha*loss1+(1-CFG.alpha)*loss2
        criterion1.to(CFG.device)
        criterion2.to(CFG.device)

        # statistics
        losses.append(loss.item())
        bird_accs.append(torch.sum(bird_preds == bird_labels.data)/CFG.batch_size)

    # result = calculate_metrics(np.array(habitat_results), np.array(habitat_targets))
    # print(result)

    return np.mean(losses), np.mean(bird_accs)

# %%
"""
Define Focal-Loss
"""
class FocalLoss(nn.Module):
    """
    The focal loss for fighting against class-imbalance
    """
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = 1e-12  # prevent training from Nan-loss error
        self.cls_weights = torch.tensor([CFG.class_weights],dtype=torch.float, requires_grad=False, device=CFG.device)

    def forward(self, logits, target):
        """
        logits & target should be tensors with shape [batch_size, num_classes]
        """
        probs = torch.sigmoid(logits)
        one_subtract_probs = 1.0 - probs
        # add epsilon
        probs_new = probs + self.epsilon
        one_subtract_probs_new = one_subtract_probs + self.epsilon
        # calculate focal loss
        print(target.shape, probs_new.shape)
        log_pt = target * torch.log(probs_new) + (1.0 - target) * torch.log(one_subtract_probs_new)
        pt = torch.exp(log_pt)
        focal_loss = -1.0 * (self.alpha * (1 - pt) ** self.gamma) * log_pt
        focal_loss = focal_loss * self.cls_weights
        return torch.mean(focal_loss)#, probs
# %%
model = MultiTaskModel_3()
model_params = model.parameters()
model.to(CFG.device)

criterion1 =  nn.CrossEntropyLoss()#LabelSmoothingCrossEntropy()
criterion2 =  FocalLoss()#nn.CrossEntropyLoss()#LabelSmoothingCrossEntropy() # nn.BCELoss()
# criterion = nn.CrossEntropyLoss(weight = torch.FloatTensor(CFG.class_weights).to(CFG.device))
# criterion = FocalLoss()
criterion1 = criterion1.to(CFG.device)
criterion2 = criterion2.to(CFG.device)

optimizer = optim.Adam(model_params, lr=CFG.lr)

exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.97)

model_ft = train(train_loader, val_loader, optimizer, criterion1, criterion2, exp_lr_scheduler, model, num_epochs=CFG.epochs)
with torch.no_grad():    
    test_loss, test_bird_acc = evaluate_epoch(test_loader, criterion1, criterion2, model_ft)   
    print(f"Test Loss: {test_loss}, Test Bird Acc: {test_bird_acc}")