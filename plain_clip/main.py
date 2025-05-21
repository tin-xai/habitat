from load import *
import torchmetrics
from tqdm import tqdm
import numpy as np
import warnings
import argparse
from termcolor import colored

warnings.filterwarnings('ignore')

#%%
parser = argparse.ArgumentParser()
### Base arguments.
parser.add_argument('--mode', type=str, default='clip', choices=METHODS,
                    help='VLM extension to use.')
parser.add_argument('--dataset', type=str, default='cub', choices=DATASETS, 
                    help='Dataset to evaluate on.')
parser.add_argument('--model_size', type=str, default='ViT-B/32', choices=BACKBONES, 
                    help='Pretrained CLIP model to use.')

parser.add_argument('--verbose', action='store_true', help='Verbose output')

parser.add_argument('--device', type=str, default='cuda:0', help='Cuda to use.')
parser.add_argument('--seed', type=int, default=1, 
                    help='Replication seed.')
parser.add_argument('--seeds', nargs='+', type=int, default=[1],
                    help='List of seeds.')

parser.add_argument('--batch_size', type=int, default=640, 
                    help='Batchsize, mainly used to compute image embeddings.')
parser.add_argument('--image_size', type=int, default=224, 
                    help='Image size.')

parser.add_argument('--aggregate', type=str, default='mean', choices=('mean', 'max'), 
                    help='How to aggregate similarites of multiple language embeddings.')

parser.add_argument('--before_text', type=str, default="", help='Text before labels')
parser.add_argument('--label_before_text', type=str, default='', 
                    help='Prompt-part going at the very beginning.')
parser.add_argument('--between_text', type=str, default=', ', help='Text between labels')
parser.add_argument('--after_text', type=str, default='', help='Text after labels')
parser.add_argument('--label_after_text', type=str, default='', 
                    help='Prompt-part going at the very end.')

parser.add_argument('--unmodify', action='store_true', help='Unmodify setting')
###
parser.add_argument('--pre_descriptor_text', type=str, default='', 
                    help='Text that goes right before the descriptor.')
parser.add_argument('--descriptor_separator', type=str, default=', ', 
                    help='Text separating descriptor part and classname.')

parser.add_argument('--descriptor_fname', default=None, type=str, help='Descriptor filename')
parser.add_argument('--descriptor_fnames', nargs='+', default=None, help='List of Descriptor filenames')

###
parser.add_argument('--compute_support_images_embedding', action='store_true', help='Flag. If set, use compute suppport images embedding.')
parser.add_argument('--use_support_images_embedding', action='store_true', help='Flag. If set, use suppport images.')
parser.add_argument('--num_support_images', type=int, default=30,
                    help='The number of support images used in CLIP evaluation')
###
parser.add_argument('--category_name_inclusion', type=str, default='prepend', choices=['append', 'prepend'], help='How to include category names')
parser.add_argument('--dont_apply_descriptor_modification', action='store_true',
                    help='Flag. If set, will not use "which (is/has/etc)" before descriptors.')

parser.add_argument('--randomization_budget', type=int, default=15,
                    help='Budget w.r.t. to DCLIP for randomization ablations')
parser.add_argument('--waffle_count', type=int, default=15,
                    help='For WaffleCLIP: Number of randomized descriptor pairs to use')

parser.add_argument('--reps', type=int, default=1, 
                    help='Number of repetitions to run a method for with changing randomization. Default value should be >7 for WaffleCLIP variants.')

parser.add_argument('--savename', type=str, default='results',
                    help='Name of csv-file in which results are stored.')
parser.add_argument('--save_class_acc', action='store_true',
                    help='Flag. If set, will save class accuracy to the result file csv.')


### for INat dataset, use common names or sci names
parser.add_argument('--sci2comm', action='store_true',
                    help='Flag. If set, will convert sci 2 common names.')

###
opt = parser.parse_args()
opt.apply_descriptor_modification = not opt.dont_apply_descriptor_modification


#%% Get dataloader and load model.
if opt.descriptor_fname is None and opt.descriptor_fnames is None:
    print(colored("Must provide a description file(s)", "red", attrs=["bold", "italic"]))

seeds = opt.seeds
if opt.descriptor_fname:
    opt.descriptor_fnames = [opt.descriptor_fname]

for seed in seeds:
    opt.seed = seed
    seed_everything(opt.seed)
    
    opt.device = device = torch.device(opt.device)
    # load model
    model, preprocess = clip.load(opt.model_size, device=device, jit=False)
    model.eval()
    model.requires_grad_(False)

    for descriptor_fname in opt.descriptor_fnames:
        opt.descriptor_fname = descriptor_fname
        # set up dataset and hyperparameters
        opt, dataset = setup(opt)
        bs = opt.batch_size
        dataloader = DataLoader(dataset, bs, shuffle=True, num_workers=16, pin_memory=True)

        print(colored(f"\nLoading model [{opt.model_size}] for dataset [{opt.dataset}] ...\n", "yellow", attrs=["bold"]))

        print("Encoding descriptions...")
        description_encodings = compute_description_encodings(opt, model)
        label_encodings = compute_label_encodings(opt, model)
        num_classes = opt.num_classes

        print("Evaluating...")
        lang_accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)
        lang_accuracy_metric_top5 = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=5).to(device)

        clip_accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)
        clip_accuracy_metric_top5 = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=5).to(device)

        confmat = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes).to(device)

        wrongly_predicted_paths = []

        for batch_number, batch in enumerate(tqdm(dataloader)):
            if len(batch) == 3:
                images, labels, path = batch
            else:
                images, labels = batch

            images = images.to(device)
            labels = labels.to(device)
            
            image_encodings = model.encode_image(images)
            image_encodings = F.normalize(image_encodings)
            
            image_labels_similarity = image_encodings @ label_encodings.T
            
            clip_predictions = image_labels_similarity.argmax(dim=1)

            clip_acc = clip_accuracy_metric(image_labels_similarity, labels)
            clip_acc_top5 = clip_accuracy_metric_top5(image_labels_similarity, labels)
            
            
            image_description_similarity = [None]*num_classes
            image_description_similarity_cumulative = [None]*num_classes
            
            for i, (k, v) in enumerate(description_encodings.items()):
                        
                dot_product_matrix = image_encodings @ v.T

                image_description_similarity[i] = dot_product_matrix
                image_description_similarity_cumulative[i] = aggregate_similarity(image_description_similarity[i], aggregation_method=opt.aggregate)
            # create tensor of similarity means
            cumulative_tensor = torch.stack(image_description_similarity_cumulative,dim=1)
            
            descr_predictions = cumulative_tensor.argmax(dim=1)

            lang_acc = lang_accuracy_metric(cumulative_tensor.softmax(dim=-1), labels)
            lang_acc_top5 = lang_accuracy_metric_top5(cumulative_tensor.softmax(dim=-1), labels)
            
            confmat(cumulative_tensor.softmax(dim=-1).argmax(dim=-1), labels)

        # Compute the final confusion matrix
        final_conf_matrix = confmat.compute()

        # Calculate class-wise accuracies
        class_accuracies = final_conf_matrix.diag() / final_conf_matrix.sum(1)

        # Handle cases where a class never appears in the batch (to avoid division by zero)
        # class_accuracies[torch.isnan(class_accuracies)] = 0

        accuracy_logs = {}
        accuracy_logs["description-based: "] = 100*lang_accuracy_metric.compute().item()
        # accuracy_logs["Total Description-based Top-5 Accuracy: "] = 100*lang_accuracy_metric_top5.compute().item()

        accuracy_logs["CLIP: "] = 100*clip_accuracy_metric.compute().item()
        # accuracy_logs["Total CLIP-Standard Top-5 Accuracy: "] = 100*clip_accuracy_metric_top5.compute().item()

        # print the dictionary
        colors =['red', 'red', 'white', 'white']
        for i, (key, value) in enumerate(accuracy_logs.items()):
            print(colored(f"{key} {value}", colors[i], attrs=["bold"]))


        ### Save results as csv.
        import csv
        os.makedirs('results', exist_ok=True)
        # Open the file in write mode
        with open(f'results/{opt.savename}.csv', 'a', newline='') as file:
            # Create a writer object
            writer = csv.writer(file)
            
            # Optionally write headers
            if opt.save_class_acc:
                if opt.use_support_images_embedding and opt.num_support_images > 0:
                    writer.writerow(["Mode", "Descriptor Path", "Model Size", "Metric", "Value", "Num Support Images"] + [str(i) for i in range(num_classes)])
                else:
                    writer.writerow(["Mode", "Descriptor Path", "Model Size", "Metric", "Value"] + [str(i) for i in range(num_classes)])
            else:
                if opt.use_support_images_embedding and opt.num_support_images > 0:
                    writer.writerow(["Mode", "Descriptor Path", "Model Size", "Metric", "Value", "Num Support Images"])
                else:
                    writer.writerow(["Mode", "Descriptor Path", "Model Size", "Metric", "Value"])

            # Write accuracy logs to CSV
            for key, value in accuracy_logs.items():
                if opt.save_class_acc:
                    if opt.use_support_images_embedding and opt.num_support_images > 0:
                        writer.writerow([opt.mode, opt.descriptor_fname, opt.model_size, key, round(value, 2), opt.num_support_images] + class_accuracies.tolist())
                    else:
                        writer.writerow([opt.mode, opt.descriptor_fname, opt.model_size, key, round(value, 2)] + class_accuracies.tolist())
                else:
                    if opt.use_support_images_embedding and opt.num_support_images > 0:
                        writer.writerow([opt.mode, opt.descriptor_fname, opt.model_size, key, round(value, 2), opt.num_support_images])
                    else:
                        writer.writerow([opt.mode, opt.descriptor_fname, opt.model_size, key, round(value, 2)])

        print(f"Accuracy logs saved to {opt.savename}")