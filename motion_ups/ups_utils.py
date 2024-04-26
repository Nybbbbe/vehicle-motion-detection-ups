import torch
import torch.nn.functional as F
from torchvision.io import read_image
import json
from tqdm import tqdm
from torch.utils.data import random_split
import pandas as pd
import matplotlib.pyplot as plt
from torchvision.models.resnet import ResNet50_Weights
from sklearn.model_selection import train_test_split
import numpy as np

PRINT_FREQ = 2 # Print every x mini-batches

def train_regular(max_epochs, model, labeled_dataloader, pseudo_labeled_dataloader, test_dataloader, optimizer, device, save_dir):
    train_metrics = []
    val_metrics = []
    best_score = 0
    epochs_since_best_score = 0

    for epoch in range(max_epochs):
        model.train()  # Set the model to training mode
        train_running_loss = 0.0
        TP = 0
        FP = 0
        FN = 0
        TN = 0

        loop = tqdm(labeled_dataloader, leave=True)
        for i, (images, spectrograms, labels) in enumerate(loop):     
            # Zero the parameter gradients
            optimizer.zero_grad()

            image_0 = images[0].to(device)
            image_1 = images[1].to(device)
            image_2 = images[2].to(device)
            image_3 = images[3].to(device)
            image_4 = images[4].to(device)

            spec_0 = spectrograms[0].to(device)
            spec_1 = spectrograms[1].to(device)
            spec_2 = spectrograms[2].to(device)
            spec_3 = spectrograms[3].to(device)
            spec_4 = spectrograms[4].to(device)
            spec_5 = spectrograms[5].to(device)
            spec_6 = spectrograms[6].to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(
                image_0, image_1, image_2, image_3, image_4,
                spec_0, spec_1, spec_2, spec_3, spec_4, spec_5, spec_6
            )

            # Convert labels and predictions to boolean values if they're not already
            labels_bool = labels.bool()
            _, predicted_classes = torch.max(outputs, 1)

            TP += ((predicted_classes == 1) & (labels_bool == 1)).sum().item()
            FP += ((predicted_classes == 1) & (labels_bool == 0)).sum().item()
            FN += ((predicted_classes == 0) & (labels_bool == 1)).sum().item()
            TN += ((predicted_classes == 0) & (labels_bool == 0)).sum().item()

            #loss = criterion(outputs, labels)
            loss = F.cross_entropy(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Print statistics
            train_running_loss += loss.item()
            if i % PRINT_FREQ == PRINT_FREQ - 1:
                accuracy = (TP + TN) / (TP + FP + FN + TN)
                recall = TP / (TP + FN) if (TP + FN) > 0 else 0
                iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
                
                print(f"[Epoch: {epoch + 1}, Batch: {i + 1}] Loss: {train_running_loss / 10:.4f}, "
                    f"Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, IoU: {iou:.4f}")
                
                metric = {
                    "loss": train_running_loss / 10,
                    "accuracy": accuracy,
                    "recall": recall,
                    "iou": iou,
                    "TP": TP,
                    "FP": FP,
                    "TN": TN,
                    "FN": FN,
                }

                train_metrics.append(metric)

                train_running_loss = 0.0
                TP = 0
                FP = 0
                FN = 0
                TN = 0

                with open(f'{save_dir}/training_metrics.json', 'w') as f:
                    json.dump(train_metrics, f, indent=4)

        loop = tqdm(pseudo_labeled_dataloader, leave=True)
        for i, (images, spectrograms, labels) in enumerate(loop):     
            # Zero the parameter gradients
            optimizer.zero_grad()

            image_0 = images[0].to(device)
            image_1 = images[1].to(device)
            image_2 = images[2].to(device)
            image_3 = images[3].to(device)
            image_4 = images[4].to(device)

            spec_0 = spectrograms[0].to(device)
            spec_1 = spectrograms[1].to(device)
            spec_2 = spectrograms[2].to(device)
            spec_3 = spectrograms[3].to(device)
            spec_4 = spectrograms[4].to(device)
            spec_5 = spectrograms[5].to(device)
            spec_6 = spectrograms[6].to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(
                image_0, image_1, image_2, image_3, image_4,
                spec_0, spec_1, spec_2, spec_3, spec_4, spec_5, spec_6
            )

            # Convert labels and predictions to boolean values if they're not already
            labels_bool = labels.bool()
            _, predicted_classes = torch.max(outputs, 1)

            TP += ((predicted_classes == 1) & (labels_bool == 1)).sum().item()
            FP += ((predicted_classes == 1) & (labels_bool == 0)).sum().item()
            FN += ((predicted_classes == 0) & (labels_bool == 1)).sum().item()
            TN += ((predicted_classes == 0) & (labels_bool == 0)).sum().item()

            #loss = criterion(outputs, labels)
            loss = F.cross_entropy(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Print statistics
            train_running_loss += loss.item()
            if i % PRINT_FREQ == PRINT_FREQ - 1:
                accuracy = (TP + TN) / (TP + FP + FN + TN)
                recall = TP / (TP + FN) if (TP + FN) > 0 else 0
                iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
                
                print(f"[Epoch: {epoch + 1}, Batch: {i + 1}] Loss: {train_running_loss / 10:.4f}, "
                    f"Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, IoU: {iou:.4f}")
                
                metric = {
                    "loss": train_running_loss / 10,
                    "accuracy": accuracy,
                    "recall": recall,
                    "iou": iou,
                    "TP": TP,
                    "FP": FP,
                    "TN": TN,
                    "FN": FN,
                }

                train_metrics.append(metric)

                train_running_loss = 0.0
                TP = 0
                FP = 0
                FN = 0
                TN = 0

                with open(f'{save_dir}/training_metrics.json', 'w') as f:
                    json.dump(train_metrics, f, indent=4)
        
        with torch.no_grad():
            model.eval()  # Set the model to training mode
            val_loss = 0.0
            val_TP = 0
            val_FP = 0
            val_FN = 0
            val_TN = 0

            loop = tqdm(test_dataloader, leave=True)
            k = 0
            for i, (images, spectrograms, labels) in enumerate(loop):     
                image_0 = images[0].to(device)
                image_1 = images[1].to(device)
                image_2 = images[2].to(device)
                image_3 = images[3].to(device)
                image_4 = images[4].to(device)

                spec_0 = spectrograms[0].to(device)
                spec_1 = spectrograms[1].to(device)
                spec_2 = spectrograms[2].to(device)
                spec_3 = spectrograms[3].to(device)
                spec_4 = spectrograms[4].to(device)
                spec_5 = spectrograms[5].to(device)
                spec_6 = spectrograms[6].to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(
                    image_0, image_1, image_2, image_3, image_4,
                    spec_0, spec_1, spec_2, spec_3, spec_4, spec_5, spec_6
                )

                loss = F.cross_entropy(outputs, labels)
                val_loss += loss.item()
                k += 1

                # Convert labels and predictions to boolean values if they're not already
                labels_bool = labels.bool()
                _, predicted_classes = torch.max(outputs, 1)

                val_TP += ((predicted_classes == 1) & (labels_bool == 1)).sum().item()
                val_FP += ((predicted_classes == 1) & (labels_bool == 0)).sum().item()
                val_FN += ((predicted_classes == 0) & (labels_bool == 1)).sum().item()
                val_TN += ((predicted_classes == 0) & (labels_bool == 0)).sum().item()
            
            accuracy = (val_TP + val_TN) / (val_TP + val_FP + val_FN + val_TN)
            recall = val_TP / (val_TP + val_FN) if (val_TP + val_FN) > 0 else 0
            iou = val_TP / (val_TP + val_FP + val_FN) if (val_TP + val_FP + val_FN) > 0 else 0
            val_loss = val_loss / k

            metric = {
                "loss": val_loss,
                "accuracy": accuracy,
                "recall": recall,
                "iou": iou,
                "TP": val_TP,
                "FP": val_FP,
                "TN": val_TN,
                "FN": val_FN,
            }

            val_metrics.append(metric)

            with open(f'{save_dir}/val_metrics.json', 'w') as f:
                json.dump(val_metrics, f, indent=4)

            if (iou > best_score):
                best_score = iou
                torch.save(model, f'{save_dir}/best_model.pth')
                epochs_since_best_score = 0
            else:
                epochs_since_best_score += 1

            print(f"[Val Epoch: {epoch + 1}] Loss: {val_loss:.4f}, "
                    f"Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, IoU: {iou:.4f}")
            
        if (epochs_since_best_score > 5):
            break

    print("Finished Training Regular")

def train_initial(max_epochs, model, labeled_dataloader, test_dataloader, optimizer, device, save_dir):
    train_metrics = []
    val_metrics = []
    best_score = 0
    epochs_since_best_score = 0

    for epoch in range(max_epochs):
        model.train()  # Set the model to training mode
        train_running_loss = 0.0
        TP = 0
        FP = 0
        FN = 0
        TN = 0

        loop = tqdm(labeled_dataloader, leave=True)
        for i, (images, spectrograms, labels) in enumerate(loop):     
            # Zero the parameter gradients
            optimizer.zero_grad()

            image_0 = images[0].to(device)
            image_1 = images[1].to(device)
            image_2 = images[2].to(device)
            image_3 = images[3].to(device)
            image_4 = images[4].to(device)

            spec_0 = spectrograms[0].to(device)
            spec_1 = spectrograms[1].to(device)
            spec_2 = spectrograms[2].to(device)
            spec_3 = spectrograms[3].to(device)
            spec_4 = spectrograms[4].to(device)
            spec_5 = spectrograms[5].to(device)
            spec_6 = spectrograms[6].to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(
                image_0, image_1, image_2, image_3, image_4,
                spec_0, spec_1, spec_2, spec_3, spec_4, spec_5, spec_6
            )

            # Convert labels and predictions to boolean values if they're not already
            labels_bool = labels.bool()
            _, predicted_classes = torch.max(outputs, 1)

            TP += ((predicted_classes == 1) & (labels_bool == 1)).sum().item()
            FP += ((predicted_classes == 1) & (labels_bool == 0)).sum().item()
            FN += ((predicted_classes == 0) & (labels_bool == 1)).sum().item()
            TN += ((predicted_classes == 0) & (labels_bool == 0)).sum().item()

            #loss = criterion(outputs, labels)
            loss = F.cross_entropy(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Print statistics
            train_running_loss += loss.item()
            if i % PRINT_FREQ == PRINT_FREQ - 1:
                accuracy = (TP + TN) / (TP + FP + FN + TN)
                recall = TP / (TP + FN) if (TP + FN) > 0 else 0
                iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
                
                print(f"[Epoch: {epoch + 1}, Batch: {i + 1}] Loss: {train_running_loss / 10:.4f}, "
                    f"Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, IoU: {iou:.4f}")
                
                metric = {
                    "loss": train_running_loss / 10,
                    "accuracy": accuracy,
                    "recall": recall,
                    "iou": iou,
                    "TP": TP,
                    "FP": FP,
                    "TN": TN,
                    "FN": FN,
                }

                train_metrics.append(metric)

                train_running_loss = 0.0
                TP = 0
                FP = 0
                FN = 0
                TN = 0

                with open(f'{save_dir}/training_metrics.json', 'w') as f:
                    json.dump(train_metrics, f, indent=4)
        
        with torch.no_grad():
            model.eval()  # Set the model to training mode
            val_loss = 0.0
            val_TP = 0
            val_FP = 0
            val_FN = 0
            val_TN = 0

            loop = tqdm(test_dataloader, leave=True)
            k = 0
            for i, (images, spectrograms, labels) in enumerate(loop):     
                image_0 = images[0].to(device)
                image_1 = images[1].to(device)
                image_2 = images[2].to(device)
                image_3 = images[3].to(device)
                image_4 = images[4].to(device)

                spec_0 = spectrograms[0].to(device)
                spec_1 = spectrograms[1].to(device)
                spec_2 = spectrograms[2].to(device)
                spec_3 = spectrograms[3].to(device)
                spec_4 = spectrograms[4].to(device)
                spec_5 = spectrograms[5].to(device)
                spec_6 = spectrograms[6].to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(
                    image_0, image_1, image_2, image_3, image_4,
                    spec_0, spec_1, spec_2, spec_3, spec_4, spec_5, spec_6
                )

                loss = F.cross_entropy(outputs, labels)
                val_loss += loss.item()
                k += 1

                # Convert labels and predictions to boolean values if they're not already
                labels_bool = labels.bool()
                _, predicted_classes = torch.max(outputs, 1)

                val_TP += ((predicted_classes == 1) & (labels_bool == 1)).sum().item()
                val_FP += ((predicted_classes == 1) & (labels_bool == 0)).sum().item()
                val_FN += ((predicted_classes == 0) & (labels_bool == 1)).sum().item()
                val_TN += ((predicted_classes == 0) & (labels_bool == 0)).sum().item()
            
            accuracy = (val_TP + val_TN) / (val_TP + val_FP + val_FN + val_TN)
            recall = val_TP / (val_TP + val_FN) if (val_TP + val_FN) > 0 else 0
            iou = val_TP / (val_TP + val_FP + val_FN) if (val_TP + val_FP + val_FN) > 0 else 0
            val_loss = val_loss / k

            metric = {
                "loss": val_loss,
                "accuracy": accuracy,
                "recall": recall,
                "iou": iou,
                "TP": val_TP,
                "FP": val_FP,
                "TN": val_TN,
                "FN": val_FN,
            }

            val_metrics.append(metric)

            with open(f'{save_dir}/val_metrics.json', 'w') as f:
                json.dump(val_metrics, f, indent=4)

            if (iou > best_score):
                best_score = iou
                torch.save(model, f'{save_dir}/best_model.pth')
                epochs_since_best_score = 0
            else:
                epochs_since_best_score += 1

            print(f"[Val Epoch: {epoch + 1}] Loss: {val_loss:.4f}, "
                    f"Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, IoU: {iou:.4f}")
            
        if (epochs_since_best_score > 5):
            break

    print("Finished Training Initial")

def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def pseudo_labeling(unlabeled_dataloader, model, device, temp_nl, kappa_n, tau_n,
                    tau_p, kappa_p, no_uncertainty=True):

    pseudo_idx = []
    pseudo_target = []
    pseudo_maxstd = []
    gt_target = []
    idx_list = []
    gt_list = []
    target_list = []

    TP = 0
    FP = 0
    FN = 0
    TN = 0
    total_loss = 0

    loop = tqdm(unlabeled_dataloader, leave=True)
    model.eval()

    if not no_uncertainty:
        f_pass = 10
        enable_dropout(model)
    else:
        f_pass = 1

    k = 0

    with torch.no_grad():
        for i, (images, spectrograms, labels, indexs) in enumerate(loop):     
            image_0 = images[0].to(device)
            image_1 = images[1].to(device)
            image_2 = images[2].to(device)
            image_3 = images[3].to(device)
            image_4 = images[4].to(device)

            spec_0 = spectrograms[0].to(device)
            spec_1 = spectrograms[1].to(device)
            spec_2 = spectrograms[2].to(device)
            spec_3 = spectrograms[3].to(device)
            spec_4 = spectrograms[4].to(device)
            spec_5 = spectrograms[5].to(device)
            spec_6 = spectrograms[6].to(device)
            labels = labels.to(device)

            out_prob = []
            out_prob_nl = []

            for _ in range(f_pass):
                outputs = model(
                    image_0, image_1, image_2, image_3, image_4,
                    spec_0, spec_1, spec_2, spec_3, spec_4, spec_5, spec_6
                )
                out_prob.append(F.softmax(outputs, dim=1))
                out_prob_nl.append(F.softmax(outputs/temp_nl, dim=1))

            out_prob = torch.stack(out_prob)
            out_prob_nl = torch.stack(out_prob_nl)

            # Hack to fix NAN values caused by overconfidence
            # epsilon = 1e-3
            # threshold = 1e-2
            # out_prob = torch.where(out_prob < threshold, out_prob + epsilon, out_prob)
            # out_prob_nl = torch.where(out_prob_nl < threshold, out_prob_nl + epsilon, out_prob_nl)

            out_std = torch.std(out_prob, dim=0, correction=0)
            out_std_nl = torch.std(out_prob_nl, dim=0, correction=0)
            out_prob = torch.mean(out_prob, dim=0)
            out_prob_nl = torch.mean(out_prob_nl, dim=0)
            max_value, max_idx = torch.max(out_prob, dim=1)
            max_std = out_std.gather(1, max_idx.view(-1,1))
            out_std_nl = out_std_nl.cpu().numpy()

            idx_list.extend(indexs.cpu().numpy().tolist())
            gt_list.extend(labels.cpu().numpy().tolist())
            target_list.extend(max_idx.cpu().numpy().tolist())

            #selecting positive pseudo-labels
            if not no_uncertainty:
                selected_idx = (max_value>=tau_p) * (max_std.squeeze(1) < kappa_p)
            else:
                selected_idx = max_value>=tau_p

            selected_idx = selected_idx.cpu()

            pseudo_maxstd.extend(max_std.squeeze(1)[selected_idx].cpu().numpy().tolist())
            pseudo_target.extend(max_idx[selected_idx].cpu().numpy().tolist())
            pseudo_idx.extend(indexs[selected_idx].cpu().numpy().tolist())
            gt_target.extend(labels[selected_idx].cpu().numpy().tolist())

            loss = F.cross_entropy(outputs, labels)

            labels_bool = labels[selected_idx].bool()
            _, predicted_classes = torch.max(outputs[selected_idx], dim=1)

            TP += ((predicted_classes == 1) & (labels_bool == 1)).sum().item()
            FP += ((predicted_classes == 1) & (labels_bool == 0)).sum().item()
            FN += ((predicted_classes == 0) & (labels_bool == 1)).sum().item()
            TN += ((predicted_classes == 0) & (labels_bool == 0)).sum().item()

            total_loss += loss
            k += 1
    
    accuracy = (TP + TN) / (TP + FP + FN + TN)
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
    total_loss = total_loss / k

    print(f"[Pseudo Labeling results] Loss: {total_loss:.4f}, "
        f"Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, IoU: {iou:.4f}")

    pseudo_target = np.array(pseudo_target)
    gt_target = np.array(gt_target)
    pseudo_maxstd = np.array(pseudo_maxstd)
    pseudo_idx = np.array(pseudo_idx)

    pseudo_labeling_acc = (pseudo_target == gt_target)*1
    pseudo_labeling_acc = (sum(pseudo_labeling_acc)/len(pseudo_labeling_acc))*100
    print(f'Pseudo-Labeling Accuracy (positive): {pseudo_labeling_acc}, Total Selected: {len(pseudo_idx)} / {len(idx_list)}')

    pseudo_label_dict = {'psuedo_labeled_indexes': pseudo_idx.tolist(), 'psuedo_labeled_targets':pseudo_target.tolist()}
    return pseudo_label_dict