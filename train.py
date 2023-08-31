from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
from byol import id2label, label2id, byol_query_key_func, train_dataset, train_dataloader, valid_dataset, valid_dataloader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from tqdm.notebook import tqdm
from torch import nn
from transformers import AdamW
import numpy as np

model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b5", ignore_mismatched_sizes=True,
                                                         num_labels=len(id2label), id2label=id2label, label2id=label2id,
                                                         reshape_last_stage=True)


optimizer = AdamW(model.parameters(), lr=0.00006)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cuda" if torch.cuda.is_available() else "mps") - make use of this line of code if training on MacBook to utilise GPU 
model.to(device)



def mIoU(pred_mask, mask, smooth=1e-10, n_classes=23):
    '''
    Defining mean intersection over union - metric used across the board for all segmentation tasks in cityscapes performances
    https://www.kaggle.com/code/ligtfeather/semantic-segmentation-is-easy-with-pytorch
    '''
    with torch.no_grad():
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        iou_per_class = []
        for clas in range(0, n_classes): #loop per pixel class
            true_class = pred_mask == clas
            true_label = mask == clas

            if true_label.long().sum().item() == 0: #no exist label in this loop
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect + smooth) / (union +smooth)
                iou_per_class.append(iou)
        return np.nanmean(iou_per_class)
    


def reshape_for_contrastive(tensor):
    # Reshape the tensor to have shape (batch_size * num_channels * height * width, embedding_dim)
    return tensor.view(tensor.size(0), -1)


def contrastive_loss_func(query, key, temperature=0.1):
    query = F.normalize(query, dim=1)
    key = F.normalize(key, dim=1)

    similarity = torch.matmul(query, key.transpose(0, 1)) / temperature

    # simple cosine similarity (different from logit cross-entropy as seen in moco)
    # this is because we are not utilising a queue here for more computational efficiency
    mask = torch.eye(similarity.size(0), device=similarity.device)
    positive_pair_similarity = similarity.masked_select(mask.bool()).view(-1)
    negative_pair_similarity = similarity.masked_select(~mask.bool()).view(similarity.size(0), -1)

    contrastive_loss = -torch.log(positive_pair_similarity.exp() / negative_pair_similarity.exp().sum(dim=1)).mean()

    return contrastive_loss



for epoch in range(1, 11):  # loop over the dataset multiple times
    print("Epoch:", epoch)
    pbar = tqdm(train_dataloader)
    accuracies = []
    losses = []
    contrastive_losses = []  
    anomaly_scores_train = []  
    anomaly_scores_val = []  
    mious = []  # List to store mIoU values
    val_accuracies = []
    val_losses = []
    val_mious =[]
    val_contrastive_losses = []
    
    model.train()

    # save the train embeddings to a list 
    # train embeddings consist of the query and key embeddings of the logits
    training_embeddings = []
    
    for idx, batch in enumerate(pbar):
        
        pixel_values = batch['encoded_inputs']["pixel_values"].to(device)
        labels = batch['encoded_inputs']["labels"].to(device)
        
        optimizer.zero_grad()

        # predict logits and outputs from the model
        outputs = model(pixel_values=pixel_values, labels=labels)
        
        # collect the intermediate layers of the logits and apply the BYOL function
        # this creates query and key embeddings strictly from the intermediate layers of the logits
        # Done implicitly - query and key embedding creation updates accordingly becauase of the joint loss 
        querkey = byol_query_key_func(outputs['logits'][0][0:3], 512, 0.08)
        query = reshape_for_contrastive(querkey['query']).to(device) 
        key = reshape_for_contrastive(querkey['key']).to(device)

        # save the query and key embeddings to a list 
        training_embeddings.append((query, key))

        # obtain a contrastive loss value for the batch
        contrastive_loss = contrastive_loss_func(query, key, temperature=0.2)
        contrastive_losses.append(contrastive_loss.item())
        

        anomaly_score = contrastive_loss.item()
        anomaly_scores_train.append(anomaly_score)

        upsampled_logits = nn.functional.interpolate(outputs.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        predicted = upsampled_logits.argmax(dim=1)

        mask = (labels != 255)
        pred_labels = predicted[mask].detach().cpu().numpy()
        true_labels = labels[mask].detach().cpu().numpy()
        accuracy = accuracy_score(pred_labels, true_labels)
        iou = mIoU(upsampled_logits, labels, n_classes=23)
        mious.append(iou)

        loss = outputs.loss
        accuracies.append(accuracy)
        losses.append(loss.item())


        # combine the contrastive and segmentation loss
        total_loss = loss + contrastive_loss

        pbar.set_postfix({'Batch': idx,
                          'Pixel-wise accuracy': sum(accuracies) / len(accuracies),
                          'Loss': sum(losses) / len(losses),
                          'mIoU': sum(mious) / len(mious),
                          'Contrastive Loss': contrastive_loss.item()}) 

        total_loss.backward()
        optimizer.step()


    # repeat the above with the valdiation set 
    else:
        model.eval()
        with torch.no_grad():
            for idx, batch in enumerate(valid_dataloader):
                
                pixel_values = batch['encoded_inputs']["pixel_values"].to(device)
                labels = batch['encoded_inputs']["labels"].to(device)

                outputs = model(pixel_values=pixel_values, labels=labels)
                querkey = byol_query_key_func(outputs['logits'][0][0:3], 512, 0.08)
                query = reshape_for_contrastive(querkey['query']).to(device) 
                key = reshape_for_contrastive(querkey['key']).to(device)

                contrastive_loss = contrastive_loss_func(query, key, temperature=0.2)

                upsampled_logits = nn.functional.interpolate(outputs.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
                predicted = upsampled_logits.argmax(dim=1)

                mask = (labels != 255)
                pred_labels = predicted[mask].detach().cpu().numpy()
                true_labels = labels[mask].detach().cpu().numpy()
                accuracy = accuracy_score(pred_labels, true_labels)
                val_iou = mIoU(upsampled_logits, labels, n_classes=23)
                val_mious.append(val_iou)  # Store validation mIoU value
                anomaly_score = contrastive_loss.item()

                # Store validation results
                anomaly_scores_val.append((anomaly_score)) 
                val_loss = outputs.loss
                val_accuracies.append(accuracy)
                val_losses.append(val_loss.item())
                val_contrastive_losses.append(contrastive_loss.item())
                

    print(f"Train Pixel-wise accuracy: {sum(accuracies)/len(accuracies)}\
         Train Loss: {sum(losses)/len(losses)}\
         Train mIoU: {sum(mious)/len(mious)}\
         Train Contrastive Loss: {sum(contrastive_losses)/len(contrastive_losses)}\
         Val Pixel-wise accuracy: {sum(val_accuracies)/len(val_accuracies)}\
         Val Loss: {sum(val_losses)/len(val_losses)}\
         Val mIoU: {sum(val_mious)/len(val_mious)}\
         Batch contrastive loss (Training): {anomaly_scores_train}\
         Batch Contrastive loss (Validation): {anomaly_scores_val}")

# saving the weights to a directory 
torch.save(model.state_dict(), 'trained_model_no_cl.pth')


# saving the training embeddings 
for idx, tensor_tuple in enumerate(training_embeddings):
    tensor_list = list(tensor_tuple)  # Convert the tuple to a list
    numpy_arrays = [tensor.numpy() for tensor in tensor_list]
    combined_array = np.concatenate(numpy_arrays)  # Combine arrays if needed
    
    filename = f"train_embeddings_tensor_{idx}.txt"
    np.savetxt(filename, combined_array)


# reimporting the training embeddings for inference
#training_embeddings = np.loadtxt('train_embeddings_tensor.txt', delimiter=' ')
