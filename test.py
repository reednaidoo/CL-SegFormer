'''
Using the trained model, evaluate it on the test dataloader. 
To effectively create the test dataloader, navigate to the byol script 
'''


model.eval()

# Initialize lists to store results
test_pixel_accuracies = []
test_query_embeddings = []
test_key_embeddings = []

# Iterate through the test dataloader
with torch.no_grad():
    for idx, batch in enumerate(test_dataloader):
        # Move data to the device
        pixel_values = batch['encoded_inputs']["pixel_values"].to(device)
        labels = batch['encoded_inputs']["labels"].to(device)

        # Perform segmentation inference
        outputs = model(pixel_values=pixel_values, labels=labels)
        upsampled_logits = nn.functional.interpolate(outputs.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        predicted = upsampled_logits.argmax(dim=1)

        #querkey = byol_query_key_func(outputs['logits'][0][0:3], 512, 0.08)
        query = reshape_for_contrastive(querkey['query']).to(device) 
        key = reshape_for_contrastive(querkey['key']).to(device)

        mask = (labels != 255)
        pred_labels = predicted[mask].detach().cpu().numpy()
        true_labels = labels[mask].detach().cpu().numpy()
        accuracy = accuracy_score(pred_labels, true_labels)
        
        # Store results
        test_pixel_accuracies.append(accuracy)
        test_query_embeddings.append(query.cpu().numpy())
        test_key_embeddings.append(key.cpu().numpy())