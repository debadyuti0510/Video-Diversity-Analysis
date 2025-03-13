import datasets
import joblib
import torch
import numpy as np
from torchvision.ops.boxes import batched_nms
from transformers import CLIPProcessor, CLIPModel, CLIPVisionModel
from sklearn.linear_model import LogisticRegression
from tqdm.auto import tqdm
from PIL import Image

# Set manual torch seed
torch.manual_seed(42)


# Set device to use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and preprocessor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
model.to(device)
vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
vision_model.to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Add age prompts for clip to look age
# Remove 0-2 and 3-9 from the list of classes
age_brackets = ["10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "more than 70"]
age_texts = [f"A person in the {c} age group" for c in age_brackets ] 
def get_embedding_and_zs(sample):

    # Age prediction
    inputs = processor(text=age_texts, images=sample["image"], return_tensors="pt", padding=True).to(device)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image # This is the text-image similarity
    age_pred = logits_per_image.argmax(dim=1)
    sample["zs_age_clip"] = [int(gp) for gp in age_pred] 
    
    # Store embeddings - output of encoder, not projection
    
    inputs = processor(images=sample["image"], return_tensors="pt", padding=True).to(device)
    outputs = vision_model(**inputs)
    sample["embeddings"] = outputs.pooler_output

    return sample

#
counts = {k: 0 for k  in range(len(age_brackets))}

def filter_top_100(sample):
    if counts[sample['age']] < 100:
        counts[sample['age']] += 1
        return True
    return False

def rename_classes(sample):
    sample["age"] -= 2
    return sample


# Load training datasets
train_ds = datasets.load_dataset("HuggingFaceM4/FairFace", "1.25", split="train", verification_mode="no_checks")
train_ds = train_ds.shuffle(seed=42).filter(lambda sample: sample["age"] not in {0,1}) # removing first two classes from training datasets
train_ds = train_ds.map(rename_classes) # renaming classes 2 onwards to 0 onwards
train_ds = train_ds.filter(filter_top_100) # Filter out the top 100 samples per class
train_ds = train_ds.map(get_embedding_and_zs, batched=True, batch_size=16)

test_valid_ds = datasets.load_dataset("HuggingFaceM4/FairFace", "1.25", split="validation", verification_mode="no_checks")
test_valid_ds = test_valid_ds.shuffle(seed=42).filter(lambda sample: sample["age"] not in {0,1}) # removing first two classes from training datasets
valid_ds = test_valid_ds.select([i for i in range(6000)]) # Take only first 6000 for validation
valid_ds = valid_ds.map(rename_classes) # renaming classes 2 onwards to 0 onwards
valid_ds = valid_ds.map(get_embedding_and_zs, batched=True, batch_size=16)
test_ds = test_valid_ds.select([i for i in range(6000, len(test_valid_ds))]) # Take after the 6000th image in the test set
test_ds = test_ds.map(get_embedding_and_zs, batched=True, batch_size=16)
test_ds = test_ds.map(rename_classes) # renaming classes 2 onwards to 0 onwards

assert(len(test_valid_ds) == len(test_ds) + len(valid_ds)) # sanity check

# Make the np arrays
X_train = np.array(train_ds["embeddings"])
y_train = np.array(train_ds["age"])

X_val= np.array(valid_ds["embeddings"])
y_val= np.array(valid_ds["age"])

X_test = np.array(test_ds["embeddings"])
y_test = np.array(test_ds["age"])

# Define and train classifier
lr_clf_age = LogisticRegression(random_state=42, max_iter=750)
lr_clf_age.fit(X_train, y_train)

# Print training and validation accuracy
train_acc = lr_clf_age.score(X_train, y_train)
valid_acc = lr_clf_age.score(X_val, y_val)
print(f"Training accuracy: {train_acc:.3f}")
print(f"Validation accuracy: {valid_acc:.3f}")

# Print out per class accuracy
# import pdb; pdb.set_trace()
# Training data 
train_cwise_acc = []
y_preds = lr_clf_age.predict(X_train)
for idx, age in enumerate(age_brackets):
    age_mask = np.array(y_train) == idx
    y_true = np.array(y_train)[age_mask]
    y_rel_preds = np.array(y_preds)[age_mask]
    age_acc = np.sum(y_true == y_rel_preds) / len(y_true) * 100
    train_cwise_acc.append(age_acc)
    print(f"LR + CLIP accuracy for {age}(class - {idx}): {age_acc:.2f}%")

# Validation data

val_cwise_acc = []
y_preds = lr_clf_age.predict(X_val) + 2
for idx, age in enumerate(age_brackets):
    age_mask = np.array(y_val) == idx
    y_true = np.array(y_val)[age_mask]
    y_rel_preds = np.array(y_preds)[age_mask]
    age_acc = np.sum(y_true == y_rel_preds) / len(y_true) * 100
    val_cwise_acc.append(age_acc)
    print(f"LR + CLIP accuracy for {age}(class - {idx}): {age_acc:.2f}%")
