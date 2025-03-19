import os
import shutil
import cv2
import joblib
import json
import torch
import numpy as np
from deepface.modules import modeling, detection, preprocessing
# from deepface.extendedmodels import Age
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image
from tqdm.auto import tqdm




FACE_DETECTION_BACKEND = "ssd" # This backend from deepface detects the faces
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ExperimentDataset(Dataset):
    def __init__(self, img_dir):
        # store the image files in sorted order
        self.img_dir = img_dir
        self.img_files = sorted(os.listdir(img_dir))



    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        
        image = read_image(img_path)
        

        # Preprocessing

        # resize to a constant size
        image = transforms.Resize((224, 224))(image)

        # # Convert RGB image to BGR
        # bgr_image = cv2.cvtColor(np.array(to_pil_image(image)), cv2.COLOR_RGB2BGR)


        return image, self.img_files[idx].split(".")[0] # Remove the extension from the images

def setup_experiment():
    # Remove faces and demography folders for the experiment
    
    if os.path.exists("experiment/faces"):
        shutil.rmtree("experiment/faces")
    os.mkdir("experiment/faces")
    
    if os.path.exists("experiment/demography"):
        shutil.rmtree("experiment/demography")
    os.mkdir("experiment/demography")

def run_experiment(video_file):

    # Loop over video and create a dump of face images
    print(f"Extracting faces from {video_file}...")
    _extract_faces_from_video(video_file)
    print("\nFace extraction done.")
    # Create a dataset from the experiment data
    experiment_data = ExperimentDataset("experiment/faces")

    # Create a data-loader to run over the faces
    inference_dataloader = DataLoader(experiment_data, batch_size=16)

    print("Loading necessary models for analysis...")

    # Get CLIP model 
    clip_model, clip_processor = _load_clip()
    clip_model.to(device)

    # # Get DeepFace age model
    # age_model = _load_age_model()

    # Get LR claassifier for age
    age_scaler, age_model = _load_age_model()


    # Get LR classifiers
    gender_model, emotion_model, race_model = _load_lr_classifiers()
    gender_labels = ["Male", "Female"]
    race_labels = ["Asian", "Indian", "Black", "White", "Middle Eastern", "Latino Hispanic"]
    emotion_labels = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger"]
    age_labels = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "more than 70"]
    age_groups = ["Up to 50", "Over 50"]

    # Loop over each batch of images
    for images, filenames in tqdm(inference_dataloader):
        # # Predict age
        # apparent_ages = _predict_age(age_model, bgr_images)

        pil_images = [to_pil_image(i) for i in images]

        # Get clip embeddings for other models
        image_inputs = clip_processor(images=pil_images, return_tensors="pt", padding=True).to(device)
        image_embeds = clip_model.get_image_features(**image_inputs)
        image_embeds = image_embeds.detach().cpu() 

        # Predict gender
        gender_probabilities = _predict_gender(gender_model, image_embeds)
        # Predict race
        race_probabilities = _predict_race(race_model, image_embeds)
        # Predict emotion
        emotion_probabilities = _predict_emotion(emotion_model, image_embeds)
        # Predict age
        age_probabilities = _predict_age(age_scaler, age_model, image_embeds)

        for idx, file in enumerate(filenames):
            demo_data = {}

            # # Update apparent ages
            # demo_data["age"] = int(apparent_ages[idx])
            # Update probabilities for age
            # demo_data["age"] = {
            #     r : float(prob) for r,prob in zip(age_labels, age_probabilities[idx]) 
            # }
            demo_data["age"] = {
                r : float(prob) for r,prob in zip(age_labels, age_probabilities[idx]) 
            }
            demo_data["dominant_age"] = age_labels[np.argmax(age_probabilities[idx])]
            demo_data["max_sum_age"] = {
                "Over 50": sum(age_probabilities[idx][-3:]),
                "Up to 50": sum(age_probabilities[idx][:-3])
            }

            demo_data["age_group"] = sorted(demo_data["max_sum_age"].items(), key=lambda x: x[1], reverse=True)[0][0] # Sort in descending order by prob sum, and get key

            # Update probabilities for race
            demo_data["race"] = {
                r : float(prob) for r,prob in zip(race_labels, race_probabilities[idx]) 
            }
            demo_data["dominant_race"] = race_labels[np.argmax(race_probabilities[idx])]

            # Update probabilities for gender
            demo_data["gender"] = {
                r : float(prob) for r,prob in zip(gender_labels, gender_probabilities[idx]) 
            }
            demo_data["dominant_gender"] = gender_labels[np.argmax(gender_probabilities[idx])]

            # Update probabilities for emotion
            demo_data["emotion"] = {
                r : float(prob) for r,prob in zip(emotion_labels, emotion_probabilities[idx]) 
            }
            demo_data["dominant_emotion"] = emotion_labels[np.argmax(emotion_probabilities[idx])]


            # Dump the dict as a json
            with open(f"experiment/demography/{file}.json", 'w') as f:
                json.dump(demo_data, f)
    print("Successfully extracted all demography information.")



def _extract_faces_from_video(video_file):
    cap = cv2.VideoCapture(video_file)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Open capture loop
    ctr = 0
    while cap.isOpened():
        ret, frame = cap.read()    
        # ret is only true when frame is read properly
        if not ret:
            break
        if frame is None:
            continue 
        # Display/Process every 5th frame
        if ctr % 5 == 0:
            # Add processing here
            _get_and_dump_faces(frame, ctr)
        if cv2.waitKey(1) == ord('q'):
            break
        
        # Calculate and display the progress
        progress = (ctr + 1) / total_frames * 100
        print(f"Extracting frames: {progress:.2f}% complete", end='\r')

        ctr += 1

    cap.release()
    cv2.destroyAllWindows()

# def _predict_age(age_model, bgr_images):
    
#     # Returns predictions of each batch - (batch_size, age_classes)
#     age_preds = age_model(np.array(bgr_images).squeeze())

#     output_indexes = np.arange(101)

#     # Multiply each age class probability by its corresponding age index and sum across the age axis
#     apparent_ages = np.sum(age_preds * output_indexes, axis=1)

#     return apparent_ages

def _predict_age(age_scaler, age_model, images):
    age_preds = age_model.predict_proba(age_scaler.transform(images))
    return age_preds

def _predict_gender(gender_model, images):
    gender_preds = gender_model.predict_proba(images)
    return gender_preds

def _predict_race(race_model, images):
    race_preds = race_model.predict_proba(images)
    return race_preds

def _predict_emotion(emotion_model, images):
    emotion_preds = emotion_model.predict_proba(images)
    return emotion_preds



def _get_and_dump_faces(frame, frame_id):
    try:
        img_objs = detection.extract_faces(
            img_path=frame,
            detector_backend=FACE_DETECTION_BACKEND,
            enforce_detection=True,
            grayscale=False,
            align=True,
            expand_percentage=0,
            anti_spoofing=False,
        )

        for idx, img_obj in enumerate(img_objs):
            img_content = img_obj["face"]
            img_region = img_obj["facial_area"]
            img_confidence = img_obj["confidence"]
            if img_content.shape[0] == 0 or img_content.shape[1] == 0:
                continue

            # Write the face to the experiment folder
            # cv2.imshow("face", img_content)
            # import pdb; pdb.set_trace()
            x,y,w,h  = img_region["x"], img_region["y"], img_region["w"], img_region["h"]
            x = max(int(x),0)
            y = max(int(y),0)
            w = max(int(w),0)
            h = max(int(h),0)
            face_region = frame[y:y+h, x:x+w,:]
            # img_content = (img_content * 255).astype(np.uint8)
            cv2.imwrite(f"experiment/faces/face_{frame_id}_{idx}.png", face_region)
            # cv2.imshow("face", img_content)
            # cv2.waitKey(0)
    
            # # Destroy all OpenCV windows
            # cv2.destroyAllWindows()
    except ValueError:
        # Raises ValueError when a face is not detected
        # print("ValueError FAM")
        pass
    except Exception as e:
        print(str(e))

def _load_clip():
    # Load model and pre-processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

# def _generate_clip_embeddings(model, processor, images):
#     """
#     Input: CLIPModel, CLIPProcessor, Batch of images as pytorch tensors
#     Output: Tensor of embeddings of shape: (batch, embedding_size)
#     """
#     images = [to_pil_image(img) for img in images]
#     inputs = processor(images=images, return_tensors="pt", padding=True)
#     outputs = model(**inputs)
#     return image_embeds

def _load_lr_classifiers():
    gender_clf = joblib.load("models/lr_clf_gender.joblib")
    emotion_clf = joblib.load("models/lr_clf_emotion.joblib")
    race_clf = joblib.load("models/lr_clf_race.joblib")
    return gender_clf, emotion_clf, race_clf


# def _load_age_model():
#     # if not os.path.exists('/root/.deepface/weights'):
#     #     os.mkdir('/root/.deepface')
#     #     os.mkdir('/root/.deepface/weights')
#     age_model = modeling.build_model("Age").model
#     return age_model

def _load_age_model():
    # if not os.path.exists('/root/.deepface/weights'):
    #     os.mkdir('/root/.deepface')
    #     os.mkdir('/root/.deepface/weights')
    age_scaler = joblib.load("models/projected_scaler.joblib")
    age_clf = joblib.load("models/lr_clf_proj_age.joblib")
    return (age_scaler, age_clf)


# _get_and_dump_faces()