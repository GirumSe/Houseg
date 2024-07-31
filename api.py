from fastapi import Depends, FastAPI, HTTPException, status, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field, validator

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms

from typing import Dict, Optional, List
import jwt
import shutil
import os
import time
import random
import string

from datetime import datetime, timedelta

import requests
from sqlalchemy import Column, ForeignKey, ForeignKeyConstraint, Index, BIGINT, VARCHAR, create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from passlib.context import CryptContext

import databases

# Secret key to encode the JWT token
SECRET_KEY = "Enter Your JWT Incription key Here"

GEOLOC_API_KEY = "Enter GEOLOC_API_KEY key Here"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 100

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database URL
DATABASE_URL = "Enter Ur MySql Database Url Here"

# Initialize database connection
database = databases.Database(DATABASE_URL)
engine = create_engine(DATABASE_URL, pool_size=10, max_overflow=20)
metadata = MetaData()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Define user table
class Locations(Base):
    __tablename__ = "locations"
    
    id = Column(BIGINT, primary_key=True, autoincrement=True)
    latitude = Column(BIGINT, nullable=False)
    longitude = Column(BIGINT, nullable=False)
    image_name = Column(VARCHAR(255), nullable=False)
    address = Column(VARCHAR(255), nullable=False)
    house_count = Column(BIGINT, nullable=False)
    user_id = Column(BIGINT, nullable=False, index=True)
    
    #__table_args__ = (Index('username_idx', 'username'),)

class Users(Base):
    __tablename__ = "users"
    
    id = Column(BIGINT,primary_key=True, index=True, autoincrement=True)
    username = Column(VARCHAR(255), nullable=False)
    email = Column(VARCHAR(255), nullable=False)
    password = Column(VARCHAR(255), nullable=False)
    
   # __table_args__ = (ForeignKeyConstraint(['username'], ['locations.username'], name='users_username_foreign'),)
    
class Token(BaseModel):
    access_token: str
    token_type: str
    name: str

class TokenData(BaseModel):
    username: str | None = None

class User(BaseModel):
    id: int
    username: str
    email: str
    password: str
    
class Location(BaseModel):
    id: int
    latitude: str
    longitude: str
    image_name: str
    address: str
    house_count: int
    user_id: int

class UserInDB(User):
    password: str

class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    
class LocationCreate(BaseModel):
    latitude: str
    longitude: str
    image_url: str    

class UserLogin(BaseModel):
    username: str
    password: str

def get_db():
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_password_hash(password):
    return pwd_context.hash(password)

def generate_unique_image_name():
    timestamp = int(time.time())
    random_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
    unique_name = f"{timestamp}_{random_str}"
    ext = "jpg"
    unique_image_name = f"{unique_name}.{ext}"
    return unique_image_name

def get_address_from_coordinates(lat, lon, api_key):
    base_url = "https://geocode.maps.co/reverse"
    params = {
        "lat": f"{lat}",
        "lon": f"{lon}",
        "api_key": api_key
    }
    
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        data = response.json()
        results = data["address"]
        state = results["state"]
        country = results["country"]
        return {
            "state": state,
            "country": country,
        }
    else:
        return {"error": "Request failed with status code " + str(response.status_code)}
    
    
# Load the model
model = fasterrcnn_resnet50_fpn_v2(weights=None)
# Load the state dictionary
num_classes = 4
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.load_state_dict(torch.load('ML_MODEL/fasterrcnn_resnet50_fpn.pth', map_location=torch.device('cpu'))) #you can Download the model from https://www.kaggle.com/models/girumsenay/fasterrcnn_resnet50_fpn
cpu_device = torch.device('cpu')

# Set the model to evaluation mode
model.eval()
# Define the transformation
def get_test_transform():
    return A.Compose([
        ToTensorV2(p=1.0)
    ])

# Function to preprocess a single image
def preprocess_image(image_path, transforms=None):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image /= 255.0
    
    # Apply transforms if provided
    if transforms:
        sample = {'image': image}
        sample = transforms(**sample)
        image = sample['image']
    
    return image

def draw_boxes_and_save_image(image_path, boxes, save_dir, image_name):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create a figure and axis
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    # Plot the predicted boxes in red
    for box in boxes:
        box = box.astype(np.int32)
        rect = plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
    
    # Save the figure
    save_path = os.path.join(save_dir, image_name)
    plt.savefig(save_path)
    plt.close(fig)

# Function to make predictions on a single image
def predict_single_image(image_path, model, device, cpu_device, save_dir, probability_threshold=0.5, iou_threshold=0.5):
    transforms = get_test_transform()
    image_tensor = preprocess_image(image_path, transforms).to(device)
    
    # Forward pass through the model
    with torch.no_grad():
        output = model([image_tensor])[0]
    
    # Move the output to the CPU
    output = {k: v.to(cpu_device) for k, v in output.items()}
    
    # Filter out predictions with confidence scores less than the threshold
    scores = output['scores'].cpu().numpy()
    boxes = output['boxes'].cpu().numpy()
    labels = output['labels'].cpu().numpy()
    
    # Apply NMS to the filtered boxes based on the scores
    filtered_idx = scores >= probability_threshold
    boxes = boxes[filtered_idx]
    labels = labels[filtered_idx]
    scores = scores[filtered_idx]

    boxes_tensor = torch.tensor(boxes)
    scores_tensor = torch.tensor(scores)
    keep_idx = nms(boxes_tensor, scores_tensor, iou_threshold=iou_threshold)
    
    boxes = boxes[keep_idx]
    labels = labels[keep_idx]
    scores = scores[keep_idx]
    
    # Draw boxes and save the image
    image_name = os.path.basename(image_path)
    draw_boxes_and_save_image(image_path, boxes, save_dir, image_name)
    
    return len(boxes)
    


@app.post("/register", response_model=User)
async def register_user(user: UserCreate, db: Session = Depends(get_db)):
    # Check if user already exists
    db_user = db.query(Users).filter(Users.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    db_email = db.query(Users).filter(Users.email == user.email).first()
    if db_email:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Hash the password
    hashed_password = get_password_hash(user.password)
    # Create new user record
    new_user = Users(
        username=user.username,
        email=user.email,
        password=hashed_password
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return new_user


@app.post("/token", response_model=Token)
async def login_for_access_token(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(Users).filter(Users.username == user.username).first()
    if not db_user or not verify_password(user.password, db_user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": db_user.username}, expires_delta=access_token_expires
    )
    return Token(access_token=access_token, token_type="bearer", name=db_user.username)


@app.post("/locations", response_model=Location)
async def create_location(    
    latitude: str = Form(...),
    longitude: str = Form(...),
    name: str = Form(...),
    image: UploadFile = File(...), 
    db: Session = Depends(get_db),
):
    try:    
        # Generate unique image name
        image_name = generate_unique_image_name()
        file_location = f"images/{image_name}"
        
        # Save the uploaded image
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        # Get the address from coordinates
        address_info = get_address_from_coordinates(latitude, longitude, GEOLOC_API_KEY)
        address = f"{address_info['country']}"
        # Predict the house count and save the visualization
        save_dir = "visualizations"
        os.makedirs(save_dir, exist_ok=True)
        house_count = predict_single_image(file_location, model, cpu_device, cpu_device, save_dir)
        
        # Fetch the user ID by name
        user = db.query(Users).filter(Users.username == name).first()
        if not user:
            raise HTTPException(status_code=400, detail="User not found")
        
        user_id = user.id
        
        # Create new location record
        new_location = Locations(
            latitude=latitude,
            longitude=longitude,
            image_name=image_name,
            address=address,
            house_count=house_count,
            user_id=user_id
        )
        
        db.add(new_location)
        db.commit()
        db.refresh(new_location)
        
        return new_location

    finally:
        image.file.close()



