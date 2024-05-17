import databases
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

from sqlalchemy import create_engine, MetaData, Table, Column, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.future import select

from pydantic import BaseModel
from typing import Dict
import jwt
from datetime import datetime, timedelta

# Secret key to encode the JWT token
SECRET_KEY = "4adc1a699a4c91598fe5aa517943c7b7e04cd27c2d616c7106630d20a0925a84"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 100

DATABASE_URL = "mysql+pymysql://houseg:houseseg1230@127.0.0.1:3306/houseg"

# Initialize database connection
database = databases.Database(DATABASE_URL)
metadata = MetaData()

app = FastAPI()

class Token(BaseModel):
    access_token: str
    token_type: str
    name: str

class TokenData(BaseModel):
    username: str | None = None

class User(BaseModel):
    username: str

class UserInDB(User):
    password: str

def verify_password(plain_password, hashed_password):
    # For simplicity, we are using plain text passwords
    # In a real application, use a hashing library like bcrypt
    return plain_password == hashed_password

def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)

def authenticate_user(fake_db, username: str, password: str):
    user = get_user(fake_db, username)
    if not user:
        return False
    if not verify_password(password, user.password):
        return False
    return user

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now() + expires_delta
    else:
        expire = datetime.now() + timedelta(minutes=30)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/token", response_model=Token)
async def login_for_access_token(request: Request):
    data = await request.json()
    user = authenticate_user(fake_users_db, data["username"], data["password"])
    if not user:
        raise HTTPException(
            status_code=400,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    print(access_token)
    return {"access_token": access_token, "token_type": "bearer", "name": user.username}
