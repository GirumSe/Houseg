from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field, validator
from typing import Dict
import jwt
from datetime import datetime, timedelta
from sqlalchemy import create_engine, MetaData, Table, Column, String, select
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from passlib.context import CryptContext
import databases

# Secret key to encode the JWT token
SECRET_KEY = "4adc1a699a4c91598fe5aa517943c7b7e04cd27c2d616c7106630d20a0925a84"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 100

# Database URL
DATABASE_URL = "mysql+pymysql://houseg:houseseg1230@127.0.0.1:3306/houseg"

# Initialize database connection
database = databases.Database(DATABASE_URL)
metadata = MetaData()

# Define user table
users = Table(
    "users", metadata,
    Column("username", String, primary_key=True),
    Column("email", String, unique=True, index=True),
    Column("password", String)
)

# SQLAlchemy specific code, as mentioned in the FastAPI docs
engine = create_engine(DATABASE_URL)
metadata.create_all(engine)

# Create session local
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

app = FastAPI()

class Token(BaseModel):
    access_token: str
    token_type: str
    name: str

class TokenData(BaseModel):
    username: str | None = None

class User(BaseModel):
    username: str
    email: EmailStr

class UserInDB(User):
    password: str

class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=6)
    confirm_password: str = Field(..., min_length=6)

    @validator('confirm_password')
    def passwords_match(cls, v, values, **kwargs):
        if 'password' in values and v != values['password']:
            raise ValueError('passwords do not match')
        return v

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

async def get_user(db, username: str):
    query = users.select().where(users.c.username == username)
    user = await db.fetch_one(query)
    if user:
        return UserInDB(**user)

async def get_user_by_email(db, email: str):
    query = users.select().where(users.c.email == email)
    user = await db.fetch_one(query)
    if user:
        return UserInDB(**user)

async def authenticate_user(db, username: str, password: str):
    user = await get_user(db, username)
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

@app.on_event("startup")
async def startup():
    await database.connect()

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()

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
    user = await authenticate_user(database, data["username"], data["password"])
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
    return {"access_token": access_token, "token_type": "bearer", "name": user.username}

@app.post("/register", status_code=status.HTTP_201_CREATED)
async def register_user(user: UserCreate):
    # Check if user already exists
    existing_user = await get_user(database, user.username)
    if existing_user:
        raise HTTPException(
            status_code=400,
            detail="Username already registered"
        )
    
    existing_email = await get_user_by_email(database, user.email)
    if existing_email:
        raise HTTPException(
            status_code=400,
            detail="Email already registered"
        )
    
    # Hash the password
    hashed_password = get_password_hash(user.password)
    
    # Insert the new user into the database
    query = users.insert().values(username=user.username, email=user.email, password=hashed_password)
    await database.execute(query)
    
    return {"message": "User registered successfully"}
