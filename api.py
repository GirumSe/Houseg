from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field, validator

from typing import Dict, Optional
import jwt

from datetime import datetime, timedelta

from sqlalchemy import Column, ForeignKey, ForeignKeyConstraint, Index, BIGINT, VARCHAR, create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from passlib.context import CryptContext

import databases

# Secret key to encode the JWT token
SECRET_KEY = "4adc1a699a4c91598fe5aa517943c7b7e04cd27c2d616c7106630d20a0925a84"
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
DATABASE_URL = "mysql+pymysql://houseg:houseseg1230@127.0.0.1:3306/houseg"

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
    username: str

class UserInDB(User):
    password: str

class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    
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
    print(hashed_password)
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

# Create the database tables
Base.metadata.create_all(bind=engine)
