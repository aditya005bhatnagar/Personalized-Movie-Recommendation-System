import pandas as pd
import hashlib

from api.database import SessionLocal, User

users = pd.read_csv("data/users.csv")

db = SessionLocal()

for _, row in users.iterrows():

    user_id = int(row["userId"])

    existing_user = db.query(User).filter(
        User.id == user_id
    ).first()

    if existing_user:
        continue

    username = f"ml_user_{user_id}"
    email = f"ml_user_{user_id}@cinematch.local"

    password = f"MovieLens@{user_id}"

    hashed_password = hashlib.sha256(
        password.encode()
    ).hexdigest()

    new_user = User(
        id=user_id,
        username=username,
        email=email,
        password=hashed_password
    )

    db.add(new_user)

db.commit()
db.close()

print("MovieLens users imported successfully.")