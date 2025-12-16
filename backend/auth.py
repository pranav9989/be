import jwt
from functools import wraps
from flask import request, jsonify, g, current_app
from datetime import datetime
from models import User

def create_access_token(user_id):
    payload = {
        "user_id": user_id,
        "exp": datetime.utcnow() + current_app.config["JWT_ACCESS_TOKEN_EXPIRE"],
        "iat": datetime.utcnow()
    }
    return jwt.encode(
        payload,
        current_app.config["JWT_SECRET_KEY"],
        algorithm="HS256"
    )

def jwt_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        auth = request.headers.get("Authorization")
        if not auth or not auth.startswith("Bearer "):
            return jsonify({"error": "Missing token"}), 401

        token = auth.split()[1]
        try:
            payload = jwt.decode(
                token,
                current_app.config["JWT_SECRET_KEY"],
                algorithms=["HS256"]
            )
            user = User.query.get(payload["user_id"])
            if not user:
                raise Exception()
            g.current_user = user
        except:
            return jsonify({"error": "Invalid or expired token"}), 401

        return fn(*args, **kwargs)
    return wrapper
