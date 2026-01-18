# auth.py
import jwt
from functools import wraps
from flask import request, jsonify, current_app
from datetime import datetime

# Import User inside functions to avoid premature SQLAlchemy initialization


def create_access_token(user_id):
    """
    Create a JWT access token for a user.
    """
    payload = {
        "user_id": user_id,
        "iat": datetime.utcnow(),
        "exp": datetime.utcnow() + current_app.config["JWT_ACCESS_TOKEN_EXPIRE"],
    }

    return jwt.encode(
        payload,
        current_app.config["JWT_SECRET_KEY"],
        algorithm="HS256"
    )


def verify_token(token):
    """
    Verify JWT token and return user_id if valid.
    """
    try:
        payload = jwt.decode(
            token,
            current_app.config["JWT_SECRET_KEY"],
            algorithms=["HS256"]
        )
        return payload.get("user_id")
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def jwt_required(f):
    """
    JWT authentication decorator.
    Injects authenticated user into flask.g
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        from models import User  # Import here to avoid premature SQLAlchemy init

        auth_header = request.headers.get("Authorization")

        if not auth_header or not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing or invalid Authorization header"}), 401

        token = auth_header.split(" ")[1]
        user_id = verify_token(token)

        if not user_id:
            return jsonify({"error": "Invalid or expired token"}), 401

        # 🔑 IMPORTANT: Use the SAME SQLAlchemy instance as app.py
        user = User.query.get(user_id)
        if not user:
            return jsonify({"error": "User not found"}), 401

        # Attach user to request context (SAFE)
        from flask import g
        g.current_user = user

        return f(*args, **kwargs)

    return decorated_function
