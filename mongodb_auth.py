#!/usr/bin/env python3
"""
MongoDB-based Farmer Authentication System
Secure user management system for Maharashtra Agricultural System
"""

import os
import json
from mongodb_config import MongoCropDB
import bcrypt
import secrets
from datetime import datetime, timedelta
from bson.objectid import ObjectId

class MongoFarmerAuth:
    def __init__(self):
        """Initialize MongoDB connection and collections"""
        self.mongo_db = MongoCropDB()
        self.connected = getattr(self.mongo_db, 'connected', False)
        
        if self.connected and hasattr(self.mongo_db, 'db') and self.mongo_db.db is not None:
            self.farmers = self.mongo_db.db['farmers']
            self.sessions = self.mongo_db.db['farmer_sessions']
            self.login_attempts = self.mongo_db.db['login_attempts']
            self.preferences = self.mongo_db.db['farmer_preferences']
            # Create indexes
            self._create_indexes()
            print("[OK] MongoDB Authentication System initialized!")
        else:
            self.farmers = None
            self.sessions = None
            self.login_attempts = None
            self.preferences = None
            print("[WARNING] MongoDB Authentication System - Running in offline mode!")
            # prepare offline user store
            self.offline_auth_file = os.getenv('OFFLINE_AUTH_FILE', 'offline_auth.json')
            try:
                if not os.path.exists(self.offline_auth_file):
                    with open(self.offline_auth_file, 'w', encoding='utf-8') as f:
                        import json
                        json.dump({'users': []}, f)
                with open(self.offline_auth_file, 'r', encoding='utf-8') as f:
                    import json
                    data = json.load(f)
                    self.offline_users = data.get('users', [])
            except Exception as ex:
                print(f"[ERROR] unable to initialize offline auth store: {ex}")
                self.offline_users = []
    
    def _create_indexes(self):
        """Create necessary indexes for collections"""
        # Farmers collection indexes
        self.farmers.create_index('username', unique=True)
        self.farmers.create_index('email', unique=True)
        self.farmers.create_index('phone')
        
        # Sessions collection indexes
        self.sessions.create_index('session_id', unique=True)
        self.sessions.create_index('farmer_id')
        self.sessions.create_index('expires_at')
        
        # Login attempts collection indexes
        self.login_attempts.create_index([('username', 1), ('attempt_time', -1)])
        self.login_attempts.create_index('ip_address')
        
        # Preferences collection indexes
        self.preferences.create_index('farmer_id', unique=True)
    
    def hash_password(self, password):
        """Securely hash password with bcrypt"""
        salt = bcrypt.gensalt()
        password_hash = bcrypt.hashpw(password.encode('utf-8'), salt)
        return password_hash.decode('utf-8'), salt.decode('utf-8')
    
    def verify_password(self, password, stored_hash):
        """Verify password against stored hash"""
        return bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8'))
    
    def register_farmer(self, username, email, password, full_name, **kwargs):
        """Register a new farmer account"""
        if not self.connected:
            # operate offline: store user locally in JSON file
            if any(u['username'] == username or u['email'] == email for u in self.offline_users):
                return {"success": False, "message": "Username or email already exists (offline)"}
            password_hash, salt = self.hash_password(password)
            farmer_doc = {
                'username': username,
                'email': email,
                'password_hash': password_hash,
                'salt': salt,
                'full_name': full_name,
                'phone': kwargs.get('phone', ''),
                'farm_name': kwargs.get('farm_name', ''),
                'district': kwargs.get('district', ''),
                'village': kwargs.get('village', ''),
                'farm_area': kwargs.get('farm_area', 0),
                'crop_types': kwargs.get('crop_types', ''),
                'registration_date': datetime.now(),
                'last_login': None,
                'is_active': True,
                'profile_picture': kwargs.get('profile_picture', ''),
                'created_at': datetime.now(),
                'updated_at': datetime.now()
            }
            self.offline_users.append(farmer_doc)
            try:
                import json
                with open(self.offline_auth_file, 'w', encoding='utf-8') as f:
                    json.dump({'users': self.offline_users}, f, default=str)
            except Exception as ex:
                print(f"[ERROR] saving offline user: {ex}")
            return {"success": True, "message": "Farmer registered offline!", "farmer_id": None}
        try:
            # Check if username or email exists
            if self.farmers.find_one({'$or': [{'username': username}, {'email': email}]}):
                return {"success": False, "message": "Username or email already exists"}
            
            # Hash password
            password_hash, salt = self.hash_password(password)
            
            # Prepare farmer document
            farmer_doc = {
                'username': username,
                'email': email,
                'password_hash': password_hash,
                'salt': salt,
                'full_name': full_name,
                'phone': kwargs.get('phone', ''),
                'farm_name': kwargs.get('farm_name', ''),
                'district': kwargs.get('district', ''),
                'village': kwargs.get('village', ''),
                'farm_area': kwargs.get('farm_area', 0),
                'crop_types': kwargs.get('crop_types', ''),
                'registration_date': datetime.now(),
                'last_login': None,
                'is_active': True,
                'profile_picture': kwargs.get('profile_picture', ''),
                'created_at': datetime.now(),
                'updated_at': datetime.now()
            }
            
            # Insert farmer record
            result = self.farmers.insert_one(farmer_doc)
            farmer_id = result.inserted_id
            
            # Create default preferences
            self.preferences.insert_one({
                'farmer_id': farmer_id,
                'language': 'en',
                'theme': 'agricultural',
                'notifications_enabled': True,
                'email_alerts': True,
                'sms_alerts': False,
                'weather_alerts': True,
                'pest_alerts': True,
                'created_at': datetime.now(),
                'updated_at': datetime.now()
            })
            
            return {
                "success": True,
                "message": "Farmer registered successfully!",
                "farmer_id": str(farmer_id)
            }
            
        except Exception as e:
            return {"success": False, "message": f"Registration failed: {str(e)}"}
    
    def authenticate_farmer(self, username, password, ip_address=None):
        """Authenticate farmer login"""
        if not self.connected:
            # offline authentication
            farmer = next((u for u in self.offline_users if u['username'] == username or u['email'] == username), None)
            if not farmer:
                return {"success": False, "message": "Invalid username or password (offline)"}
            if not self.verify_password(password, farmer['password_hash']):
                return {"success": False, "message": "Invalid username or password (offline)"}
            return {"success": True, "message": "Login successful (offline)!", "farmer_id": None, "username": farmer['username'], "full_name": farmer['full_name']}
        try:
            # Get farmer record
            farmer = self.farmers.find_one({
                '$or': [{'username': username}, {'email': username}]
            })
            
            # Log login attempt
            attempt_doc = {
                'username': username,
                'ip_address': ip_address,
                'success': False,
                'attempt_time': datetime.now(),
                'error_message': ''
            }
            
            if not farmer:
                attempt_doc['error_message'] = 'User not found'
                self.login_attempts.insert_one(attempt_doc)
                return {"success": False, "message": "Invalid username or password"}
            
            if not farmer.get('is_active', True):
                attempt_doc['error_message'] = 'Account deactivated'
                self.login_attempts.insert_one(attempt_doc)
                return {"success": False, "message": "Account is deactivated"}
            
            # Verify password
            if not self.verify_password(password, farmer['password_hash']):
                attempt_doc['error_message'] = 'Invalid password'
                self.login_attempts.insert_one(attempt_doc)
                return {"success": False, "message": "Invalid username or password"}
            
            # Update successful login
            attempt_doc['success'] = True
            attempt_doc['error_message'] = 'Login successful'
            self.login_attempts.insert_one(attempt_doc)
            
            # Update last login
            self.farmers.update_one(
                {'_id': farmer['_id']},
                {'$set': {'last_login': datetime.now()}}
            )
            
            return {
                "success": True,
                "message": "Login successful!",
                "farmer_id": str(farmer['_id']),
                "username": farmer['username'],
                "full_name": farmer['full_name']
            }
            
        except Exception as e:
            return {"success": False, "message": f"Authentication failed: {str(e)}"}
    
    def create_session(self, farmer_id, ip_address=None, user_agent=None):
        """Create a new session for authenticated farmer"""
        session_id = secrets.token_urlsafe(32)
        session_token = secrets.token_urlsafe(64)
        expires_at = datetime.now() + timedelta(days=7)
        
        try:
            session_doc = {
                'session_id': session_id,
                'farmer_id': ObjectId(farmer_id),
                'session_token': session_token,
                'expires_at': expires_at,
                'created_at': datetime.now(),
                'ip_address': ip_address,
                'user_agent': user_agent,
                'is_active': True
            }
            
            self.sessions.insert_one(session_doc)
            
            return {
                "success": True,
                "session_id": session_id,
                "session_token": session_token,
                "expires_at": expires_at.isoformat()
            }
            
        except Exception as e:
            return {"success": False, "message": f"Session creation failed: {str(e)}"}
    
    def validate_session(self, session_id, session_token):
        """Validate farmer session"""
        try:
            # Find active session
            session = self.sessions.find_one({
                'session_id': session_id,
                'session_token': session_token,
                'is_active': True,
                'expires_at': {'$gt': datetime.now()}
            })
            
            if session:
                # Get farmer details
                farmer = self.farmers.find_one({'_id': session['farmer_id']})
                if farmer:
                    return {
                        "success": True,
                        "farmer_id": str(farmer['_id']),
                        "username": farmer['username'],
                        "full_name": farmer['full_name'],
                        "district": farmer['district'],
                        "expires_at": session['expires_at']
                    }
            
            return {"success": False, "message": "Invalid or expired session"}
            
        except Exception as e:
            return {"success": False, "message": f"Session validation failed: {str(e)}"}
    
    def invalidate_session(self, session_id):
        """Invalidate/logout a session"""
        try:
            result = self.sessions.update_one(
                {'session_id': session_id},
                {'$set': {'is_active': False}}
            )
            
            if result.modified_count > 0:
                return {"success": True, "message": "Session invalidated successfully"}
            return {"success": False, "message": "Session not found"}
            
        except Exception as e:
            return {"success": False, "message": f"Session invalidation failed: {str(e)}"}
    
    def get_farmer_profile(self, farmer_id):
        """Get farmer profile information"""
        try:
            farmer = self.farmers.find_one({'_id': ObjectId(farmer_id)})
            
            if farmer:
                return {
                    "success": True,
                    "profile": {
                        "farmer_id": str(farmer['_id']),
                        "username": farmer['username'],
                        "email": farmer['email'],
                        "phone": farmer.get('phone', ''),
                        "full_name": farmer['full_name'],
                        "farm_name": farmer.get('farm_name', ''),
                        "district": farmer.get('district', ''),
                        "village": farmer.get('village', ''),
                        "farm_area": farmer.get('farm_area', 0),
                        "crop_types": farmer.get('crop_types', ''),
                        "registration_date": farmer['registration_date'],
                        "last_login": farmer.get('last_login'),
                        "profile_picture": farmer.get('profile_picture', '')
                    }
                }
            else:
                return {"success": False, "message": "Farmer profile not found"}
                
        except Exception as e:
            return {"success": False, "message": f"Profile retrieval failed: {str(e)}"}
    
    def update_farmer_profile(self, farmer_id, updates):
        """Update farmer profile information"""
        try:
            # Don't allow updates to critical fields
            protected_fields = {'username', 'email', 'password_hash', 'salt', '_id'}
            update_data = {k: v for k, v in updates.items() if k not in protected_fields}
            update_data['updated_at'] = datetime.now()
            
            result = self.farmers.update_one(
                {'_id': ObjectId(farmer_id)},
                {'$set': update_data}
            )
            
            if result.modified_count > 0:
                return {"success": True, "message": "Profile updated successfully"}
            return {"success": False, "message": "No changes made to profile"}
            
        except Exception as e:
            return {"success": False, "message": f"Profile update failed: {str(e)}"}

# Initialize the authentication system when imported
if __name__ == "__main__":
    # Test the authentication system initialization
    auth = MongoFarmerAuth()
    print("MongoDB Authentication System initialized successfully!")