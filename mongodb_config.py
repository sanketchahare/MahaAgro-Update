"""
MongoDB Configuration for Maharashtra Crop System
Handles database connections and operations
"""

from pymongo import MongoClient
from datetime import datetime, timedelta
import os
import ssl
from dotenv import load_dotenv

load_dotenv()  # Load environment variables


class MongoCropDB:
    def __init__(self, connection_string=None):
        """Initialize MongoDB connection with SSL support"""
        if connection_string is None:
            connection_string = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")

        try:
            # Try to connect with SSL/TLS if it's an Atlas connection
            if "mongodb.net" in connection_string:
                # For MongoDB Atlas, use TLS/SSL
                self.client = MongoClient(
                    connection_string,
                    serverSelectionTimeoutMS=5000,
                    socketTimeoutMS=10000,
                    connectTimeoutMS=10000,
                    retryWrites=True,
                    w="majority",
                    tls=True,
                    tlsAllowInvalidCertificates=False,
                )
            else:
                # For local MongoDB, no SSL needed
                self.client = MongoClient(
                    connection_string, serverSelectionTimeoutMS=5000
                )

            # Test connection immediately
            self.client.server_info()
            self.connected = True

            self.db = self.client["maharashtra_agri_db"]

            # Initialize collections
            self.farmers = self.db["farmers"]
            self.crop_analysis = self.db["crop_analysis"]
            self.weather_data = self.db["weather_data"]
            self.pest_predictions = self.db["pest_predictions"]
            self.soil_analysis = self.db["soil_analysis"]

            # Create indexes for better query performance
            self.create_indexes()
            print("[OK] MongoDB connection successful!")

        except Exception as e:
            print(f"[WARNING] MongoDB connection failed: {str(e)}")
            if "MONGODB_URI" not in os.environ or "localhost" in connection_string:
                print(
                    "[INFO] No MONGODB_URI environment variable or using localhost. "
                    "Set MONGODB_URI to your Atlas or production database."
                )
            else:
                print(
                    "[INFO] Please verify network connectivity and credentials for the MongoDB URI."
                )
            print("[INFO] Running in offline mode - features may be limited")
            self.connected = False
            self.client = None
            self.db = None
            self.farmers = None
            self.crop_analysis = None
            self.weather_data = None
            self.pest_predictions = None
            self.soil_analysis = None
            # prepare offline storage file for basic persistence
            try:
                self.offline_path = os.getenv('MONGODB_OFFLINE_FILE', 'mongo_offline.json')
                if not os.path.exists(self.offline_path):
                    with open(self.offline_path, 'w') as f:
                        import json
                        json.dump({}, f)
                print(f"[INFO] Offline data will be recorded to {self.offline_path}")
            except Exception as ex:
                print(f"[ERROR] Could not initialize offline storage: {ex}")
            # prepare offline storage file for basic persistence
            try:
                self.offline_path = os.getenv('MONGODB_OFFLINE_FILE', 'mongo_offline.json')
                if not os.path.exists(self.offline_path):
                    with open(self.offline_path, 'w') as f:
                        import json
                        json.dump({}, f)
                print(f"[INFO] Offline data will be recorded to {self.offline_path}")
            except Exception as ex:
                print(f"[ERROR] Could not initialize offline storage: {ex}")

    def _write_offline(self, collection: str, document: dict):
        """Append a document to the offline JSON store (used when MongoDB is unavailable)."""
        try:
            import json
            with open(self.offline_path, 'r+', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except Exception:
                    data = {}
                data.setdefault(collection, []).append(document)
                f.seek(0)
                f.truncate()
                json.dump(data, f, default=str)
        except Exception as e:
            print(f"[ERROR] offline write failed for {collection}: {e}")

    def create_indexes(self):
        """Create indexes for frequently accessed fields"""
        if not self.connected:
            return

        try:
            # Farmers collection indexes
            self.farmers.create_index([("farmer_id", 1)], unique=True)
            self.farmers.create_index([("username", 1)], unique=True)
            self.farmers.create_index([("email", 1)], unique=True)
            self.farmers.create_index([("district", 1)])

            # Crop analysis indexes
            self.crop_analysis.create_index([("farmer_id", 1), ("timestamp", -1)])
            self.crop_analysis.create_index([("district", 1), ("crop_type", 1)])

            # Weather data indexes
            self.weather_data.create_index([("district", 1), ("timestamp", -1)])

            # Pest predictions indexes
            self.pest_predictions.create_index(
                [("district", 1), ("prediction_date", -1)]
            )

            # Soil analysis indexes
            self.soil_analysis.create_index([("farmer_id", 1), ("analysis_date", -1)])
        except Exception as e:
            print(f"[WARNING] Could not create indexes: {e}")

    def save_crop_analysis(self, analysis_data):
        """Save crop analysis results"""
        if not self.connected:
            # offline fallback
            self._write_offline('crop_analysis', analysis_data)
            return {'offline': True}

        try:
            # Ensure proper data types
            analysis_doc = {
                "timestamp": (
                    datetime.now()
                    if isinstance(analysis_data["timestamp"], str)
                    else analysis_data["timestamp"]
                ),
                "district": str(analysis_data["district"]),
                "crop_type": str(analysis_data["crop_type"]),
                "growth_stage": str(analysis_data["growth_stage"]),
                "farm_area": float(analysis_data["farm_area"]),
                "disease_detected": str(analysis_data["disease_detected"]),
                "confidence": float(analysis_data["confidence"]),
                "ndvi_value": float(analysis_data["ndvi_value"]),
                "soil_ph": float(analysis_data["soil_ph"]),
                "nitrogen": float(analysis_data["nitrogen"]),
                "phosphorus": float(analysis_data["phosphorus"]),
                "potassium": float(analysis_data["potassium"]),
                "recommendations": analysis_data["recommendations"],
                "farmer_id": analysis_data.get("farmer_id"),
                "created_at": analysis_data.get("created_at", datetime.now()),
            }

            result = self.crop_analysis.insert_one(analysis_doc)
            if result.inserted_id:
                print(
                    f"[OK] Crop analysis saved successfully with ID: {result.inserted_id}"
                )
                return result
            else:
                print("[ERROR] Failed to save crop analysis")
                return None

        except Exception as e:
            print(f"[ERROR] Error saving crop analysis: {e}")
            return None

    def save_weather_data(self, weather_data):
        """Save weather data"""
        if not self.connected:
            self._write_offline('weather_data', weather_data)
            return {'offline': True}
        try:
            weather_data["timestamp"] = datetime.now()
            return self.weather_data.insert_one(weather_data)
        except Exception as e:
            print(f"[ERROR] Error saving weather data: {e}")
            return None

    def save_pest_prediction(self, pest_data):
        """Save pest prediction results"""
        if not self.connected:
            self._write_offline('pest_predictions', pest_data)
            return {'offline': True}
        try:
            pest_data["prediction_date"] = datetime.now()
            return self.pest_predictions.insert_one(pest_data)
        except Exception as e:
            print(f"[ERROR] Error saving pest prediction: {e}")
            return None

    def save_soil_analysis(self, soil_data):
        """Save soil analysis results"""
        if not self.connected:
            self._write_offline('soil_analysis', soil_data)
            return {'offline': True}
        try:
            soil_data["analysis_date"] = datetime.now()
            return self.soil_analysis.insert_one(soil_data)
        except Exception as e:
            print(f"[ERROR] Error saving soil analysis: {e}")
            return None

    def get_farmer_history(self, farmer_id):
        """Get farmer's historical data"""
        if not self.connected:
            return {"crop_analysis": [], "soil_analysis": []}
        try:
            return {
                "crop_analysis": list(
                    self.crop_analysis.find({"farmer_id": farmer_id}, {"_id": 0})
                    .sort("timestamp", -1)
                    .limit(10)
                ),
                "soil_analysis": list(
                    self.soil_analysis.find({"farmer_id": farmer_id}, {"_id": 0})
                    .sort("analysis_date", -1)
                    .limit(5)
                ),
            }
        except Exception as e:
            print(f"[ERROR] Error fetching farmer history: {e}")
            return {"crop_analysis": [], "soil_analysis": []}

    def get_district_summary(self, district):
        """Get district-wise summary"""
        if not self.connected:
            return []
        try:
            pipeline = [
                {"$match": {"district": district}},
                {"$sort": {"timestamp": -1}},
                {"$limit": 100},
                {
                    "$group": {
                        "_id": "$crop_type",
                        "avg_ndvi": {"$avg": "$ndvi_value"},
                        "diseases_detected": {"$addToSet": "$disease_detected"},
                        "analysis_count": {"$sum": 1},
                    }
                },
            ]
            return list(self.crop_analysis.aggregate(pipeline))
        except Exception as e:
            print(f"[ERROR] Error fetching district summary: {e}")
            return []

    def get_weather_history(self, district, days=30):
        """Get weather history for a district"""
        if not self.connected:
            return []
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            return list(
                self.weather_data.find(
                    {"district": district, "timestamp": {"$gte": cutoff_date}},
                    {"_id": 0},
                ).sort("timestamp", -1)
            )
        except Exception as e:
            print(f"[ERROR] Error fetching weather history: {e}")
            return []

    def get_pest_alerts(self, district):
        """Get recent pest alerts for a district"""
        if not self.connected:
            return []
        try:
            return list(
                self.pest_predictions.find(
                    {"district": district, "risk_level": {"$in": ["High", "Critical"]}},
                    {"_id": 0},
                )
                .sort("prediction_date", -1)
                .limit(5)
            )
        except Exception as e:
            print(f"[ERROR] Error fetching pest alerts: {e}")
            return []

    def close(self):
        """Close MongoDB connection"""
        if self.connected and self.client:
            self.client.close()
