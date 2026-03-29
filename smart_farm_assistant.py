#!/usr/bin/env python3
"""
Smart Farm Assistant - Complete Agricultural Management System
A comprehensive Streamlit application for farmers with proper functionality,
clean interface, and actionable insights.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sqlite3
import cv2
from PIL import Image
import tensorflow as tf
import os
import requests
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Smart Farm Assistant",
    page_icon="🚜",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for professional appearance
st.markdown(
    """
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-left: 10px;
        padding-right: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
    .metric-card {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e1e5e9;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .success-card {
        background: linear-gradient(45deg, #56ab2f 0%, #a8e6cf 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
    }
    .warning-card {
        background: linear-gradient(45deg, #ffa726 0%, #ffcc02 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
    }
    .danger-card {
        background: linear-gradient(45deg, #ff5722 0%, #ff8a80 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
    }
    .recommendation-item {
        background: #f8f9fa;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
</style>
""",
    unsafe_allow_html=True,
)


class SmartFarmAssistant:
    def __init__(self):
        """Initialize Smart Farm Assistant"""
        self.setup_database()
        self.load_models()

        # Farm data
        self.maharashtra_districts = [
            "Pune",
            "Mumbai",
            "Nashik",
            "Nagpur",
            "Aurangabad",
            "Solapur",
            "Ahmednagar",
            "Kolhapur",
            "Sangli",
            "Satara",
            "Raigad",
            "Thane",
            "Osmanabad",
            "Beed",
            "Latur",
            "Nanded",
            "Akola",
            "Buldhana",
            "Amravati",
            "Washim",
            "Yavatmal",
            "Wardha",
            "Gondia",
            "Bhandara",
        ]

        self.crop_types = {
            "Cereals": ["Rice", "Wheat", "Maize", "Jowar", "Bajra"],
            "Cash Crops": ["Cotton", "Sugarcane", "Soybean", "Sunflower"],
            "Vegetables": ["Tomato", "Potato", "Onion", "Chili", "Brinjal"],
            "Fruits": ["Mango", "Orange", "Banana", "Grapes", "Pomegranate"],
        }

        # Crop requirements database
        self.crop_requirements = {
            "Rice": {
                "water": "High",
                "soil_ph": "6.0-7.5",
                "temperature": "20-35°C",
                "season": "Kharif",
            },
            "Wheat": {
                "water": "Moderate",
                "soil_ph": "6.0-7.5",
                "temperature": "15-25°C",
                "season": "Rabi",
            },
            "Cotton": {
                "water": "Moderate",
                "soil_ph": "5.8-8.0",
                "temperature": "21-30°C",
                "season": "Kharif",
            },
            "Sugarcane": {
                "water": "High",
                "soil_ph": "6.0-7.5",
                "temperature": "21-27°C",
                "season": "Year-round",
            },
            "Tomato": {
                "water": "Moderate",
                "soil_ph": "6.0-6.8",
                "temperature": "20-25°C",
                "season": "Winter",
            },
            "Potato": {
                "water": "Moderate",
                "soil_ph": "5.2-6.4",
                "temperature": "15-20°C",
                "season": "Winter",
            },
        }

    def setup_database(self):
        """Setup SQLite database for storing farm data"""
        try:
            conn = sqlite3.connect("smart_farm_db.db")
            cursor = conn.cursor()

            # Farm records table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS farm_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT DEFAULT (datetime('now', 'localtime')),
                    district TEXT,
                    crop_type TEXT,
                    farm_size REAL,
                    crop_health_score REAL,
                    disease_detected TEXT,
                    soil_ph REAL,
                    nitrogen REAL,
                    phosphorus REAL,
                    potassium REAL,
                    recommendations TEXT
                )
            """
            )

            conn.commit()
            conn.close()
        except Exception as e:
            st.error(f"Database setup error: {str(e)}")

    def load_models(self):
        """Load AI models for crop analysis"""
        try:
            if os.path.exists("best_model.h5"):
                self.disease_model = tf.keras.models.load_model("best_model.h5")

                if os.path.exists("class_names.txt"):
                    with open("class_names.txt", "r") as f:
                        self.class_names = [line.strip() for line in f.readlines()]
                else:
                    self.class_names = [
                        "Healthy",
                        "Early Blight",
                        "Late Blight",
                        "Bacterial Spot",
                    ]

                st.success("✅ AI Disease Detection Model Loaded Successfully!")
            else:
                self.disease_model = None
                st.warning(
                    "⚠️ AI model not found. Disease detection will use sample data."
                )

        except Exception as e:
            self.disease_model = None
            st.error(f"❌ Model loading error: {str(e)}")

    def validate_image_file(self, uploaded_file):
        """Professional image file validation with detailed diagnostics"""
        import io

        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "quality_metrics": {},
            "severity": "success",  # success, warning, error, critical
        }

        try:
            # 1. File type validation
            file_extension = uploaded_file.name.split(".")[-1].lower()
            if file_extension not in ["jpg", "jpeg", "png"]:
                validation_result["valid"] = False
                validation_result["errors"].append(
                    f"Unsupported file format: .{file_extension}. Only JPG, JPEG, PNG accepted."
                )
                validation_result["severity"] = "error"
                return validation_result

            # 2. File size validation (limit to 15MB)
            file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
            if file_size_mb > 15:
                validation_result["valid"] = False
                validation_result["errors"].append(
                    f"File size too large: {file_size_mb:.2f}MB. Maximum allowed: 15MB."
                )
                validation_result["severity"] = "error"
                return validation_result

            # 3. Image loading validation
            try:
                image = Image.open(io.BytesIO(uploaded_file.getvalue()))
            except Exception as e:
                validation_result["valid"] = False
                validation_result["errors"].append(
                    f"Corrupted or invalid image file: {str(e)}"
                )
                validation_result["severity"] = "critical"
                return validation_result

            # 4. Image dimensions validation
            width, height = image.size
            aspect_ratio = width / height

            if width < 150 or height < 150:
                validation_result["warnings"].append(
                    f"Low resolution detected ({width}x{height}px). Minimum recommended: 150x150px."
                )
                validation_result["severity"] = (
                    "warning"
                    if validation_result["severity"] == "success"
                    else validation_result["severity"]
                )

            if width > 5000 or height > 5000:
                validation_result["warnings"].append(
                    f"Very high resolution ({width}x{height}px). May slow down analysis."
                )

            # 5. Color mode validation
            if image.mode not in ["RGB", "RGBA", "L"]:
                validation_result["warnings"].append(
                    f"Non-standard color mode: {image.mode}. Converting to RGB."
                )

            # 6. Quality metrics
            img_array = np.array(image)
            if len(img_array.shape) == 2:  # Grayscale
                img_array = np.stack([img_array] * 3, axis=2)

            if img_array.shape[2] == 4:  # RGBA to RGB
                img_array = img_array[:, :, :3]

            img_normalized = img_array.astype(np.float32) / 255.0

            brightness = np.mean(img_normalized)
            contrast = np.std(img_normalized)

            validation_result["quality_metrics"] = {
                "file_size_mb": round(file_size_mb, 2),
                "dimensions": f"{width}x{height}",
                "aspect_ratio": round(aspect_ratio, 2),
                "brightness": round(brightness, 3),
                "contrast": round(contrast, 3),
                "color_mode": image.mode,
                "format": image.format,
            }

            # 7. Content validation (too dark/bright)
            if brightness < 0.05:
                validation_result["errors"].append(
                    "Image is too dark. Brightness is critically low. Please retake in better lighting."
                )
                validation_result["valid"] = False
                validation_result["severity"] = "error"
            elif brightness > 0.95:
                validation_result["errors"].append(
                    "Image is overexposed/too bright. Please retake with less glare."
                )
                validation_result["valid"] = False
                validation_result["severity"] = "error"
            elif brightness < 0.15 or brightness > 0.85:
                validation_result["warnings"].append(
                    "Lighting is suboptimal. For best results, use natural daylight."
                )
                if validation_result["severity"] == "success":
                    validation_result["severity"] = "warning"

            # 8. Contrast validation
            if contrast < 0.02:
                validation_result["warnings"].append(
                    "Low contrast detected. Image may lack detail clarity."
                )
                if validation_result["severity"] == "success":
                    validation_result["severity"] = "warning"

        except Exception as e:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Validation error: {str(e)}")
            validation_result["severity"] = "critical"

        return validation_result

    def display_professional_image_alert(self, validation_result):
        """Display professional, color-coded image validation feedback"""

        # Success case
        if validation_result["severity"] == "success" and validation_result["valid"]:
            st.success("✓ Image validation passed successfully.")
            return

        # Error case
        if (
            validation_result["severity"] in ["error", "critical"]
            and not validation_result["valid"]
        ):
            error_header = (
                "🔴 Critical Image Error"
                if validation_result["severity"] == "critical"
                else "🟠 Image Error"
            )
            with st.container(border=True):
                st.markdown(f"### {error_header}")
                for error in validation_result["errors"]:
                    st.markdown(f"**• {error}**")

            # Show quality metrics if available
            if validation_result["quality_metrics"]:
                st.markdown("**Image Analysis:**")
                cols = st.columns(len(validation_result["quality_metrics"]))
                for col, (key, value) in zip(
                    cols, validation_result["quality_metrics"].items()
                ):
                    with col:
                        st.metric(key.replace("_", " ").title(), value)
            return

        # Warning case
        if validation_result["severity"] == "warning":
            with st.container(border=True):
                st.markdown("### 🟡 Image Quality Warning")

                col_warn, col_metrics = st.columns([1.5, 1])

                with col_warn:
                    st.markdown("**Detected Issues:**")
                    for warning in validation_result["warnings"]:
                        st.markdown(f"• {warning}")

                with col_metrics:
                    if validation_result["quality_metrics"]:
                        st.markdown("**Metrics:**")
                        metrics = validation_result["quality_metrics"]
                        st.markdown(f"• **Size:** {metrics.get('dimensions', 'N/A')}")
                        st.markdown(
                            f"• **Brightness:** {metrics.get('brightness', 'N/A')}"
                        )
                        st.markdown(f"• **Contrast:** {metrics.get('contrast', 'N/A')}")

                st.info(
                    "✓ Analysis will proceed, but accuracy may be affected. For best results, consider retaking the image."
                )

    def analyze_crop_health(self, uploaded_image, crop_type):
        """Comprehensive crop health analysis"""
        if uploaded_image is None:
            return None

        try:
            # Reset file pointer
            uploaded_image.seek(0)

            # Image preprocessing with better error handling
            image = Image.open(uploaded_image)

            # Convert to RGB if necessary
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Resize and normalize
            image_resized = image.resize((224, 224))
            image_array = np.array(image_resized) / 255.0

            # Disease detection (using model or simulation)
            if self.disease_model is not None:
                try:
                    image_batch = np.expand_dims(image_array, axis=0)
                    predictions = self.disease_model.predict(image_batch, verbose=0)
                    predicted_class = np.argmax(predictions[0])
                    confidence = float(predictions[0][predicted_class])

                    disease = (
                        self.class_names[predicted_class]
                        if predicted_class < len(self.class_names)
                        else "Unknown"
                    )
                except Exception as model_error:
                    st.warning(
                        f"Model prediction failed: {model_error}. Using simulation."
                    )
                    # Fallback to simulation
                    diseases = [
                        "Healthy",
                        "Early Blight",
                        "Late Blight",
                        "Bacterial Spot",
                    ]
                    disease = np.random.choice(diseases)
                    confidence = np.random.uniform(0.7, 0.95)
            else:
                # Simulation for demo - make it more realistic
                np.random.seed(hash(str(image_array.mean())) % 1000)
                diseases = ["Healthy", "Early Blight", "Late Blight", "Bacterial Spot"]
                disease = np.random.choice(
                    diseases, p=[0.4, 0.25, 0.20, 0.15]
                )  # Weighted probabilities
                confidence = np.random.uniform(0.65, 0.95)

            # Calculate comprehensive health score
            health_score = self.calculate_health_score(disease, confidence, crop_type)

            # Generate recommendations
            recommendations = self.generate_crop_recommendations(
                disease, crop_type, health_score
            )

            result = {
                "disease": disease,
                "confidence": confidence * 100,
                "health_score": health_score,
                "recommendations": recommendations,
                "severity": self.get_disease_severity(disease, confidence),
            }

            return result

        except Exception as e:
            st.error(f"Image analysis error: {str(e)}")
            # Return a default result instead of None
            return {
                "disease": "Analysis Failed",
                "confidence": 0.0,
                "health_score": 50.0,
                "recommendations": [
                    "Please try uploading a different image",
                    "Ensure image is clear and well-lit",
                ],
                "severity": "Unknown",
            }

    def calculate_health_score(self, disease, confidence, crop_type):
        """Calculate comprehensive health score (0-100)"""
        base_score = 85

        if disease.lower() == "healthy":
            score = min(95, base_score + (confidence * 10))
        else:
            severity_penalty = {
                "early blight": 15,
                "late blight": 25,
                "bacterial spot": 20,
                "leaf curl": 18,
                "mosaic virus": 22,
            }
            penalty = severity_penalty.get(disease.lower(), 20)
            score = max(20, base_score - (confidence * penalty))

        return round(score, 1)

    def get_disease_severity(self, disease, confidence):
        """Get disease severity level"""
        if disease.lower() == "healthy":
            return "Healthy"
        elif confidence > 0.8:
            return "Severe"
        elif confidence > 0.6:
            return "Moderate"
        else:
            return "Mild"

    def generate_crop_recommendations(self, disease, crop_type, health_score):
        """Generate actionable farming recommendations"""
        recommendations = []

        # Disease-specific recommendations
        disease_actions = {
            "healthy": [
                "Continue current farming practices",
                "Regular monitoring for early disease detection",
                "Apply organic fertilizers as per schedule",
                "Maintain proper irrigation timing",
            ],
            "early blight": [
                "Remove affected leaves immediately",
                "Apply copper-based fungicide spray",
                "Improve air circulation between plants",
                "Reduce overhead watering",
                "Apply potassium-rich fertilizer to boost immunity",
            ],
            "late blight": [
                "Remove and destroy infected plants",
                "Apply preventive fungicide treatment",
                "Ensure proper field drainage",
                "Avoid working in fields when wet",
                "Use resistant varieties in next planting",
            ],
            "bacterial spot": [
                "Apply copper-based bactericide",
                "Remove infected plant debris",
                "Use drip irrigation instead of overhead",
                "Apply calcium supplements",
                "Rotate with non-solanaceous crops",
            ],
        }

        recommendations.extend(
            disease_actions.get(
                disease.lower(), ["Consult agricultural expert for treatment"]
            )
        )

        # Health score based recommendations
        if health_score < 50:
            recommendations.extend(
                [
                    "Consider soil testing for nutrient deficiency",
                    "Apply balanced NPK fertilizer",
                    "Check irrigation system efficiency",
                ]
            )
        elif health_score < 70:
            recommendations.extend(
                [
                    "Monitor crop closely for next 7 days",
                    "Apply organic compost to improve soil health",
                ]
            )

        return recommendations

    def get_weather_data(self, district, days=7):
        """Get weather data for analysis"""
        try:
            # Generate realistic weather data for Maharashtra
            np.random.seed(hash(district) % 1000)
            dates = [
                (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
                for i in range(days - 1, -1, -1)
            ]

            # Seasonal adjustments for Maharashtra
            month = datetime.now().month
            if month in [12, 1, 2]:  # Winter
                temp_base, temp_var = 18, 12
                humidity_base, humidity_var = 45, 25
                rain_intensity = 0.5
            elif month in [3, 4, 5]:  # Summer
                temp_base, temp_var = 30, 15
                humidity_base, humidity_var = 35, 20
                rain_intensity = 0.3
            else:  # Monsoon/Post-monsoon
                temp_base, temp_var = 25, 10
                humidity_base, humidity_var = 70, 20
                rain_intensity = 2.0

            # Generate weather arrays with proper bounds
            temperatures = np.clip(
                temp_base + temp_var * np.random.random(days), 5, 50
            ).round(1)
            humidity = np.clip(
                humidity_base + humidity_var * np.random.random(days), 20, 100
            ).round(1)
            rainfall = np.clip(
                np.random.exponential(rain_intensity, days), 0, 200
            ).round(1)
            wind_speed = np.clip(3 + 7 * np.random.random(days), 0, 25).round(1)

            weather_data = {
                "dates": dates,
                "temperature": temperatures.tolist(),
                "humidity": humidity.tolist(),
                "rainfall": rainfall.tolist(),
                "wind_speed": wind_speed.tolist(),
            }

            return weather_data

        except Exception as e:
            st.error(f"Weather data generation error: {str(e)}")
            # Return default weather data
            dates = [
                (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
                for i in range(days - 1, -1, -1)
            ]
            return {
                "dates": dates,
                "temperature": [25.0] * days,
                "humidity": [60.0] * days,
                "rainfall": [2.0] * days,
                "wind_speed": [5.0] * days,
            }

    def create_weather_charts(self, weather_data):
        """Create professional weather visualization charts"""
        try:
            # Temperature trend
            temp_fig = px.line(
                x=weather_data["dates"],
                y=weather_data["temperature"],
                title="Temperature Trend (7 Days)",
                labels={"x": "Date", "y": "Temperature (°C)"},
            )
            temp_fig.update_traces(
                line=dict(color="#ff6b6b", width=3),
                mode="lines+markers",
                marker=dict(size=8),
            )
            temp_fig.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(size=12),
                height=350,
                title_font=dict(size=16, color="#2c3e50"),
                showlegend=False,
            )

            # Humidity and Rainfall combined chart
            humidity_rain_fig = go.Figure()

            # Add humidity as bars
            humidity_rain_fig.add_trace(
                go.Bar(
                    x=weather_data["dates"],
                    y=weather_data["humidity"],
                    name="Humidity (%)",
                    marker_color="rgba(54, 162, 235, 0.7)",
                    yaxis="y",
                )
            )

            # Add rainfall as line
            humidity_rain_fig.add_trace(
                go.Scatter(
                    x=weather_data["dates"],
                    y=weather_data["rainfall"],
                    name="Rainfall (mm)",
                    line=dict(color="#28a745", width=3),
                    mode="lines+markers",
                    yaxis="y2",
                )
            )

            humidity_rain_fig.update_layout(
                title="Humidity & Rainfall Pattern",
                xaxis=dict(title="Date"),
                yaxis=dict(title="Humidity (%)", side="left"),
                yaxis2=dict(title="Rainfall (mm)", side="right", overlaying="y"),
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(size=12),
                height=350,
                title_font=dict(size=16, color="#2c3e50"),
                legend=dict(x=0.01, y=0.99),
            )

            return temp_fig, humidity_rain_fig

        except Exception as e:
            st.error(f"Chart creation error: {str(e)}")
            # Return simple fallback charts
            import plotly.graph_objects as go

            temp_fig = go.Figure()
            temp_fig.add_trace(
                go.Scatter(x=[0, 1], y=[25, 25], mode="lines", name="Temperature")
            )
            temp_fig.update_layout(title="Temperature Chart (Error)", height=350)

            humidity_fig = go.Figure()
            humidity_fig.add_trace(
                go.Scatter(x=[0, 1], y=[60, 60], mode="lines", name="Humidity")
            )
            humidity_fig.update_layout(title="Humidity Chart (Error)", height=350)

            return temp_fig, humidity_fig

    def analyze_soil_health(self, ph, nitrogen, phosphorus, potassium):
        """Analyze soil health based on NPK values"""
        # Soil health scoring
        ph_score = 100 if 6.0 <= ph <= 7.5 else max(50, 100 - abs(ph - 6.75) * 20)
        n_score = min(100, max(20, nitrogen * 2))
        p_score = min(100, max(20, phosphorus * 3))
        k_score = min(100, max(20, potassium * 2.5))

        overall_score = (ph_score + n_score + p_score + k_score) / 4

        # Generate soil recommendations
        recommendations = []

        if ph < 6.0:
            recommendations.append("Soil is acidic - apply lime to increase pH")
        elif ph > 7.5:
            recommendations.append(
                "Soil is alkaline - apply organic matter to reduce pH"
            )

        if nitrogen < 30:
            recommendations.append("Low nitrogen - apply urea or compost")
        if phosphorus < 25:
            recommendations.append("Low phosphorus - apply DAP or rock phosphate")
        if potassium < 35:
            recommendations.append("Low potassium - apply muriate of potash")

        if not recommendations:
            recommendations.append("Soil health is good - maintain current practices")

        return {
            "overall_score": round(overall_score, 1),
            "ph_status": "Optimal" if 6.0 <= ph <= 7.5 else "Needs adjustment",
            "nutrient_status": {
                "nitrogen": "Good" if nitrogen >= 30 else "Low",
                "phosphorus": "Good" if phosphorus >= 25 else "Low",
                "potassium": "Good" if potassium >= 35 else "Low",
            },
            "recommendations": recommendations,
        }

    def save_farm_record(self, data):
        """Save farm analysis data to database"""
        try:
            conn = sqlite3.connect("smart_farm_db.db")
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO farm_records 
                (district, crop_type, farm_size, crop_health_score, disease_detected, 
                 soil_ph, nitrogen, phosphorus, potassium, recommendations)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                data,
            )

            conn.commit()
            conn.close()
            return True
        except Exception as e:
            st.error(f"Database error: {str(e)}")
            return False


def main():
    """Main Streamlit application"""

    # Initialize assistant
    if "assistant" not in st.session_state:
        st.session_state.assistant = SmartFarmAssistant()

    assistant = st.session_state.assistant

    # Main header
    st.title("🚜 Smart Farm Assistant")
    st.markdown("**Your Complete Agricultural Management System**")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("🏡 Farm Settings")

        selected_district = st.selectbox(
            "📍 Select District", assistant.maharashtra_districts, index=0
        )

        crop_category = st.selectbox(
            "🌱 Crop Category", list(assistant.crop_types.keys())
        )

        selected_crop = st.selectbox(
            "🌾 Select Crop", assistant.crop_types[crop_category]
        )

        farm_size = st.number_input(
            "🏞️ Farm Size (Hectares)",
            min_value=0.1,
            max_value=1000.0,
            value=2.0,
            step=0.5,
        )

        st.markdown("---")
        st.markdown("### 📊 Quick Stats")

        # Quick weather info
        weather_data = assistant.get_weather_data(selected_district, 1)
        current_temp = weather_data["temperature"][0]
        current_humidity = weather_data["humidity"][0]

        st.metric("🌡️ Current Temp", f"{current_temp}°C")
        st.metric("💧 Humidity", f"{current_humidity}%")

    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "🏠 Dashboard",
            "🔍 Crop Analysis",
            "🌤️ Weather",
            "🌱 Soil Health",
            "📈 Records",
        ]
    )

    with tab1:
        st.header("🏠 Farm Dashboard")

        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(
                """
            <div class="metric-card">
                <h3>🌾 Current Crop</h3>
                <h2>{}</h2>
                <p>Growth Season</p>
            </div>
            """.format(
                    selected_crop
                ),
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                """
            <div class="metric-card">
                <h3>🏞️ Farm Size</h3>
                <h2>{} Ha</h2>
                <p>Total Area</p>
            </div>
            """.format(
                    farm_size
                ),
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                """
            <div class="metric-card">
                <h3>🌡️ Temperature</h3>
                <h2>{}°C</h2>
                <p>Current Weather</p>
            </div>
            """.format(
                    current_temp
                ),
                unsafe_allow_html=True,
            )

        with col4:
            st.markdown(
                """
            <div class="metric-card">
                <h3>💧 Humidity</h3>
                <h2>{}%</h2>
                <p>Moisture Level</p>
            </div>
            """.format(
                    current_humidity
                ),
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # Crop requirements info
        if selected_crop in assistant.crop_requirements:
            req = assistant.crop_requirements[selected_crop]
            st.subheader(f"🌱 {selected_crop} Growing Requirements")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    f"""
                <div class="info-card">
                    <h4>💧 Water Requirement</h4>
                    <p><strong>{req['water']}</strong></p>
                    
                    <h4>🌡️ Optimal Temperature</h4>
                    <p><strong>{req['temperature']}</strong></p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with col2:
                st.markdown(
                    f"""
                <div class="info-card">
                    <h4>🧪 Soil pH Range</h4>
                    <p><strong>{req['soil_ph']}</strong></p>
                    
                    <h4>📅 Growing Season</h4>
                    <p><strong>{req['season']}</strong></p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

    with tab2:
        st.header("🔍 Crop Health Analysis")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📷 Upload Crop Image")
            uploaded_file = st.file_uploader(
                "Choose an image of your crop",
                type=["jpg", "jpeg", "png"],
                help="Upload a clear photo of crop leaves for disease analysis",
            )

            if uploaded_file:
                # Professional image validation
                validation_result = assistant.validate_image_file(uploaded_file)

                # Display professional validation alerts
                assistant.display_professional_image_alert(validation_result)

                # Only proceed if not critical error
                if validation_result["severity"] not in ["error", "critical"]:
                    st.image(
                        uploaded_file, caption="Uploaded Image", use_column_width=True
                    )

            else:
                # Animated placeholder message
                st.markdown(
                    """
                    <style>
                        @keyframes pulse-glow {
                            0%, 100% {
                                box-shadow: 0 0 10px rgba(76, 175, 80, 0.3), inset 0 0 10px rgba(76, 175, 80, 0.1);
                                transform: scale(1);
                            }
                            50% {
                                box-shadow: 0 0 20px rgba(76, 175, 80, 0.6), inset 0 0 15px rgba(76, 175, 80, 0.2);
                                transform: scale(1.01);
                            }
                        }
                        
                        @keyframes float-up {
                            0%, 100% { transform: translateY(0px); }
                            50% { transform: translateY(-8px); }
                        }
                        
                        @keyframes fade-in-out {
                            0%, 100% { opacity: 0.7; }
                            50% { opacity: 1; }
                        }
                        
                        .upload-placeholder {
                            background: linear-gradient(135deg, rgba(76, 175, 80, 0.08) 0%, rgba(102, 187, 106, 0.08) 100%);
                            border: 3px solid rgba(76, 175, 80, 0.4);
                            border-radius: 18px;
                            padding: 2.5rem 2rem;
                            text-align: center;
                            animation: pulse-glow 2.5s ease-in-out infinite;
                            margin: 1.5rem 0;
                        }
                        
                        .upload-icon {
                            font-size: 3.5rem;
                            animation: float-up 2.5s ease-in-out infinite;
                            display: inline-block;
                            margin-bottom: 0.8rem;
                        }
                        
                        .upload-text {
                            font-size: 1.35rem;
                            font-weight: 500;
                            color: #2E7D32;
                            margin: 0.8rem 0;
                            letter-spacing: 0.3px;
                        }
                        
                        .upload-subtext {
                            font-size: 1rem;
                            color: #558B2F;
                            opacity: 0.9;
                            margin-top: 0.6rem;
                            animation: fade-in-out 2.5s ease-in-out infinite;
                        }
                    </style>
                    
                    <div class="upload-placeholder">
                        <div class="upload-icon">📸</div>
                        <div class="upload-text">⚠️ Please upload a valid crop leaf image</div>
                        <div class="upload-subtext">for accurate disease detection</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with col2:
            if uploaded_file:
                # Check validation result before showing analyze button
                if "validation_result" not in st.session_state:
                    if uploaded_file:
                        st.session_state.validation_result = (
                            assistant.validate_image_file(uploaded_file)
                        )

                validation = st.session_state.get(
                    "validation_result", {"severity": "success"}
                )
                is_disabled = validation["severity"] in ["error", "critical"]
                button_label = (
                    "⚠️ Fix Issues First" if is_disabled else "🔍 Analyze Crop Health"
                )

                if st.button(
                    button_label, type="primary", width="stretch", disabled=is_disabled
                ):
                    with st.spinner("Analyzing crop health..."):
                        result = assistant.analyze_crop_health(
                            uploaded_file, selected_crop
                        )

                        if result:
                            st.session_state.analysis_result = result

        # Display results
        if "analysis_result" in st.session_state:
            result = st.session_state.analysis_result

            st.markdown("---")
            st.subheader("📊 Analysis Results")

            # Health score display
            health_score = result["health_score"]
            if health_score >= 80:
                card_class = "success-card"
                status = "Excellent"
            elif health_score >= 60:
                card_class = "warning-card"
                status = "Good"
            else:
                card_class = "danger-card"
                status = "Needs Attention"

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(
                    f"""
                <div class="{card_class}">
                    <h3>🏥 Health Score</h3>
                    <h1>{health_score}/100</h1>
                    <p>{status}</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with col2:
                st.markdown(
                    f"""
                <div class="info-card">
                    <h3>🦠 Disease Status</h3>
                    <h2>{result['disease']}</h2>
                    <p>Confidence: {result['confidence']:.1f}%</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with col3:
                st.markdown(
                    f"""
                <div class="info-card">
                    <h3>⚠️ Severity Level</h3>
                    <h2>{result['severity']}</h2>
                    <p>Action Required</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            # Recommendations
            st.subheader("💡 Recommended Actions")
            for i, rec in enumerate(result["recommendations"], 1):
                st.markdown(
                    f"""
                <div class="recommendation-item">
                    <strong>{i}.</strong> {rec}
                </div>
                """,
                    unsafe_allow_html=True,
                )

    with tab3:
        st.header("🌤️ Weather Analysis")

        weather_data = assistant.get_weather_data(selected_district, 7)

        # Weather summary
        avg_temp = np.mean(weather_data["temperature"])
        total_rainfall = np.sum(weather_data["rainfall"])
        avg_humidity = np.mean(weather_data["humidity"])

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "🌡️ Avg Temperature",
                f"{avg_temp:.1f}°C",
                f"{weather_data['temperature'][-1] - weather_data['temperature'][-2]:.1f}°C",
            )

        with col2:
            st.metric(
                "🌧️ Total Rainfall",
                f"{total_rainfall:.1f} mm",
                f"{weather_data['rainfall'][-1]:.1f} mm (today)",
            )

        with col3:
            st.metric(
                "💧 Avg Humidity",
                f"{avg_humidity:.1f}%",
                f"{weather_data['humidity'][-1] - weather_data['humidity'][-2]:.1f}%",
            )

        # Weather charts
        temp_fig, humidity_rain_fig = assistant.create_weather_charts(weather_data)

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(temp_fig, width="stretch")
        with col2:
            st.plotly_chart(humidity_rain_fig, width="stretch")

        # Weather recommendations
        st.subheader("🌦️ Weather-Based Recommendations")

        recommendations = []
        if avg_temp > 35:
            recommendations.append(
                "🌡️ High temperatures - Increase irrigation frequency and provide shade"
            )
        elif avg_temp < 15:
            recommendations.append(
                "🌡️ Low temperatures - Use mulching and protect from frost"
            )

        if total_rainfall > 100:
            recommendations.append(
                "🌧️ Heavy rainfall - Ensure proper drainage to prevent waterlogging"
            )
        elif total_rainfall < 10:
            recommendations.append("🌧️ Low rainfall - Plan for supplemental irrigation")

        if avg_humidity > 85:
            recommendations.append(
                "💧 High humidity - Monitor for fungal diseases and improve ventilation"
            )
        elif avg_humidity < 40:
            recommendations.append(
                "💧 Low humidity - Increase irrigation and consider mulching"
            )

        if not recommendations:
            recommendations.append(
                "✅ Weather conditions are favorable for crop growth"
            )

        for rec in recommendations:
            st.markdown(
                f"<div class='recommendation-item'>{rec}</div>", unsafe_allow_html=True
            )

    with tab4:
        st.header("🌱 Soil Health Assessment")

        st.subheader("🧪 Enter Soil Test Results")

        col1, col2 = st.columns(2)

        with col1:
            soil_ph = st.slider("pH Level", 4.0, 9.0, 6.5, 0.1, key="soil_ph_slider")
            nitrogen = st.number_input(
                "Nitrogen (N) - kg/ha", 0, 200, 45, 5, key="nitrogen_input"
            )

        with col2:
            phosphorus = st.number_input(
                "Phosphorus (P) - kg/ha", 0, 100, 30, 5, key="phosphorus_input"
            )
            potassium = st.number_input(
                "Potassium (K) - kg/ha", 0, 200, 40, 5, key="potassium_input"
            )

        # Store values in session state
        st.session_state.soil_ph = soil_ph
        st.session_state.nitrogen = nitrogen
        st.session_state.phosphorus = phosphorus
        st.session_state.potassium = potassium

        if st.button("🔍 Analyze Soil Health", type="primary"):
            soil_analysis = assistant.analyze_soil_health(
                soil_ph, nitrogen, phosphorus, potassium
            )
            st.session_state.soil_analysis = soil_analysis

        if "soil_analysis" in st.session_state:
            analysis = st.session_state.soil_analysis

            st.markdown("---")
            st.subheader("📊 Soil Analysis Results")

            # Overall score
            score = analysis["overall_score"]
            if score >= 80:
                card_class = "success-card"
                status = "Excellent"
            elif score >= 60:
                card_class = "warning-card"
                status = "Good"
            else:
                card_class = "danger-card"
                status = "Needs Improvement"

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown(
                    f"""
                <div class="{card_class}">
                    <h3>🌱 Soil Health</h3>
                    <h1>{score}/100</h1>
                    <p>{status}</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with col2:
                st.markdown(
                    f"""
                <div class="info-card">
                    <h3>🧪 pH Status</h3>
                    <h2>{soil_ph}</h2>
                    <p>{analysis['ph_status']}</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with col3:
                st.markdown(
                    f"""
                <div class="info-card">
                    <h3>🍃 NPK Status</h3>
                    <p>N: {analysis['nutrient_status']['nitrogen']}</p>
                    <p>P: {analysis['nutrient_status']['phosphorus']}</p>
                    <p>K: {analysis['nutrient_status']['potassium']}</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with col4:
                # NPK Chart
                nutrients = ["Nitrogen", "Phosphorus", "Potassium"]
                values = [nitrogen, phosphorus, potassium]

                fig = px.bar(
                    x=nutrients,
                    y=values,
                    title="NPK Levels",
                    color=values,
                    color_continuous_scale="RdYlGn",
                )
                fig.update_layout(height=200, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            # Soil recommendations
            st.subheader("🧪 Soil Improvement Recommendations")
            for i, rec in enumerate(analysis["recommendations"], 1):
                st.markdown(
                    f"""
                <div class="recommendation-item">
                    <strong>{i}.</strong> {rec}
                </div>
                """,
                    unsafe_allow_html=True,
                )

    with tab5:
        st.header("📈 Farm Records")

        # Save current analysis
        if st.button("💾 Save Current Analysis", type="primary"):
            if (
                "analysis_result" in st.session_state
                and "soil_analysis" in st.session_state
            ):
                crop_result = st.session_state.analysis_result
                soil_result = st.session_state.soil_analysis

                # Get soil values from session state if available
                soil_ph_value = getattr(st.session_state, "soil_ph", 6.5)
                nitrogen_value = getattr(st.session_state, "nitrogen", 45)
                phosphorus_value = getattr(st.session_state, "phosphorus", 30)
                potassium_value = getattr(st.session_state, "potassium", 40)

                record_data = (
                    selected_district,
                    selected_crop,
                    farm_size,
                    crop_result["health_score"],
                    crop_result["disease"],
                    soil_ph_value,
                    nitrogen_value,
                    phosphorus_value,
                    potassium_value,
                    str(crop_result["recommendations"]),
                )

                if assistant.save_farm_record(record_data):
                    st.success("✅ Farm record saved successfully!")
                else:
                    st.error("❌ Error saving record")
            else:
                st.warning("⚠️ Please complete crop and soil analysis first")

        # Display records
        try:
            conn = sqlite3.connect("smart_farm_db.db")
            df = pd.read_sql_query(
                "SELECT date, district, crop_type, crop_health_score, disease_detected FROM farm_records ORDER BY date DESC LIMIT 10",
                conn,
            )
            conn.close()

            if not df.empty:
                st.subheader("📊 Recent Records")
                st.dataframe(df, width="stretch")

                # Health score trend
                if len(df) > 1:
                    fig = px.line(
                        df,
                        x="date",
                        y="crop_health_score",
                        title="Crop Health Score Trend",
                        markers=True,
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, width="stretch")
            else:
                st.info(
                    "📝 No records found. Start analyzing your crops to build history!"
                )

        except Exception as e:
            st.error(f"Error loading records: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p>🚜 <strong>Smart Farm Assistant</strong> - Empowering Farmers with Technology</p>
            <p>Made with ❤️ for sustainable agriculture in Maharashtra</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
