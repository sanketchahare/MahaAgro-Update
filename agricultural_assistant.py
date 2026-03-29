#!/usr/bin/env python3
"""
Agricultural Assistant - Simple Crop Health & Weather Analysis
A user-friendly Streamlit application for farmers to analyze crop health,
get weather insights, and receive farming recommendations.
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

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Agricultural Assistant",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better appearance
st.markdown(
    """
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4682B4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
    }
    .recommendation-box {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #90EE90;
    }
</style>
""",
    unsafe_allow_html=True,
)


class SimpleAgriAssistant:
    def __init__(self):
        """Initialize the Agricultural Assistant"""
        self.openweather_api_key = os.getenv("OPENWEATHER_API_KEY", "demo_key")

        # Load AI model for disease detection
        self.load_disease_model()

        # Initialize database
        self.init_database()

        # Maharashtra districts
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

        # Common crops in Maharashtra
        self.common_crops = [
            "Rice",
            "Wheat",
            "Cotton",
            "Sugarcane",
            "Soybean",
            "Tomato",
            "Potato",
            "Onion",
            "Maize",
            "Sunflower",
        ]

    def load_disease_model(self):
        """Load disease detection model"""
        try:
            if os.path.exists("best_model.h5"):
                self.disease_model = tf.keras.models.load_model("best_model.h5")

                # Load class names
                if os.path.exists("class_names.txt"):
                    with open("class_names.txt", "r") as f:
                        self.class_names = [line.strip() for line in f.readlines()]
                else:
                    self.class_names = ["Healthy", "Early Blight", "Late Blight"]

                st.success("✅ Disease detection model loaded successfully!")
            else:
                self.disease_model = None
                st.warning(
                    "⚠️ Disease detection model not found. Upload feature disabled."
                )

        except Exception as e:
            self.disease_model = None
            st.error(f"❌ Error loading model: {str(e)}")

    def init_database(self):
        """Initialize SQLite database"""
        try:
            conn = sqlite3.connect("agricultural_assistant.db")
            cursor = conn.cursor()

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS crop_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    district TEXT,
                    crop_type TEXT,
                    health_score REAL,
                    disease_detected TEXT,
                    recommendations TEXT
                )
            """
            )

            conn.commit()
            conn.close()

        except Exception as e:
            st.error(f"Database error: {str(e)}")

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

    def analyze_crop_image(self, uploaded_file):
        """Analyze uploaded crop image for disease detection"""
        if self.disease_model is None:
            return {"error": "Disease detection model not available"}

        try:
            # Load and preprocess image
            image = Image.open(uploaded_file)
            image_array = np.array(image.resize((224, 224))) / 255.0
            image_batch = np.expand_dims(image_array, axis=0)

            # Make prediction
            predictions = self.disease_model.predict(image_batch)
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])

            # Calculate health score
            health_score = self.calculate_health_score(confidence, predicted_class)

            raw_label = (
                self.class_names[predicted_class]
                if predicted_class < len(self.class_names)
                else "Unknown"
            )
            friendly = self.format_disease_name(raw_label.replace(" ", "_"))
            key = raw_label.replace(" ", "_")

            return {
                "disease": friendly,
                "raw_disease": raw_label,
                "disease_description": self.get_disease_description(key),
                "confidence": confidence * 100,
                "health_score": health_score,
                "recommendations": self.get_disease_recommendations(raw_label),
            }

        except Exception as e:
            return {"error": str(e)}

    def calculate_health_score(self, confidence, predicted_class):
        """Calculate crop health score (0-100)"""
        if self.class_names[predicted_class].lower() == "healthy":
            return min(95, 70 + (confidence * 25))
        else:
            return max(20, 80 - (confidence * 60))

    def get_weather_data(self, district):
        """Get weather data for the selected district"""
        try:
            # Generate sample weather data (in real app, use API)
            dates = [
                (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
                for i in range(7, 0, -1)
            ]

            np.random.seed(42)
            weather_data = {
                "dates": dates,
                "temperature": (20 + 15 * np.random.random(7)).round(1),
                "humidity": (40 + 40 * np.random.random(7)).round(1),
                "rainfall": np.random.exponential(2, 7).round(1),
                "wind_speed": (2 + 8 * np.random.random(7)).round(1),
            }

            return weather_data

        except Exception as e:
            st.error(f"Weather data error: {str(e)}")
            return None

    def create_simple_weather_charts(self, weather_data):
        """Create simple, easy-to-understand weather charts"""

        # Temperature chart
        temp_fig = px.line(
            x=weather_data["dates"],
            y=weather_data["temperature"],
            title="Temperature Over Last 7 Days",
            labels={"x": "Date", "y": "Temperature (°C)"},
        )
        temp_fig.update_traces(line_color="#FF6B6B", line_width=3)
        temp_fig.update_layout(
            plot_bgcolor="white", paper_bgcolor="white", font=dict(size=12), height=300
        )

        # Humidity chart
        humidity_fig = px.bar(
            x=weather_data["dates"],
            y=weather_data["humidity"],
            title="Humidity Levels",
            labels={"x": "Date", "y": "Humidity (%)"},
        )
        humidity_fig.update_traces(marker_color="#4ECDC4")
        humidity_fig.update_layout(
            plot_bgcolor="white", paper_bgcolor="white", font=dict(size=12), height=300
        )

        # Rainfall chart
        rainfall_fig = px.bar(
            x=weather_data["dates"],
            y=weather_data["rainfall"],
            title="Rainfall Pattern",
            labels={"x": "Date", "y": "Rainfall (mm)"},
        )
        rainfall_fig.update_traces(marker_color="#95A5A6")
        rainfall_fig.update_layout(
            plot_bgcolor="white", paper_bgcolor="white", font=dict(size=12), height=300
        )

        return temp_fig, humidity_fig, rainfall_fig

    def format_disease_name(self, label: str) -> str:
        """Humanize a raw disease label (underscores -> spaces, title case)."""
        if not isinstance(label, str) or label == "":
            return "Unknown"
        prefix = ""
        if label.startswith("Possible_"):
            prefix = "Possible "
            label = label[len("Possible_") :]
        formatted = label.replace("_", " ").title()
        return f"{prefix}{formatted}"

    def get_disease_description(self, disease_key: str) -> str:
        """Brief explanation of disease for user clarity."""
        info = {
            "Healthy": "Crop appears healthy with no disease symptoms.",
            "Early_Blight": "Fungal spots on leaves/stems that spread under warm humid conditions.",
            "Late_Blight": "Rapidly destructive fungal infection causing water-soaked lesions.",
        }
        return info.get(disease_key, "No description available.")

    def get_disease_recommendations(self, disease):
        """Get recommendations based on detected disease"""
        recommendations = {
            "Healthy": [
                "Continue current care practices",
                "Monitor regularly for early signs of problems",
                "Ensure adequate water and nutrients",
                "Consider preventive organic treatments",
            ],
            "Early Blight": [
                "Remove affected leaves immediately",
                "Apply copper-based fungicide",
                "Improve air circulation around plants",
                "Water at soil level, avoid wetting leaves",
            ],
            "Late Blight": [
                "Remove and destroy affected plants",
                "Apply preventive fungicide treatments",
                "Ensure good drainage in the field",
                "Consider resistant varieties for next season",
            ],
        }

        key = disease.replace(" ", "_")
        return recommendations.get(disease, ["Consult with local agricultural expert"])

    def get_weather_recommendations(self, weather_data):
        """Get recommendations based on weather conditions"""
        avg_temp = np.mean(weather_data["temperature"])
        avg_humidity = np.mean(weather_data["humidity"])
        total_rainfall = np.sum(weather_data["rainfall"])

        recommendations = []

        if avg_temp > 35:
            recommendations.append(
                "🌡️ High temperatures detected - increase irrigation frequency"
            )
        elif avg_temp < 15:
            recommendations.append(
                "🌡️ Low temperatures - protect crops from cold damage"
            )

        if avg_humidity > 80:
            recommendations.append("💧 High humidity - watch for fungal diseases")
        elif avg_humidity < 40:
            recommendations.append("💧 Low humidity - increase watering")

        if total_rainfall > 50:
            recommendations.append("🌧️ High rainfall - ensure good drainage")
        elif total_rainfall < 10:
            recommendations.append("🌧️ Low rainfall - plan for irrigation")

        if not recommendations:
            recommendations.append("✅ Weather conditions are favorable for crops")

        return recommendations

    def save_analysis_to_db(
        self, district, crop_type, health_score, disease, recommendations
    ):
        """Save analysis results to database"""
        try:
            conn = sqlite3.connect("agricultural_assistant.db")
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO crop_analysis (district, crop_type, health_score, disease_detected, recommendations)
                VALUES (?, ?, ?, ?, ?)
            """,
                (district, crop_type, health_score, disease, str(recommendations)),
            )

            conn.commit()
            conn.close()

        except Exception as e:
            st.error(f"Error saving to database: {str(e)}")


def main():
    """Main Streamlit application"""

    # Initialize the assistant
    if "assistant" not in st.session_state:
        st.session_state.assistant = SimpleAgriAssistant()

    assistant = st.session_state.assistant

    # Main header
    st.markdown(
        '<h1 class="main-header">🌾 Agricultural Assistant</h1>', unsafe_allow_html=True
    )
    st.markdown(
        '<p style="text-align: center; font-size: 1.2rem; color: #666;">Your friendly farming companion for crop health and weather analysis</p>',
        unsafe_allow_html=True,
    )

    # Sidebar for user inputs
    st.sidebar.header("🔧 Settings")

    selected_district = st.sidebar.selectbox(
        "Select Your District:", assistant.maharashtra_districts, index=0
    )

    selected_crop = st.sidebar.selectbox(
        "Select Your Crop:", assistant.common_crops, index=0
    )

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Dashboard", "📷 Crop Analysis", "🌤️ Weather Analysis", "📈 History"]
    )

    # Dashboard Tab
    with tab1:
        st.markdown(
            '<h2 class="sub-header">Farm Overview Dashboard</h2>',
            unsafe_allow_html=True,
        )

        # Weather data for dashboard
        weather_data = assistant.get_weather_data(selected_district)

        if weather_data:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                current_temp = weather_data["temperature"][-1]
                st.metric(
                    label="🌡️ Current Temperature",
                    value=f"{current_temp}°C",
                    delta=f"{current_temp - weather_data['temperature'][-2]:.1f}°C",
                )

            with col2:
                current_humidity = weather_data["humidity"][-1]
                st.metric(
                    label="💧 Humidity",
                    value=f"{current_humidity}%",
                    delta=f"{current_humidity - weather_data['humidity'][-2]:.1f}%",
                )

            with col3:
                recent_rainfall = weather_data["rainfall"][-1]
                st.metric(
                    label="🌧️ Recent Rainfall",
                    value=f"{recent_rainfall} mm",
                    delta=f"{recent_rainfall - weather_data['rainfall'][-2]:.1f} mm",
                )

            with col4:
                wind_speed = weather_data["wind_speed"][-1]
                st.metric(
                    label="💨 Wind Speed",
                    value=f"{wind_speed} m/s",
                    delta=f"{wind_speed - weather_data['wind_speed'][-2]:.1f} m/s",
                )

        # Quick recommendations
        st.markdown(
            '<h3 class="sub-header">Quick Recommendations</h3>', unsafe_allow_html=True
        )
        if weather_data:
            recommendations = assistant.get_weather_recommendations(weather_data)
            for rec in recommendations:
                st.markdown(
                    f'<div class="recommendation-box">{rec}</div><br>',
                    unsafe_allow_html=True,
                )

    # Crop Analysis Tab
    with tab2:
        st.markdown(
            '<h2 class="sub-header">Crop Health Analysis</h2>', unsafe_allow_html=True
        )

        uploaded_file = st.file_uploader(
            "Upload a photo of your crop:",
            type=["jpg", "jpeg", "png"],
            help="Take a clear photo of your crop leaves for disease detection",
        )

        if uploaded_file is not None:
            # Professional image validation
            validation_result = assistant.validate_image_file(uploaded_file)

            # Display professional validation alerts
            assistant.display_professional_image_alert(validation_result)

            # Only proceed if not critical error
            if validation_result["severity"] not in ["error", "critical"]:
                col1, col2 = st.columns([1, 1])

                with col1:
                    st.image(
                        uploaded_file,
                        caption="Uploaded Crop Image",
                        use_column_width=True,
                    )

                with col2:
                    is_disabled = validation_result["severity"] in ["error", "critical"]
                    button_label = (
                        "⚠️ Fix Issues First"
                        if is_disabled
                        else "🔍 Analyze Crop Health"
                    )

                    if st.button(button_label, type="primary", disabled=is_disabled):
                        with st.spinner("Analyzing your crop image..."):
                            result = assistant.analyze_crop_image(uploaded_file)

                            if "error" not in result:
                                st.success("Analysis Complete!")

                                # Display results
                                st.metric(
                                    "🏥 Health Score",
                                    f"{result['health_score']:.1f}/100",
                                )
                                st.metric("🦠 Disease Detected", result["disease"])
                                st.metric(
                                    "🎯 Confidence", f"{result['confidence']:.1f}%"
                                )

                                # Recommendations
                                st.markdown(
                                    '<h4 class="sub-header">Recommendations:</h4>',
                                    unsafe_allow_html=True,
                                )
                                recommendations = result["recommendations"]
                                for i, rec in enumerate(recommendations, 1):
                                    st.markdown(f"{i}. {rec}")

                                # Save to database
                                assistant.save_analysis_to_db(
                                    selected_district,
                                    selected_crop,
                                    result["health_score"],
                                    result["disease"],
                                    recommendations,
                                )

                            else:
                                st.error(f"Analysis failed: {result['error']}")

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

    # Weather Analysis Tab
    with tab3:
        st.markdown(
            '<h2 class="sub-header">Weather Analysis</h2>', unsafe_allow_html=True
        )

        weather_data = assistant.get_weather_data(selected_district)

        if weather_data:
            # Create charts
            temp_fig, humidity_fig, rainfall_fig = (
                assistant.create_simple_weather_charts(weather_data)
            )

            # Display charts
            col1, col2 = st.columns(2)

            with col1:
                st.plotly_chart(temp_fig, width="stretch")
                st.plotly_chart(rainfall_fig, width="stretch")

            with col2:
                st.plotly_chart(humidity_fig, width="stretch")

                # Weather insights
                st.markdown(
                    '<h4 class="sub-header">Weather Insights</h4>',
                    unsafe_allow_html=True,
                )
                avg_temp = np.mean(weather_data["temperature"])
                avg_humidity = np.mean(weather_data["humidity"])
                total_rainfall = np.sum(weather_data["rainfall"])

                st.write(f"**Average Temperature:** {avg_temp:.1f}°C")
                st.write(f"**Average Humidity:** {avg_humidity:.1f}%")
                st.write(f"**Total Rainfall:** {total_rainfall:.1f} mm")

    # History Tab
    with tab4:
        st.markdown(
            '<h2 class="sub-header">Analysis History</h2>', unsafe_allow_html=True
        )

        try:
            conn = sqlite3.connect("agricultural_assistant.db")
            df = pd.read_sql_query(
                "SELECT * FROM crop_analysis ORDER BY timestamp DESC LIMIT 10", conn
            )
            conn.close()

            if not df.empty:
                st.dataframe(df, width="stretch")

                # Simple chart of health scores over time
                if len(df) > 1:
                    fig = px.line(
                        df, x="timestamp", y="health_score", title="Health Score Trends"
                    )
                    fig.update_layout(
                        plot_bgcolor="white", paper_bgcolor="white", height=400
                    )
                    st.plotly_chart(fig, width="stretch")
            else:
                st.info("No analysis history found. Start by analyzing some crops!")

        except Exception as e:
            st.error(f"Error loading history: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #666;">Made with ❤️ for farmers | Agricultural Assistant v1.0</p>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
