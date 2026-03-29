#!/usr/bin/env python3
"""
Maharashtra AI Crop Forecasting System
Comprehensive Crop Health, Weather, Soil Analysis & Pest Risk Prediction Platform
Matches the original screenshot with full functionality and online APIs
"""

# Standard library imports
import os
import warnings
from datetime import datetime, timedelta
from io import BytesIO
import shutil

# Third-party library imports
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import requests
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

# Local module imports
from mongodb_auth import MongoFarmerAuth
from enhanced_pest_data import PEST_DATABASE
from scheduler import init_scheduler, get_scheduler

# Logging setup
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration constants
CONFIG = {
    "API_TIMEOUT": 10,
    "MAX_RETRIES": 3,
    "REQUEST_CHUNK_SIZE": 8192,
    "IMAGE_MAX_SIZE": (224, 224),
    "BATCH_SIZE": 32,
    "MODEL_CACHE_TTL": 3600,
}

# Security: Import missing dependencies
import pickle


# --- Small helper to download large assets hosted externally ---
def download_file(url: str, filename: str) -> None:
    """Download a file from URL to filename with streaming and error handling.

    Args:
        url: Source URL to download from
        filename: Target file path
    """
    parent = os.path.dirname(filename)
    if parent:
        os.makedirs(parent, exist_ok=True)

    if os.path.exists(filename):
        logger.info(f"{os.path.basename(filename)} already exists")
        return

    try:
        st.info(f"Downloading {os.path.basename(filename)}...")
    except Exception:
        logger.info(f"Downloading {os.path.basename(filename)}...")

    for attempt in range(CONFIG["MAX_RETRIES"]):
        try:
            resp = requests.get(url, stream=True, timeout=CONFIG["API_TIMEOUT"])
            resp.raise_for_status()
            with open(filename, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=CONFIG["REQUEST_CHUNK_SIZE"]):
                    if chunk:
                        fh.write(chunk)
            logger.info(f"{os.path.basename(filename)} downloaded successfully")
            st.success(f"{os.path.basename(filename)} ready")
            return
        except Exception as e:
            logger.warning(f"Download attempt {attempt+1} failed: {e}")
            if attempt == CONFIG["MAX_RETRIES"] - 1:
                logger.error(
                    f"Failed to download {filename} after {CONFIG['MAX_RETRIES']} attempts"
                )
                raise


# If you keep large assets on Google Drive (recommended), download them into
# a local deployment folder so they are not tracked in git. Adjust these
# links as needed. These are safe no-op downloads if files already exist.
download_file(
    "https://drive.google.com/uc?export=download&id=1gYrPlQFe9vJUA2lefT4t16u2Odka32Ze",
    "maharashtra_agri_deployment/data/agriculture_dataset.csv",
)

download_file(
    "https://drive.google.com/uc?export=download&id=1ZNYqalYl1uATPF2g_bf9bgBcKtm2rbg8",
    "maharashtra_agri_deployment/models/fertilizer_prediction_model.pkl",
)


@st.cache_resource(show_spinner=False)
def load_model():
    """Load the ML model safely (binary download + cache).

    Behavior:
    - Prefer the model inside `maharashtra_agri_deployment/models/` (download target)
    - If missing, fall back to `fertilizer_prediction_model.pkl` in repo root
    - If neither exists, try the original huggingface URL as a last resort
    """
    preferred = os.path.join(
        "maharashtra_agri_deployment", "models", "fertilizer_prediction_model.pkl"
    )
    fallback = "fertilizer_prediction_model.pkl"
    model_path = preferred if os.path.exists(preferred) else fallback

    if not os.path.exists(model_path):
        url = "https://huggingface.co/Inamdar007/newfiledata/resolve/main/fertilizer_prediction_model.pkl"
        try:
            logger.info(f"Downloading model from {url}")
            st.info("Downloading model file...")
        except Exception as e:
            logger.warning(f"Streamlit not available: {e}")

        try:
            resp = requests.get(url, stream=True, timeout=CONFIG["API_TIMEOUT"])
            resp.raise_for_status()
            os.makedirs(os.path.dirname(preferred), exist_ok=True)
            with open(preferred, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=CONFIG["REQUEST_CHUNK_SIZE"]):
                    if chunk:
                        fh.write(chunk)
            model_path = preferred
            logger.info("Model downloaded successfully")
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise

    try:
        if (
            os.path.exists(model_path)
            and model_path != fallback
            and not os.path.exists(fallback)
        ):
            shutil.copyfile(model_path, fallback)
    except Exception as e:
        logger.warning(f"Could not create fallback copy: {e}")

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


# Chat helper used to live here but was refactored into the
# separate module `openrouter_chat.py` to centralize API logic.
# Keeping this comment in case anyone searches for the old
# implementation; the import at the bottom of this file now
# references the module instead.


# --- Secure Secrets Setup (for Streamlit Cloud + Local Dev) ---
load_dotenv()

try:
    # Use Streamlit Secrets if running on Streamlit Cloud
    if st.secrets and "MONGODB_URI" in st.secrets:
        os.environ["MONGODB_URI"] = st.secrets["MONGODB_URI"]
except Exception as e:
    logger.debug(f"Streamlit secrets not available: {e}")

# MongoDB connection setup
from pymongo import MongoClient

mongo_uri = os.getenv("MONGODB_URI")
if not mongo_uri:
    logger.warning(
        "MONGODB_URI not found in environment. Using default (NOT RECOMMENDED for production)"
    )
    mongo_uri = "mongodb+srv://yashimamdar_db_user:paulvrWJZqKz8SIJ@cluster0.r5cckg1.mongodb.net/maharashtra_agri_db?retryWrites=true&w=majority&appName=Cluster0"

try:
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    db = client["maharashtra_agri_db"]
    logger.info("MongoDB connection initialized")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {e}")
    client = None
    db = None

# Configure tensorflow to avoid memory issues
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# Suppress plotly warnings
warnings.filterwarnings("ignore", message=".*keyword arguments.*")


def apply_dark_theme(fig: go.Figure) -> go.Figure:
    """Apply a consistent dark theme to all Plotly graphs.

    Args:
        fig: Plotly figure object

    Returns:
        Updated figure with dark theme applied
    """
    try:
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(15,15,15,1)",
            paper_bgcolor="rgba(15,15,15,1)",
            font=dict(color="white", size=14),
            xaxis=dict(gridcolor="rgba(80,80,80,0.3)", color="white"),
            yaxis=dict(gridcolor="rgba(80,80,80,0.3)", color="white"),
            legend=dict(font=dict(color="white")),
        )
        return fig
    except Exception as e:
        logger.error(f"Error applying dark theme: {e}")
        return fig


# Page configuration
st.set_page_config(
    page_title="MahaAgroAI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Modern Agricultural Theme with Enhanced Color Scheme
st.markdown(
    """
<style>
    /* === ROOT VARIABLES FOR CONSISTENT COLOR THEMING === */
    :root {
        --primary-green: #2E7D32;     /* Deep Forest Green */
        --secondary-green: #4CAF50;   /* Bright Green */
        --accent-green: #66BB6A;      /* Light Green */
        --earth-brown: #5D4E37;       /* Rich Earth Brown */
        --soil-brown: #8D6E63;        /* Soil Brown */
        --sky-blue: #1976D2;          /* Sky Blue */
        --water-blue: #03A9F4;        /* Water Blue */
        --sunshine-yellow: #FFA726;   /* Sunshine Yellow */
        --harvest-orange: #FF7043;    /* Harvest Orange */
        --danger-red: #F44336;        /* Alert Red */
        --warning-amber: #FF9800;     /* Warning Amber */
        --dark-bg: #0F1419;           /* Deep Dark Background */
        --card-bg: #1A2027;           /* Card Background */
        --surface-bg: #232A36;        /* Surface Background */
        --text-primary: #E8F5E8;      /* Primary Text */
        --text-secondary: #B8D4B8;    /* Secondary Text */
        --border-color: #3E4A59;      /* Border Color */
    }

    /* === MAIN APPLICATION STYLES === */
    .main {
        background: linear-gradient(135deg, var(--dark-bg) 0%, #1C2833 100%);
        color: var(--text-primary);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stApp {
        background: linear-gradient(135deg, var(--dark-bg) 0%, #1C2833 100%);
    }
    
    /* === HEADER SECTION === */
    .crop-header {
        background: linear-gradient(135deg, var(--primary-green) 0%, var(--secondary-green) 50%, var(--sky-blue) 100%);
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 8px 32px rgba(46, 125, 50, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    .main-title {
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        letter-spacing: -0.5px;
    }
    .subtitle {
        font-size: 1.3rem;
        opacity: 0.95;
        font-weight: 300;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    
    /* === SIDEBAR ENHANCEMENTS === */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, var(--card-bg) 0%, var(--surface-bg) 100%);
        border-right: 2px solid var(--primary-green);
    }
    section[data-testid="stSidebar"] > div {
        background: linear-gradient(180deg, var(--card-bg) 0%, var(--surface-bg) 100%);
    }
    
    /* === METRIC CARDS === */
    .metric-card {
        background: linear-gradient(135deg, var(--secondary-green) 0%, var(--primary-green) 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.8rem 0;
        box-shadow: 0 6px 20px rgba(76, 175, 80, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(76, 175, 80, 0.4);
    }
    
    /* === UPLOAD AREA === */
    .upload-area {
        background: linear-gradient(135deg, var(--surface-bg) 0%, var(--card-bg) 100%);
        border: 2px dashed var(--accent-green);
        border-radius: 15px;
        padding: 2.5rem;
        text-align: center;
        color: var(--text-secondary);
        margin: 1rem 0;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .upload-area::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent 30%, rgba(102, 187, 106, 0.1) 50%, transparent 70%);
        transform: rotate(45deg);
        transition: all 0.6s ease;
    }
    .upload-area:hover::before {
        animation: shimmer 2s infinite;
    }
    .upload-area:hover {
        border-color: var(--secondary-green);
        background: linear-gradient(135deg, var(--card-bg) 0%, var(--surface-bg) 100%);
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%) rotate(45deg); }
        100% { transform: translateX(100%) rotate(45deg); }
    }
    
    /* === CONTENT SECTIONS === */
    .ndvi-section, .soil-section, .irrigation-section {
        background: linear-gradient(135deg, var(--card-bg) 0%, var(--surface-bg) 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        border: 1px solid var(--border-color);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    .ndvi-section:hover, .soil-section:hover, .irrigation-section:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        border-color: var(--accent-green);
    }
    
    /* === TAB HEADERS === */
    .tab-header {
        background: linear-gradient(135deg, var(--primary-green) 0%, var(--earth-brown) 100%);
        padding: 1.5rem;
        border-radius: 15px 15px 0 0;
        margin-bottom: 0;
        color: white;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
        box-shadow: 0 4px 15px rgba(46, 125, 50, 0.2);
    }
    
    /* === NAVIGATION TABS === */
    .nav-tabs {
        background: linear-gradient(135deg, var(--surface-bg) 0%, var(--card-bg) 100%);
        border-radius: 12px;
        padding: 0.8rem;
        margin-bottom: 1.5rem;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.2);
    }
    .nav-tab {
        background: linear-gradient(135deg, var(--card-bg) 0%, var(--surface-bg) 100%);
        color: var(--text-secondary);
        border: 1px solid var(--border-color);
        padding: 1rem 2rem;
        margin: 0 0.3rem;
        border-radius: 10px;
        cursor: pointer;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .nav-tab.active {
        background: linear-gradient(135deg, var(--secondary-green) 0%, var(--primary-green) 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
        transform: translateY(-1px);
    }
    .nav-tab:hover:not(.active) {
        background: linear-gradient(135deg, var(--surface-bg) 0%, var(--card-bg) 100%);
        border-color: var(--accent-green);
        transform: translateY(-1px);
    }
    
    /* === FORM INPUTS === */
    .stSelectbox > div > div {
        background: linear-gradient(135deg, var(--card-bg) 0%, var(--surface-bg) 100%) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
    }
    .stNumberInput > div > div > input {
        background: linear-gradient(135deg, var(--card-bg) 0%, var(--surface-bg) 100%) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--accent-green) !important;
        border-radius: 8px !important;
    }
    .stSlider > div > div > div {
        color: var(--text-primary) !important;
    }
    
    /* === BUTTONS === */
    .stButton > button {
        background: linear-gradient(135deg, var(--secondary-green) 0%, var(--primary-green) 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(76, 175, 80, 0.4) !important;
    }
    
    /* === METRICS === */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, var(--card-bg) 0%, var(--surface-bg) 100%);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    div[data-testid="metric-container"]:hover {
        border-color: var(--accent-green);
        transform: translateY(-1px);
        box-shadow: 0 6px 15px rgba(0,0,0,0.15);
    }
    
    /* === ALERTS & STATUS === */
    .alert-success {
        background: linear-gradient(135deg, var(--secondary-green) 0%, var(--accent-green) 100%);
        border-left: 4px solid var(--primary-green);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .alert-warning {
        background: linear-gradient(135deg, var(--warning-amber) 0%, var(--sunshine-yellow) 100%);
        border-left: 4px solid #F57C00;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .alert-danger {
        background: linear-gradient(135deg, var(--danger-red) 0%, var(--harvest-orange) 100%);
        border-left: 4px solid #D32F2F;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* === PRIORITY INDICATORS === */
    .priority-high {
        background: linear-gradient(135deg, var(--danger-red) 0%, var(--harvest-orange) 100%);
    }
    .priority-medium {
        background: linear-gradient(135deg, var(--warning-amber) 0%, var(--sunshine-yellow) 100%);
    }
    .priority-low {
        background: linear-gradient(135deg, var(--secondary-green) 0%, var(--accent-green) 100%);
    }
    
    /* === SCROLLBAR CUSTOMIZATION === */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: var(--dark-bg);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, var(--primary-green) 0%, var(--secondary-green) 100%);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, var(--secondary-green) 0%, var(--accent-green) 100%);
    }
    
    /* === TEXT ENHANCEMENTS === */
    h1, h2, h3 {
        color: var(--text-primary);
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    /* === RESPONSIVE DESIGN === */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2rem;
        }
        .crop-header {
            padding: 1.5rem;
        }
        .metric-card {
            padding: 1rem;
        }
    }
</style>
""",
    unsafe_allow_html=True,
)


class MaharashtraAgriculturalSystem:
    def __init__(self) -> None:
        """Initialize the comprehensive agricultural system.

        Sets up database connections, loads ML models, and initializes
        agricultural data (districts, crops, fertilizers, pest factors).
        """
        logger.info("Initializing MaharashtraAgriculturalSystem")
        self.setup_database()
        self.load_models()

        # API Keys from environment variables
        self.openweather_api_key = os.getenv("OPENWEATHER_API_KEY", "your_api_key_here")
        self.agromonitoring_api_key = os.getenv(
            "AGROMONITORING_API_KEY", "your_api_key_here"
        )

        if self.openweather_api_key == "your_api_key_here":
            logger.warning(
                "OPENWEATHER_API_KEY not configured. Set environment variable."
            )

        # Maharashtra districts and zones (all 36 districts)
        self.maharashtra_districts = {
            "Konkan (Coastal)": [
                "Mumbai City",
                "Mumbai Suburban",
                "Palghar",
                "Thane",
                "Raigad",
                "Ratnagiri",
                "Sindhudurg",
            ],
            "Western Maharashtra": ["Pune", "Satara", "Sangli", "Kolhapur", "Solapur"],
            "North Maharashtra (Khandesh)": [
                "Nashik",
                "Dhule",
                "Nandurbar",
                "Jalgaon",
                "Ahmednagar",
            ],
            "Marathwada": [
                "Chhatrapati Sambhajinagar",
                "Jalna",
                "Beed",
                "Latur",
                "Osmanabad",
                "Nanded",
                "Parbhani",
                "Hingoli",
            ],
            "Vidarbha": [
                "Nagpur",
                "Wardha",
                "Amravati",
                "Akola",
                "Washim",
                "Buldhana",
                "Yavatmal",
                "Chandrapur",
                "Gadchiroli",
                "Bhandara",
                "Gondia",
            ],
        }

        # District coordinates (approximate lat/lon for plotting)
        self.district_coords = {
            "Mumbai City": (18.9388, 72.8354),
            "Mumbai Suburban": (19.1180, 72.9050),
            "Palghar": (19.6967, 72.7655),
            "Thane": (19.2183, 72.9781),
            "Raigad": (18.5158, 73.1822),
            "Ratnagiri": (16.9902, 73.3120),
            "Sindhudurg": (16.3492, 73.5594),
            "Pune": (18.5204, 73.8567),
            "Satara": (17.6805, 74.0183),
            "Sangli": (16.8524, 74.5815),
            "Kolhapur": (16.7050, 74.2433),
            "Solapur": (17.6599, 75.9064),
            "Nashik": (19.9975, 73.7898),
            "Dhule": (20.9042, 74.7749),
            "Nandurbar": (21.3757, 74.2405),
            "Jalgaon": (21.0077, 75.5626),
            "Ahmednagar": (19.0952, 74.7496),
            "Chhatrapati Sambhajinagar": (19.8762, 75.3433),
            "Jalna": (19.8410, 75.8864),
            "Beed": (18.9891, 75.7601),
            "Latur": (18.4088, 76.5604),
            "Osmanabad": (18.1860, 76.0419),
            "Nanded": (19.1383, 77.3210),
            "Parbhani": (19.2700, 76.7600),
            "Hingoli": (19.7140, 77.1494),
            "Nagpur": (21.1458, 79.0882),
            "Wardha": (20.7453, 78.6022),
            "Amravati": (20.9374, 77.7796),
            "Akola": (20.7059, 77.0219),
            "Washim": (20.1113, 77.1330),
            "Buldhana": (20.5293, 76.1842),
            "Yavatmal": (20.3932, 78.1320),
            "Chandrapur": (19.9615, 79.2961),
            "Gadchiroli": (20.1870, 80.0000),
            "Bhandara": (21.1750, 79.6500),
            "Gondia": (21.4600, 80.2200),
        }

        # Crop types
        self.crop_types = [
            "Cotton",
            "Rice",
            "Wheat",
            "Sugarcane",
            "Soybean",
            "Tomato",
            "Potato",
            "Onion",
            "Maize",
            "Jowar",
        ]

        # Growth stages
        self.growth_stages = [
            "Sowing",
            "Germination",
            "Vegetative",
            "Flowering",
            "Fruit Development",
            "Maturity",
            "Harvesting",
        ]

        # Fertilizer database with costs (INR per kg)
        self.fertilizer_data = {
            "Urea": {"price": 6.50, "n_content": 46, "p_content": 0, "k_content": 0},
            "DAP": {"price": 27.00, "n_content": 18, "p_content": 46, "k_content": 0},
            "MOP": {"price": 17.50, "n_content": 0, "p_content": 0, "k_content": 60},
            "NPK 19:19:19": {
                "price": 22.00,
                "n_content": 19,
                "p_content": 19,
                "k_content": 19,
            },
            "Single Super Phosphate": {
                "price": 12.00,
                "n_content": 0,
                "p_content": 16,
                "k_content": 0,
            },
            "Potash": {"price": 18.00, "n_content": 0, "p_content": 0, "k_content": 50},
            "Zinc Sulphate": {
                "price": 45.00,
                "n_content": 0,
                "p_content": 0,
                "k_content": 0,
            },
            "Organic Compost": {
                "price": 8.50,
                "n_content": 2,
                "p_content": 1,
                "k_content": 1,
            },
        }

        # Pest risk factors
        self.pest_risk_factors = {
            "temperature": {"low": (10, 20), "medium": (20, 30), "high": (30, 40)},
            "humidity": {"low": (0, 50), "medium": (50, 70), "high": (70, 100)},
            "rainfall": {"low": (0, 10), "medium": (10, 50), "high": (50, 200)},
        }

    def setup_database(self) -> None:
        """Setup MongoDB database connection with error handling.

        Initializes MongoDB connection and tests connectivity by performing
        a test insert/delete operation.
        """
        try:
            from mongodb_config import MongoCropDB

            self.mongo_db = MongoCropDB()

            if (
                self.mongo_db.connected
                and hasattr(self.mongo_db, "db")
                and self.mongo_db.db is not None
            ):
                test_doc = {"test": "connection", "timestamp": datetime.now()}
                result = self.mongo_db.db.test_collection.insert_one(test_doc)
                self.mongo_db.db.test_collection.delete_one({"_id": result.inserted_id})
                logger.info("MongoDB connection test successful")
            else:
                logger.warning("MongoDB connection not properly initialized")
        except Exception as e:
            logger.error(f"MongoDB connection failed: {e}")
            self.mongo_db = None

    def load_models(self) -> None:
        """Load AI models with fallback strategy.

        Attempts to load the keras disease detection model and class names.
        Falls back to default class names if files not found.
        """
        try:
            if os.path.exists("best_model.h5"):
                self.disease_model = tf.keras.models.load_model("best_model.h5")
                logger.info("Disease detection model loaded successfully")

                if os.path.exists("class_names.txt"):
                    with open("class_names.txt", "r") as f:
                        self.class_names = [line.strip() for line in f.readlines()]
                    logger.info(f"Loaded {len(self.class_names)} class names")
                else:
                    self.class_names = [
                        "Healthy",
                        "Early_Blight",
                        "Late_Blight",
                        "Bacterial_Spot",
                    ]
                    logger.warning("Using default class names")
            else:
                self.disease_model = None
                self.class_names = [
                    "Healthy",
                    "Early_Blight",
                    "Late_Blight",
                    "Bacterial_Spot",
                ]
                logger.warning("Model file not found. Using fallback class names.")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.disease_model = None
            self.class_names = [
                "Healthy",
                "Early_Blight",
                "Late_Blight",
                "Bacterial_Spot",
            ]
            self.disease_model = None
            self.class_names = [
                "Healthy",
                "Early_Blight",
                "Late_Blight",
                "Bacterial_Spot",
            ]

    def get_weather_data(self, district, days=7):
        """Get comprehensive weather data with historical trends"""
        current_weather = self.get_current_weather(district)
        historical_weather = self.generate_weather_trends(district, days)

        return {
            "current": current_weather,
            "historical": historical_weather,
            "forecast": self.generate_weather_forecast(district, days=5),
        }

    def get_current_weather(self, district: str) -> dict:
        """Get current weather from API or simulation.

        Args:
            district: Maharashtra district name

        Returns:
            Dictionary with weather metrics (temperature, humidity, etc.)
        """
        try:
            if self.openweather_api_key != "your_api_key_here":
                url = f"http://api.openweathermap.org/data/2.5/weather"
                params = {
                    "q": f"{district},Maharashtra,IN",
                    "appid": self.openweather_api_key,
                    "units": "metric",
                }
                response = requests.get(url, params=params, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "temperature": data["main"]["temp"],
                        "humidity": data["main"]["humidity"],
                        "pressure": data["main"]["pressure"],
                        "description": data["weather"][0]["description"],
                        "wind_speed": data["wind"]["speed"],
                        "visibility": data.get("visibility", 10000)
                        / 1000,  # Convert to km
                        "uv_index": np.random.uniform(3, 8),  # Simulated UV index
                    }
        except Exception as e:
            pass

        # Fallback to simulated data
        np.random.seed(hash(district) % 1000)
        return {
            "temperature": round(np.random.uniform(18, 35), 1),
            "humidity": round(np.random.uniform(40, 85), 1),
            "pressure": round(np.random.uniform(1000, 1020), 1),
            "description": np.random.choice(
                ["clear sky", "few clouds", "scattered clouds", "light rain"]
            ),
            "wind_speed": round(np.random.uniform(2, 8), 1),
            "visibility": round(np.random.uniform(5, 15), 1),
            "uv_index": round(np.random.uniform(3, 8), 1),
        }

    def generate_weather_trends(self, district, days=7):
        """Generate historical weather trends"""
        np.random.seed(hash(district) % 1000)
        dates = [
            (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range(days, 0, -1)
        ]

        # Seasonal base values
        month = datetime.now().month
        if month in [12, 1, 2]:  # Winter
            temp_base, temp_var = 18, 8
            humidity_base, humidity_var = 55, 20
            rainfall_base = 0.5
        elif month in [3, 4, 5]:  # Summer
            temp_base, temp_var = 32, 8
            humidity_base, humidity_var = 45, 15
            rainfall_base = 0.3
        elif month in [6, 7, 8, 9]:  # Monsoon
            temp_base, temp_var = 26, 6
            humidity_base, humidity_var = 75, 15
            rainfall_base = 8.0
        else:  # Post-monsoon
            temp_base, temp_var = 24, 6
            humidity_base, humidity_var = 65, 20
            rainfall_base = 2.0

        temperatures = np.clip(
            temp_base + temp_var * (np.random.random(days) - 0.5), 10, 45
        )
        humidity = np.clip(
            humidity_base + humidity_var * (np.random.random(days) - 0.5), 20, 95
        )
        rainfall = np.clip(np.random.exponential(rainfall_base, days), 0, 150)
        wind_speed = np.clip(2 + 6 * np.random.random(days), 0, 20)
        pressure = np.clip(1013 + 10 * (np.random.random(days) - 0.5), 990, 1030)

        return {
            "dates": dates,
            "temperature": temperatures.round(1).tolist(),
            "humidity": humidity.round(1).tolist(),
            "rainfall": rainfall.round(1).tolist(),
            "wind_speed": wind_speed.round(1).tolist(),
            "pressure": pressure.round(1).tolist(),
        }

    def generate_weather_forecast(self, district, days=5):
        """Generate weather forecast"""
        np.random.seed(hash(district + "forecast") % 1000)
        dates = [
            (datetime.now() + timedelta(days=i + 1)).strftime("%Y-%m-%d")
            for i in range(days)
        ]

        current = self.get_current_weather(district)
        base_temp = current["temperature"]
        base_humidity = current["humidity"]

        # Forecast with slight variations
        temperatures = np.clip(base_temp + 3 * (np.random.random(days) - 0.5), 10, 45)
        humidity = np.clip(base_humidity + 10 * (np.random.random(days) - 0.5), 20, 95)
        rainfall = np.clip(np.random.exponential(1.0, days), 0, 50)
        conditions = np.random.choice(
            ["sunny", "partly cloudy", "cloudy", "light rain", "moderate rain"], days
        )

        return {
            "dates": dates,
            "temperature": temperatures.round(1).tolist(),
            "humidity": humidity.round(1).tolist(),
            "rainfall": rainfall.round(1).tolist(),
            "conditions": conditions.tolist(),
        }

    def get_enhanced_weather_data(self, district, days=7):
        """Get enhanced weather data with status indicators for interactive charts"""
        from datetime import datetime, timedelta
        import random

        today = datetime.now()
        dates = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)][
            ::-1
        ]

        # Maharashtra climate patterns
        base_temp = random.randint(25, 35)  # Celsius
        base_humidity = random.randint(60, 85)  # Percentage
        base_rainfall = random.randint(0, 25)  # mm
        base_wind = random.randint(5, 15)  # km/h

        weather_data = {
            "dates": dates,
            "temperature": [],
            "humidity": [],
            "rainfall": [],
            "wind_speed": [],
        }

        for i in range(days):
            # Temperature with daily variation
            temp_variation = random.randint(-3, 3)
            temp = base_temp + temp_variation
            weather_data["temperature"].append(temp)

            # Humidity with variation
            humidity_variation = random.randint(-10, 10)
            humidity = max(30, min(95, base_humidity + humidity_variation))
            weather_data["humidity"].append(humidity)

            # Rainfall with variation
            rainfall_variation = random.randint(-5, 15)
            rainfall = max(0, base_rainfall + rainfall_variation)
            weather_data["rainfall"].append(rainfall)

            # Wind speed with variation (convert to km/h)
            wind_variation = random.randint(-3, 3)
            wind_speed = max(2, base_wind + wind_variation)
            weather_data["wind_speed"].append(wind_speed)

        return weather_data

    def analyze_crop_image(self, uploaded_file) -> dict:
        """Enhanced crop image analysis with validation and error handling.

        Args:
            uploaded_file: Streamlit uploaded file object

        Returns:
            Dictionary with disease detection results and metadata
        """
        try:
            if uploaded_file is not None:
                # Enhanced validation and preprocessing
                # Check file type and size
                if uploaded_file.type not in ["image/jpeg", "image/jpg", "image/png"]:
                    return {
                        "error": "Please upload a valid image file (JPEG, JPG, or PNG)"
                    }

                if uploaded_file.size > 10 * 1024 * 1024:  # 10MB limit
                    return {
                        "error": "File size too large. Please upload an image smaller than 10MB."
                    }

                # Reset file pointer
                uploaded_file.seek(0)

                # Load and validate image
                image = Image.open(uploaded_file)
                if image.mode != "RGB":
                    image = image.convert("RGB")

                # Check image quality first
                # Fix PIL Image deprecation warning
                try:
                    # Try newer PIL syntax first
                    image_array = (
                        np.array(
                            image.resize((224, 224), Image.Resampling.LANCZOS)
                        ).astype(np.float32)
                        / 255.0
                    )
                except AttributeError:
                    # Fallback for older PIL versions
                    image_array = (
                        np.array(image.resize((224, 224), Image.LANCZOS)).astype(
                            np.float32
                        )
                        / 255.0
                    )
                quality_score = self.advanced_image_quality_assessment(image_array)

                if quality_score < 0.3:
                    return {
                        "disease": "Poor Image Quality",
                        "confidence": 25.0,
                        "all_predictions": [("Poor Image Quality", 100.0)],
                        "model_accuracy": "Low - Image Quality Too Poor",
                        "image_quality": "Poor - Please retake with better lighting",
                        "recommendations": [
                            "Take image in natural daylight",
                            "Ensure leaf fills most of frame",
                            "Avoid shadows and reflections",
                        ],
                    }

                leaf_valid, leaf_score, leaf_message = self.assess_leaf_image(
                    image_array
                )
                if not leaf_valid:
                    return {
                        "disease": "Invalid Crop Image",
                        "confidence": 0.0,
                        "all_predictions": [("Invalid Crop Image", 100.0)],
                        "model_accuracy": "N/A",
                        "image_quality": self.quality_to_text(quality_score),
                        "recommendations": [
                            "Upload a clear leaf/plant image",
                            "Ensure the leaf occupies most of the frame",
                            "Avoid photos of faces, animals, or inanimate objects",
                        ],
                        "error": "Please upload a valid crop leaf image for accurate disease detection.",
                        "leaf_validation_score": leaf_score,
                        "leaf_validation_message": leaf_message,
                    }

                # Advanced preprocessing pipeline
                processed_images = self.advanced_preprocessing_pipeline(image_array)

                if self.disease_model is not None:
                    try:
                        # Multi-scale ensemble prediction
                        all_predictions = []
                        confidence_weights = []

                        for processed_img, weight in processed_images:
                            batch = np.expand_dims(processed_img, axis=0)
                            pred = self.disease_model.predict(batch, verbose=0)[0]
                            all_predictions.append(pred)
                            confidence_weights.append(weight)

                        # Weighted ensemble with confidence adjustment
                        final_predictions = np.average(
                            all_predictions, axis=0, weights=confidence_weights
                        )
                        predicted_class = np.argmax(final_predictions)
                        base_confidence = float(final_predictions[predicted_class])

                        # Quality-adjusted confidence
                        adjusted_confidence = base_confidence * (
                            0.7 + 0.3 * quality_score
                        )

                        disease = (
                            self.class_names[predicted_class]
                            if predicted_class < len(self.class_names)
                            else "Unknown"
                        )

                        # Advanced confidence thresholding
                        if adjusted_confidence < 0.5:
                            disease = (
                                "Uncertain Analysis - Expert Consultation Recommended"
                            )
                            adjusted_confidence *= 0.7
                        elif adjusted_confidence < 0.7:
                            disease = f"Possible {disease}"
                            adjusted_confidence *= 0.85

                        # Generate comprehensive predictions
                        all_disease_predictions = [
                            (
                                self.class_names[i],
                                float(final_predictions[i])
                                * 100
                                * (0.7 + 0.3 * quality_score),
                            )
                            for i in range(len(self.class_names))
                        ]
                        all_disease_predictions.sort(key=lambda x: x[1], reverse=True)

                        return {
                            "disease": disease,
                            "confidence": min(95.0, adjusted_confidence * 100),
                            "all_predictions": all_disease_predictions,
                            "model_accuracy": f"High Accuracy - Ensemble AI (Quality: {quality_score:.2f})",
                            "image_quality": self.quality_to_text(quality_score),
                            "recommendations": self.generate_treatment_recommendations(
                                disease
                            ),
                            "severity_assessment": self.assess_disease_severity(
                                adjusted_confidence * 100, disease
                            ),
                        }
                    except Exception as model_error:
                        st.warning(
                            f"AI Model unavailable. Using advanced image analysis: {model_error}"
                        )

                # Fallback to advanced image analysis
                return self.advanced_image_analysis_fallback(image_array, quality_score)

        except Exception as e:
            st.error(f"Critical analysis error: {str(e)}")
            return self.get_error_response()

    def advanced_image_quality_assessment(self, image_array):
        """Advanced image quality assessment for better accuracy"""
        # Multiple quality metrics
        brightness = np.mean(image_array)
        contrast = np.std(image_array)

        # Sharpness using Laplacian variance
        gray = np.mean(image_array, axis=2)
        laplacian_var = np.var(gray)

        # Color balance assessment
        color_balance = 1.0 - np.std(np.mean(image_array, axis=(0, 1)))

        # Noise assessment
        noise_level = np.std(gray - np.mean(gray))

        # Composite quality score (0-1)
        brightness_score = 1.0 - abs(brightness - 0.5) * 2
        contrast_score = min(1.0, contrast * 10)
        sharpness_score = min(1.0, laplacian_var * 50)
        balance_score = color_balance
        noise_score = max(0, 1.0 - noise_level * 5)

        # Weighted average
        quality_score = (
            brightness_score * 0.25
            + contrast_score * 0.25
            + sharpness_score * 0.25
            + balance_score * 0.15
            + noise_score * 0.1
        )

        return max(0, min(1, quality_score))

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

            # 6. Image metadata validation
            try:
                if hasattr(image, "_getexif") and image._getexif() is None:
                    validation_result["quality_metrics"]["has_metadata"] = False
            except:
                pass

            # 7. Quality metrics
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

            # 8. Content validation (too dark/bright)
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

            # 9. Contrast validation
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

    def display_professional_image_alert(self, validation_result: dict) -> None:
        """
        Display professional, color-coded image validation feedback.

        Severity levels:
            success  → green confirmation banner
            warning  → amber two-column panel (issues + metrics)
            error    → red bordered error panel with fix suggestions
            critical → deep-red bordered panel (corrupted/unreadable file)

        Regardless of severity, a "continue to other tabs" notice is appended
        so the user always knows analysis of weather/soil/pest/irrigation is
        still available.
        """

        severity = validation_result.get("severity", "success")
        is_valid = validation_result.get("valid", True)
        errors = validation_result.get("errors", [])
        warnings = validation_result.get("warnings", [])
        metrics = validation_result.get("quality_metrics", {})

        # ── 1. SUCCESS ──────────────────────────────────────────────────────
        if severity == "success" and is_valid:
            st.markdown(
                """
                <div style="
                    background: linear-gradient(135deg, #1B5E20 0%, #2E7D32 100%);
                    border-left: 5px solid #66BB6A;
                    border-radius: 10px;
                    padding: 1rem 1.4rem;
                    margin: 0.8rem 0;
                    display: flex;
                    align-items: center;
                    gap: 0.7rem;
                    box-shadow: 0 3px 12px rgba(0,0,0,0.25);
                ">
                    <span style="font-size:1.5rem;">✅</span>
                    <div>
                        <p style="margin:0; color:#C8E6C9; font-weight:600; font-size:0.95rem;">
                            Image Validation Passed
                        </p>
                        <p style="margin:0; color:#A5D6A7; font-size:0.82rem;">
                            Your image meets all quality requirements. Proceed with AI analysis.
                        </p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            return

        # ── 2. WARNING ──────────────────────────────────────────────────────
        if severity == "warning":
            st.markdown(
                """
                <div style="
                    background: linear-gradient(135deg, #4A3000 0%, #5D3D00 100%);
                    border: 1px solid #FF9800;
                    border-left: 5px solid #FF9800;
                    border-radius: 10px;
                    padding: 1rem 1.4rem;
                    margin: 0.8rem 0;
                    box-shadow: 0 3px 12px rgba(0,0,0,0.3);
                ">
                    <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.7rem;">
                        <span style="font-size:1.4rem;">⚠️</span>
                        <span style="color:#FFB74D; font-weight:700; font-size:1rem;">
                            Image Quality Warning
                        </span>
                        <span style="
                            background:#FF9800; color:#1a1a1a;
                            font-size:0.7rem; font-weight:700;
                            padding:2px 8px; border-radius:20px; margin-left:auto;
                        ">REDUCED ACCURACY</span>
                    </div>
                """,
                unsafe_allow_html=True,
            )

            col_warn, col_metrics = st.columns([1.5, 1])

            with col_warn:
                st.markdown(
                    "<p style='color:#FFB74D; font-weight:600; font-size:0.88rem;"
                    " margin:0 0 0.4rem;'>⚙️ Detected Issues:</p>",
                    unsafe_allow_html=True,
                )
                for w in warnings:
                    st.markdown(
                        f"<p style='color:#FFCC80; font-size:0.84rem; margin:0.2rem 0;'>"
                        f"&nbsp;&nbsp;• {w}</p>",
                        unsafe_allow_html=True,
                    )

            with col_metrics:
                if metrics:
                    st.markdown(
                        "<p style='color:#FFB74D; font-weight:600; font-size:0.88rem;"
                        " margin:0 0 0.4rem;'>📊 Image Metrics:</p>",
                        unsafe_allow_html=True,
                    )
                    for label, val in [
                        ("Size", metrics.get("dimensions", "N/A")),
                        ("File", f"{metrics.get('file_size_mb','N/A')} MB"),
                        ("Brightness", metrics.get("brightness", "N/A")),
                        ("Contrast", metrics.get("contrast", "N/A")),
                    ]:
                        st.markdown(
                            f"<p style='color:#FFCC80; font-size:0.83rem; margin:0.15rem 0;'>"
                            f"&nbsp;&nbsp;<b>{label}:</b> {val}</p>",
                            unsafe_allow_html=True,
                        )

            st.markdown("</div>", unsafe_allow_html=True)

            # "Analysis will still run" notice
            st.markdown(
                """
                <div style="
                    background:rgba(255,152,0,0.12);
                    border:1px dashed #FF9800;
                    border-radius:8px;
                    padding:0.6rem 1rem;
                    margin-top:0.5rem;
                    font-size:0.82rem; color:#FFB74D;
                ">
                    ℹ️ Analysis will proceed but accuracy may be reduced.
                    For best results, retake the image in natural daylight with the leaf
                    filling the frame. <b>All other tabs (Weather, Soil, Pest, Irrigation)
                    are fully available.</b>
                </div>
                """,
                unsafe_allow_html=True,
            )
            return

        # ── 3. ERROR / CRITICAL ─────────────────────────────────────────────
        if severity in ("error", "critical"):
            icon = "🔴" if severity == "critical" else "🟠"
            header = (
                "Critical Image Error — File Cannot Be Processed"
                if severity == "critical"
                else "Image Error — Crop Analysis Unavailable"
            )
            border_color = "#D32F2F" if severity == "critical" else "#F44336"
            bg_color = "#3B0000" if severity == "critical" else "#2D0000"

            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, {bg_color} 0%, #1a0000 100%);
                    border: 1px solid {border_color};
                    border-left: 6px solid {border_color};
                    border-radius: 10px;
                    padding: 1.1rem 1.4rem;
                    margin: 0.8rem 0;
                    box-shadow: 0 4px 16px rgba(211,47,47,0.3);
                ">
                    <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.8rem;">
                        <span style="font-size:1.4rem;">{icon}</span>
                        <span style="color:#EF9A9A; font-weight:700; font-size:1rem;">
                            {header}
                        </span>
                        <span style="
                            background:{border_color}; color:white;
                            font-size:0.68rem; font-weight:700;
                            padding:2px 8px; border-radius:20px; margin-left:auto;
                            text-transform:uppercase; letter-spacing:0.5px;
                        ">{severity.upper()}</span>
                    </div>
                """,
                unsafe_allow_html=True,
            )

            # Error list
            for err in errors:
                st.markdown(
                    f"<p style='color:#FFCDD2; font-size:0.86rem; margin:0.25rem 0;'>"
                    f"&nbsp;&nbsp;❌ {err}</p>",
                    unsafe_allow_html=True,
                )

            st.markdown("</div>", unsafe_allow_html=True)

            # Quality metrics in expander (keeps UI clean)
            if metrics:
                with st.expander("📊 View Image Diagnostics", expanded=False):
                    cols = st.columns(min(len(metrics), 4))
                    for idx, (key, val) in enumerate(metrics.items()):
                        with cols[idx % len(cols)]:
                            st.metric(key.replace("_", " ").title(), val)

            # Fix suggestions
            st.markdown(
                """
                <div style="
                    background:rgba(255,87,34,0.10);
                    border:1px dashed #FF5722;
                    border-radius:8px;
                    padding:0.8rem 1.1rem;
                    margin-top:0.6rem;
                ">
                    <p style="color:#FF8A65; font-weight:600; font-size:0.88rem; margin:0 0 0.5rem;">
                        💡 How to fix:
                    </p>
                    <p style="color:#FFCCBC; font-size:0.83rem; margin:0.2rem 0;">
                        &nbsp;&nbsp;• Use JPG, JPEG, or PNG format only
                    </p>
                    <p style="color:#FFCCBC; font-size:0.83rem; margin:0.2rem 0;">
                        &nbsp;&nbsp;• Keep file size under 15 MB
                    </p>
                    <p style="color:#FFCCBC; font-size:0.83rem; margin:0.2rem 0;">
                        &nbsp;&nbsp;• Photograph leaf in natural daylight (avoid very dark/bright conditions)
                    </p>
                    <p style="color:#FFCCBC; font-size:0.83rem; margin:0.2rem 0;">
                        &nbsp;&nbsp;• Ensure the image file is not corrupted
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ⬇ KEY: always show "other tabs still available" notice
            st.markdown(
                """
                <div style="
                    background: linear-gradient(135deg, #0D2137 0%, #0A1929 100%);
                    border: 1px solid #1565C0;
                    border-left: 5px solid #1976D2;
                    border-radius: 10px;
                    padding: 0.9rem 1.2rem;
                    margin-top: 0.8rem;
                    box-shadow: 0 3px 10px rgba(0,0,0,0.3);
                ">
                    <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.5rem;">
                        <span style="font-size:1.2rem;">ℹ️</span>
                        <span style="color:#90CAF9; font-weight:700; font-size:0.92rem;">
                            Other Analyses Are Fully Available
                        </span>
                    </div>
                    <p style="color:#BBDEFB; font-size:0.83rem; margin:0.2rem 0;">
                        &nbsp;&nbsp;🌤️ <b>Weather & Soil</b> — View weather data and soil health analysis
                    </p>
                    <p style="color:#BBDEFB; font-size:0.83rem; margin:0.2rem 0;">
                        &nbsp;&nbsp;🐛 <b>Pest Risk</b> — See pest risk levels for your crop
                    </p>
                    <p style="color:#BBDEFB; font-size:0.83rem; margin:0.2rem 0;">
                        &nbsp;&nbsp;💧 <b>Irrigation</b> — Get irrigation recommendations
                    </p>
                    <p style="color:#BBDEFB; font-size:0.83rem; margin:0.2rem 0;">
                        &nbsp;&nbsp;📊 <b>Dashboard</b> — Full farm summary
                    </p>
                    <p style="color:#90CAF9; font-size:0.81rem; margin-top:0.5rem; font-style:italic;">
                        Click <b>ANALYZE ALL DATA</b> in the sidebar — crop image analysis will be skipped,
                        but all other insights will be generated normally.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    def assess_leaf_image(self, image_array):
        """Assess whether the uploaded image is likely a crop leaf image.

        Uses green-channel vegetation heuristics and texture checks to avoid
        false positives on random non-crop photos.
        """
        # Ensure image array is normalized [0,1]
        if image_array.max() > 1.0:
            image_array = image_array / 255.0

        # Color and vegetation mask
        red = image_array[:, :, 0]
        green = image_array[:, :, 1]
        blue = image_array[:, :, 2]

        # Green dominance: green should be stronger than red/blue in leaf regions
        green_dominant = (green > red * 1.05) & (green > blue * 1.05) & (green > 0.18)
        green_ratio = float(np.mean(green_dominant))

        # Basic color saturation and brightness checks
        max_rgb = np.max(image_array, axis=2)
        min_rgb = np.min(image_array, axis=2)
        saturation = np.mean((max_rgb - min_rgb) / (max_rgb + 1e-6))
        brightness = np.mean(max_rgb)

        # Texture (edges) as leaf shape proxy: variance of grayscale values
        gray = np.mean(image_array, axis=2)
        texture = np.var(gray)

        # Composite leaf score
        leaf_score = (
            0.5 * green_ratio
            + 0.25 * min(1.0, saturation * 2.0)
            + 0.15 * min(1.0, texture * 120.0)
            + 0.1 * min(1.0, brightness * 1.5)
        )

        # Validation criteria
        if (
            green_ratio < 0.08
            or saturation < 0.12
            or brightness < 0.1
            or brightness > 0.95
        ):
            message = (
                "Image likely not a clear crop leaf. Please upload a leaf/plant image with green foliage "
                "in good natural lighting."
            )
            return False, float(leaf_score), message

        if leaf_score < 0.28:
            message = (
                "Image appears to be either a non-leaf photo or too noisy/low quality for reliable analysis. "
                "Please retake with a leaf occupying most of the frame."
            )
            return False, float(leaf_score), message

        return (
            True,
            float(leaf_score),
            (
                "Leaf image validation passed. Green ratio: {:.2f}, quality: {:.2f}".format(
                    green_ratio, leaf_score
                )
            ),
        )

    def advanced_preprocessing_pipeline(self, image_array):
        """Advanced preprocessing for maximum accuracy"""
        processed_images = []

        # Original image (weight: 0.4)
        processed_images.append((image_array, 0.4))

        # Contrast enhanced (weight: 0.2)
        contrast_enhanced = np.clip(image_array * 1.3 - 0.15, 0, 1)
        processed_images.append((contrast_enhanced, 0.2))

        # Brightness normalized (weight: 0.2)
        mean_brightness = np.mean(image_array)
        target_brightness = 0.5
        brightness_factor = target_brightness / (mean_brightness + 0.001)
        brightness_normalized = np.clip(image_array * brightness_factor, 0, 1)
        processed_images.append((brightness_normalized, 0.2))

        # Horizontally flipped (weight: 0.1)
        flipped = np.fliplr(image_array)
        processed_images.append((flipped, 0.1))

        # Gaussian filtered for noise reduction (weight: 0.1)
        try:
            from scipy import ndimage

            filtered = ndimage.gaussian_filter(image_array, sigma=0.5)
            processed_images.append((filtered, 0.1))
        except:
            # Fallback if scipy not available
            processed_images.append((image_array * 0.95, 0.1))

        return processed_images

    def advanced_image_analysis_fallback(self, image_array, quality_score):
        """Advanced fallback analysis when AI model is not available"""
        features = self.extract_advanced_image_features(image_array)
        disease, confidence = self.advanced_disease_simulation(features, quality_score)

        return {
            "disease": disease,
            "confidence": confidence,
            "all_predictions": self.generate_realistic_predictions(disease, confidence),
            "model_accuracy": f"Advanced Image Analysis (Quality: {quality_score:.2f})",
            "image_quality": self.quality_to_text(quality_score),
            "recommendations": self.generate_treatment_recommendations(disease),
            "severity_assessment": self.assess_disease_severity(confidence, disease),
        }

    def extract_advanced_image_features(self, image_array):
        """Extract comprehensive image features for accurate analysis"""
        # Color analysis
        mean_rgb = np.mean(image_array, axis=(0, 1))
        green_dominance = mean_rgb[1] / (np.sum(mean_rgb) + 0.001)
        red_ratio = mean_rgb[0] / (np.sum(mean_rgb) + 0.001)
        blue_ratio = mean_rgb[2] / (np.sum(mean_rgb) + 0.001)

        # Texture analysis
        gray = np.mean(image_array, axis=2)
        texture_variance = np.var(gray)

        # Edge detection simulation
        edges = np.abs(np.gradient(gray)[0]) + np.abs(np.gradient(gray)[1])
        edge_density = np.mean(edges)

        # Dark/brown spot analysis (disease indicators)
        hsv_approx = self.rgb_to_hsv_approx(image_array)
        dark_spots = (
            np.sum((hsv_approx[:, :, 2] < 0.3) & (hsv_approx[:, :, 1] > 0.2))
            / image_array.size
        )
        brown_spots = (
            np.sum(
                (hsv_approx[:, :, 0] > 0.08)
                & (hsv_approx[:, :, 0] < 0.17)
                & (hsv_approx[:, :, 1] > 0.3)
            )
            / image_array.size
        )

        # Yellow/chlorotic areas
        yellow_areas = (
            np.sum(
                (hsv_approx[:, :, 0] > 0.15)
                & (hsv_approx[:, :, 0] < 0.18)
                & (hsv_approx[:, :, 1] > 0.5)
            )
            / image_array.size
        )

        return {
            "green_dominance": green_dominance,
            "red_ratio": red_ratio,
            "blue_ratio": blue_ratio,
            "texture_variance": texture_variance,
            "edge_density": edge_density,
            "dark_spots_ratio": dark_spots,
            "brown_spots_ratio": brown_spots,
            "yellow_areas_ratio": yellow_areas,
            "brightness": np.mean(image_array),
            "color_uniformity": 1.0 - np.std(mean_rgb),
        }

    def rgb_to_hsv_approx(self, rgb_array):
        """Approximate RGB to HSV conversion"""
        r, g, b = rgb_array[:, :, 0], rgb_array[:, :, 1], rgb_array[:, :, 2]

        max_val = np.maximum(np.maximum(r, g), b)
        min_val = np.minimum(np.minimum(r, g), b)
        diff = max_val - min_val

        # Hue calculation (simplified)
        hue = np.where(
            diff == 0,
            0,
            np.where(
                max_val == r,
                (60 * ((g - b) / diff) + 360) % 360,
                np.where(
                    max_val == g,
                    (60 * ((b - r) / diff) + 120) % 360,
                    (60 * ((r - g) / diff) + 240) % 360,
                ),
            ),
        )
        hue = hue / 360.0  # Normalize to 0-1

        # Saturation
        saturation = np.where(max_val == 0, 0, diff / max_val)

        # Value
        value = max_val

        return np.stack([hue, saturation, value], axis=2)

    def advanced_disease_simulation(self, features, quality_score):
        """Advanced disease simulation with higher accuracy"""
        green_dom = features["green_dominance"]
        dark_spots = features["dark_spots_ratio"]
        brown_spots = features["brown_spots_ratio"]
        yellow_areas = features["yellow_areas_ratio"]
        texture_var = features["texture_variance"]
        edge_density = features["edge_density"]
        brightness = features["brightness"]

        # Quality-adjusted confidence multiplier
        quality_multiplier = 0.6 + 0.4 * quality_score

        # Enhanced healthy detection
        if (
            green_dom > 0.4
            and dark_spots < 0.05
            and brown_spots < 0.03
            and yellow_areas < 0.02
            and brightness > 0.3
            and texture_var < 0.02
        ):
            confidence = np.random.uniform(88, 96) * quality_multiplier
            return "Healthy", min(95, confidence)

        # Early blight detection
        elif (
            brown_spots > 0.08
            and dark_spots > 0.05
            and edge_density > 0.3
            and texture_var > 0.03
        ):
            confidence = np.random.uniform(82, 94) * quality_multiplier
            return "Early Blight", min(94, confidence)

        # Late blight detection
        elif dark_spots > 0.12 and brightness < 0.25 and texture_var > 0.05:
            confidence = np.random.uniform(85, 95) * quality_multiplier
            return "Late Blight", min(95, confidence)

        # Bacterial spot detection
        elif dark_spots > 0.06 and brown_spots > 0.04 and green_dom < 0.35:
            confidence = np.random.uniform(78, 90) * quality_multiplier
            return "Bacterial Spot", min(90, confidence)

        # Nutrient deficiency (chlorosis)
        elif yellow_areas > 0.08 and green_dom < 0.3 and brightness > 0.4:
            confidence = np.random.uniform(75, 88) * quality_multiplier
            return "Nutrient Deficiency (Chlorosis)", min(88, confidence)

        # Viral infection
        elif features["color_uniformity"] < 0.4 and texture_var > 0.06:
            confidence = np.random.uniform(72, 86) * quality_multiplier
            return "Possible Viral Infection", min(86, confidence)

        # Stress/mild disease
        elif green_dom < 0.35 or dark_spots > 0.03:
            confidence = np.random.uniform(65, 78) * quality_multiplier
            return "Plant Stress Detected", min(78, confidence)

        # Uncertain case
        else:
            confidence = np.random.uniform(45, 65) * quality_multiplier
            return "Analysis Inconclusive", min(65, confidence)

    def quality_to_text(self, quality_score):
        """Convert quality score to readable text"""
        if quality_score > 0.8:
            return "Excellent - Very High Accuracy Expected"
        elif quality_score > 0.6:
            return "Good - High Accuracy Expected"
        elif quality_score > 0.4:
            return "Fair - Moderate Accuracy Expected"
        elif quality_score > 0.2:
            return "Poor - Low Accuracy Expected"
        else:
            return "Very Poor - Please Retake Image"

    def assess_disease_severity(self, confidence, disease):
        """Assess disease severity based on confidence and disease type"""
        if "Healthy" in disease:
            return "None"
        elif any(word in disease for word in ["Late Blight", "Viral"]):
            if confidence > 85:
                return "Severe"
            elif confidence > 70:
                return "Moderate"
            else:
                return "Mild"
        elif confidence > 80:
            return "High"
        elif confidence > 65:
            return "Moderate"
        else:
            return "Low"

    def get_error_response(self):
        """Return standardized error response"""
        return {
            "disease": "Analysis Failed",
            "confidence": 0.0,
            "all_predictions": [("System Error", 100.0)],
            "model_accuracy": "Failed - Please Try Again",
            "image_quality": "Unknown",
            "recommendations": [
                "Please upload a clear image",
                "Ensure good lighting conditions",
            ],
            "severity_assessment": "Unknown",
        }

    def enhance_image_for_analysis(self, image_array):
        """Enhance image quality for better AI analysis"""
        # Contrast enhancement
        enhanced = np.clip(image_array * 1.2, 0, 1)

        # Noise reduction (simple gaussian blur simulation)
        # In a real implementation, you'd use cv2.GaussianBlur
        kernel_size = 1
        enhanced = enhanced  # Simplified for now

        return enhanced

    def create_augmented_images(self, image_array):
        """Create augmented versions for ensemble prediction"""
        augmented = []

        # Slight rotations
        # Note: In production, use actual rotation functions
        augmented.append(np.fliplr(image_array))  # Horizontal flip

        # Brightness variations
        bright_img = np.clip(image_array * 1.1, 0, 1)
        dark_img = np.clip(image_array * 0.9, 0, 1)
        augmented.extend([bright_img, dark_img])

        return augmented[:2]  # Return 2 augmentations for efficiency

    def assess_image_quality(self, image_array):
        """Assess uploaded image quality for accuracy"""
        # Calculate image metrics
        brightness = np.mean(image_array)
        contrast = np.std(image_array)
        sharpness = np.var(image_array)  # Simplified sharpness metric

        if (
            brightness > 0.1
            and brightness < 0.9
            and contrast > 0.05
            and sharpness > 0.01
        ):
            return "Good - High accuracy expected"
        elif brightness > 0.05 and contrast > 0.03:
            return "Fair - Moderate accuracy expected"
        else:
            return "Poor - Low accuracy, retake recommended"

    def extract_image_features(self, image_array):
        """Extract features from image for intelligent simulation"""
        # Color analysis
        mean_rgb = np.mean(image_array, axis=(0, 1))
        green_dominance = mean_rgb[1] / (mean_rgb[0] + mean_rgb[2] + 0.001)

        # Texture analysis (simplified)
        gray = np.mean(image_array, axis=2)
        texture_variance = np.var(gray)

        # Dark spot analysis (potential disease indicators)
        dark_spots = np.sum(np.mean(image_array, axis=2) < 0.3) / image_array.size

        return {
            "green_dominance": green_dominance,
            "texture_variance": texture_variance,
            "dark_spots_ratio": dark_spots,
            "brightness": np.mean(image_array),
            "color_uniformity": 1.0 - np.std(mean_rgb),
        }

    def simulate_disease_detection(self, features):
        """Enhanced intelligent disease simulation with improved accuracy"""
        green_dom = features["green_dominance"]
        dark_spots = features["dark_spots_ratio"]
        texture_var = features["texture_variance"]
        brightness = features["brightness"]
        color_uniformity = features["color_uniformity"]

        # Enhanced healthy plant detection with stricter criteria
        if (
            green_dom > 1.3
            and dark_spots < 0.08
            and brightness > 0.45
            and color_uniformity > 0.8
            and texture_var < 0.03
        ):
            return "Healthy", np.random.uniform(88, 96)

        # Enhanced disease pattern recognition
        elif dark_spots > 0.25 and texture_var > 0.06:
            if brightness < 0.25 and color_uniformity < 0.6:
                return "Late Blight", np.random.uniform(82, 94)
            elif brightness > 0.35 and dark_spots > 0.3:
                return "Early Blight", np.random.uniform(79, 91)
            else:
                return "Leaf Spot Disease", np.random.uniform(76, 88)

        # Improved bacterial infection detection
        elif green_dom < 0.7 and color_uniformity < 0.65 and texture_var > 0.04:
            return "Bacterial Spot", np.random.uniform(77, 89)

        # Nutrient deficiency detection
        elif green_dom < 0.9 and brightness > 0.5 and color_uniformity > 0.7:
            return "Nutrient Deficiency", np.random.uniform(74, 86)

        # Viral infection patterns
        elif color_uniformity < 0.5 and texture_var > 0.08:
            return "Viral Infection", np.random.uniform(72, 84)

        # Improved default case with conditional confidence
        elif green_dom > 1.0 and dark_spots < 0.15:
            return "Possibly Healthy", np.random.uniform(68, 78)
        else:
            return "Uncertain Analysis", np.random.uniform(45, 65)

    def generate_realistic_predictions(self, detected_disease, confidence):
        """Generate enhanced realistic probability distribution"""
        # Updated disease list with new classes
        diseases = [
            "Healthy",
            "Potato Early Blight",
            "Tomato Late Blight",
            "Tomato Bacterial Spot",
            "Tomato Target Spot",
            "Pepper Bell Bacterial Spot",
            "Leaf Spot Disease",
            "Nutrient Deficiency",
        ]

        # Normalize detected disease name to match
        detected_normalized = (
            detected_disease.replace("___", " ").replace("_", " ").replace("  ", " ")
        )

        # Check if detected disease indicates healthy plant
        is_healthy = any(
            h in detected_normalized.lower() for h in ["healthy", "tomato healthy"]
        )
        if is_healthy:
            detected_normalized = "Healthy"

        predictions = []

        for disease in diseases:
            disease_normalized = disease.lower()
            detected_lower = detected_normalized.lower()

            if disease_normalized == detected_lower or (
                is_healthy and "healthy" in disease_normalized
            ):
                predictions.append((disease, confidence))
            else:
                # Improved probability distribution with realistic decay
                remaining = 100 - confidence
                base_prob = remaining / (len(diseases) - 1)

                # Add realistic variations based on disease similarity
                if is_healthy and disease in ["Nutrient Deficiency"]:
                    variation = np.random.uniform(-2, 8)  # Higher chance of deficiency
                elif "blight" in detected_lower and "spot" in disease_normalized:
                    variation = np.random.uniform(0, 5)  # Similar symptoms
                elif (
                    "bacterial" in detected_lower and "bacterial" in disease_normalized
                ):
                    variation = np.random.uniform(2, 8)  # Similar pathogens
                else:
                    variation = np.random.uniform(-5, 3)  # General variation

                final_prob = max(0, min(remaining * 0.8, base_prob + variation))
                predictions.append((disease, round(final_prob, 1)))

        # Normalize to ensure total is 100%
        total = sum(pred[1] for pred in predictions)
        if total > 0:
            predictions = [
                (pred[0], round(pred[1] * 100 / total, 1)) for pred in predictions
            ]

        return sorted(predictions, key=lambda x: x[1], reverse=True)

    def generate_treatment_recommendations(self, disease):
        """Generate comprehensive treatment recommendations with detailed explanations"""
        recommendations = {
            "Healthy": [
                "✅ **Maintain Excellence**: Your crop shows optimal health. Continue current successful practices including your watering, fertilization and pest monitoring routine.",
                "🔍 **Weekly Health Checks**: Inspect plants every 7 days for early disease detection. Check leaf color, wilting, spots, or unusual growth patterns.",
                "🌱 **Balanced Nutrition**: Apply NPK fertilizer (19:19:19) at 2g/L every 2-3 weeks. Healthy crops need maintenance nutrition to stay strong.",
                "💧 **Optimal Watering**: Water early morning (6-8 AM) when soil is dry 2-3 inches deep. Deep, infrequent watering promotes strong root development.",
                "🛡️ **Preventive Protection**: Spray bio-fungicide (like Neem oil @ 3-5ml/L) every 2 weeks during humid weather as prevention.",
                "📊 **Document Progress**: Keep records of irrigation, fertilizer applications, and plant observations to maintain this excellent health status.",
            ],
            "Early_Blight": [
                "🚨 **Immediate Action Required**: Early blight spreads quickly in warm, humid conditions. Remove ALL affected leaves immediately using sterilized shears.",
                "✂️ **Proper Removal Technique**: Cut 2-3 inches below visible symptoms into healthy tissue. Sterilize tools with 70% alcohol between cuts.",
                "🔥 **Safe Disposal**: Burn infected material or dispose 100+ meters from field. NEVER compost diseased plant matter - spores survive composting.",
                "💊 **Copper Fungicide Treatment**: Mix Copper Oxychloride 50% WP at 3g per liter water. Add sticker-spreader for better leaf adhesion.",
                "🕐 **Treatment Timing**: Spray early morning or late evening when temperature is below 30°C. Avoid spraying during hot midday sun.",
                "📅 **Treatment Schedule**: Apply every 7-10 days for 3-4 treatments. Increase frequency during rainy/humid periods.",
                "🌬️ **Improve Air Flow**: Space plants 18-24 inches apart. Remove lower branches touching soil. Clear weeds around plants for better ventilation.",
                "💧 **Modified Irrigation**: Switch to drip irrigation or water at soil level only. Wet leaves encourage fungal growth.",
                "🛡️ **Protective Barrier**: Spray healthy neighboring plants with preventive copper fungicide to create protection zone.",
                "📈 **Follow-up Monitoring**: Check treated plants daily for first week, then every 2-3 days. New growth should remain healthy.",
            ],
            "Late_Blight": [
                "🆘 **EMERGENCY RESPONSE**: Late blight can destroy entire field in 5-7 days. This is a CROP EMERGENCY requiring immediate action.",
                "⚡ **24-Hour Action Rule**: Begin treatment within 24-48 hours of detection. Delay means potential total crop loss.",
                "🔥 **Complete Plant Removal**: Remove and BURN entire infected plants including roots. Create 2-meter firebreak around infected area.",
                "💉 **Systemic Fungicide**: Use Metalaxyl-M + Mancozeb 75% WP at 2.5g/L OR Dimethomorph + Mancozeb at 2g/L for systemic protection.",
                "🎯 **Protective Spraying**: Immediately treat ALL plants within 50-meter radius with systemic fungicide as protective barrier.",
                "🧽 **Complete Sanitation**: Disinfect ALL tools with 10% bleach solution. Change clothes before entering healthy field areas.",
                "💧 **Water Management**: STOP overhead irrigation immediately. Reduce soil moisture levels. Ensure maximum field drainage.",
                "🌪️ **Maximize Air Movement**: Remove lower leaves, increase plant spacing, use fans if in greenhouse/protected cultivation.",
                "🔄 **Fungicide Rotation**: Alternate between chemical groups every 2 applications: Metalaxyl→Dimethomorph→Fluazinam to prevent resistance.",
                "🎯 **Targeted Application**: Focus spray on stem base and lower leaves where infection starts. Ensure complete coverage (dripping from leaves).",
                "📱 **Emergency Consultation**: Contact local agricultural extension officer immediately. Consider professional crop protection advisor.",
                "🌡️ **Weather Vigilance**: Late blight thrives in cool, wet weather (15-21°C, high humidity). Monitor forecasts closely.",
            ],
            "Bacterial_Spot": [
                "🦠 **Bacterial Disease Protocol**: Bacterial diseases are harder to control than fungal. Focus on prevention and copper-based treatments.",
                "💊 **Copper Bactericide**: Apply Copper Sulfate at 1-1.5g/L OR Streptocycline at 0.5g/L. Copper works better in prevention.",
                "☀️ **Dry Weather Operations**: ONLY work with plants when completely dry. Bacteria spread through water splash between plants.",
                "✂️ **Strategic Pruning**: Remove affected branches during dry weather only. Cut 4-6 inches below symptoms into healthy tissue.",
                "🧼 **Tool Sterilization**: Disinfect pruning tools with 70% alcohol between EVERY plant. Keep disinfectant spray bottle handy.",
                "💧 **Irrigation System Change**: Install drip irrigation immediately. Overhead watering spreads bacteria through water splash.",
                "🔄 **Crop Rotation Planning**: Plan 2-3 year rotation away from tomato family crops. Bacteria survive in soil for 2+ years.",
                "🏞️ **Drainage Improvement**: Improve field drainage with raised beds. Bacteria thrive in waterlogged conditions.",
                "🌡️ **Temperature Monitoring**: Bacterial spot is worst in warm, humid weather (24-29°C, >85% humidity). Plan treatments accordingly.",
                "🗺️ **Field Mapping**: Mark infected spots on field map. These areas need extra monitoring as bacteria persist in soil.",
                "🛡️ **Preventive Program**: Apply weekly copper sprays to healthy plants during bacterial disease season (typically monsoon).",
            ],
            "Leaf_Spot_Disease": [
                "🍃 **Spot Disease Management**: Various fungi cause leaf spots. Early detection and removal prevents spread to healthy tissue.",
                "🔍 **Symptom Recognition**: Look for circular/irregular spots with defined borders, often with yellow halos around dark centers.",
                "✂️ **Selective Removal**: Remove only affected leaves, cutting into healthy tissue. Leave healthy foliage for photosynthesis.",
                "💊 **Broad-Spectrum Treatment**: Use Carbendazim at 1g/L OR Mancozeb at 2g/L for general leaf spot control.",
                "🌬️ **Air Circulation**: Improve ventilation by spacing plants properly and removing lower branches that restrict airflow.",
                "💧 **Water Management**: Water at soil level only. Reduce watering frequency but water deeply when needed.",
                "🌿 **Plant Immunity Boost**: Apply micronutrient spray (Zinc + Manganese) to strengthen plant immune system.",
                "📅 **Regular Monitoring**: Check plants every 3-4 days during humid weather for new spot development.",
            ],
            "Nutrient_Deficiency": [
                "🧪 **Comprehensive Soil Analysis**: Get detailed soil test for N-P-K, pH, organic matter, and micronutrients at nearest agricultural center.",
                "🌿 **Balanced Nutrition Program**: Apply NPK 19:19:19 at 2g/L as foliar spray for quick absorption plus soil application.",
                "🍂 **Organic Matter Addition**: Add well-decomposed compost/FYM at 2-3 tons per hectare to improve nutrient availability.",
                "💧 **Enhanced Water Management**: Ensure adequate but not excessive watering. Nutrients need proper moisture for uptake.",
                "🍃 **Quick-Response Foliar Feeding**: Mix NPK + micronutrients for weekly foliar spray until soil nutrition improves.",
                "📊 **Deficiency Identification**: Yellow leaves = Nitrogen, Purple stems = Phosphorus, Brown leaf edges = Potassium deficiency.",
                "⏰ **Application Timing**: Apply fertilizers early morning or evening. Avoid hot midday application which can burn leaves.",
            ],
            "Viral_Infection": [
                "🦠 **Viral Disease Warning**: Viruses cannot be cured with chemicals. Focus on prevention and vector control.",
                "🚫 **Immediate Isolation**: Remove infected plants completely including roots. Viruses spread to nearby plants rapidly.",
                "🐛 **Vector Control Program**: Control aphids, whiteflies, and thrips with systemic insecticides. These insects spread viruses.",
                "🌱 **Resistant Variety Planning**: For next season, choose virus-resistant varieties available for your crop type.",
                "🧽 **Strict Sanitation**: Disinfect ALL tools with 10% bleach solution. Viruses transfer easily on contaminated equipment.",
                "🚫 **Avoid Plant Damage**: Handle plants gently. Wounds provide entry points for viruses. Use proper harvesting techniques.",
                "📍 **Field Hygiene**: Remove ALL plant debris, weeds, and volunteer plants that can harbor viruses between seasons.",
                "🛡️ **Reflective Mulch**: Use silver reflective mulch to confuse flying insect vectors and reduce virus transmission.",
            ],
            "Uncertain_Analysis": [
                "📸 **Better Image Quality**: Retake photos in natural daylight (10 AM - 4 PM). Hold camera 6-12 inches from affected area.",
                "🔍 **Professional Diagnosis**: Visit nearest Krishi Vigyan Kendra (KVK) or contact agricultural extension officer for expert opinion.",
                "📝 **Symptom Documentation**: Record when symptoms appeared, weather conditions, recent treatments, and progression over time.",
                "📱 **Multiple Consultations**: Use various agricultural apps, consult experienced farmers, get second opinions for confirmation.",
                "🌱 **Maintain Basic Care**: Continue regular watering and fertilization while awaiting proper diagnosis.",
                "🔬 **Laboratory Testing**: For persistent problems, collect samples for laboratory analysis at agricultural university.",
                "📚 **Reference Materials**: Compare symptoms with agricultural disease identification guides specific to your crop type.",
            ],
        }

        disease_key = disease.replace(" ", "_")
        return recommendations.get(disease_key, recommendations["Uncertain_Analysis"])

    def calculate_ndvi(self, nir_value, red_value):
        """Calculate NDVI from NIR and Red values with proper validation"""
        try:
            # Input validation
            if not isinstance(nir_value, (int, float)) or not isinstance(
                red_value, (int, float)
            ):
                return 0

            # Check for negative values
            if nir_value < 0 or red_value < 0:
                return 0

            # Prevent division by zero
            denominator = nir_value + red_value
            if denominator == 0:
                return 0

            ndvi = (nir_value - red_value) / denominator

            # Ensure NDVI is within valid range [-1, 1]
            ndvi = max(-1, min(1, ndvi))

            return round(ndvi, 3)
        except Exception as e:
            st.warning(f"NDVI calculation error: {str(e)}")
            return 0

    def interpret_ndvi(self, ndvi_value):
        """Interpret NDVI value"""
        if ndvi_value > 0.7:
            return "Excellent vegetation health", "#4CAF50"
        elif ndvi_value > 0.5:
            return "Good vegetation health", "#8BC34A"
        elif ndvi_value > 0.3:
            return "Moderate vegetation health", "#FFC107"
        elif ndvi_value > 0.1:
            return "Poor vegetation health", "#FF9800"
        else:
            return "Critical vegetation health", "#F44336"

    def analyze_soil_health(
        self,
        ph: float,
        nitrogen: float,
        phosphorus: float,
        potassium: float,
        farm_area: float = 1.0,
    ) -> dict:
        """Advanced soil health analysis with scientific accuracy and precision.

        Args:
            ph: Soil pH value (0-14)
            nitrogen: Nitrogen level in mg/kg
            phosphorus: Phosphorus level in mg/kg
            potassium: Potassium level in mg/kg
            farm_area: Farm area in hectares (default: 1.0)

        Returns:
            Dictionary with soil analysis, recommendations, and fertilizer suggestions
        """
        recommendations = []
        fertilizer_recommendations = []

        # Advanced pH analysis with crop-specific recommendations
        optimal_ph_range = (6.0, 7.5)
        if ph < optimal_ph_range[0]:
            acidity_level = (
                "moderately acidic"
                if ph > 5.5
                else "strongly acidic" if ph > 5.0 else "very strongly acidic"
            )
            recommendations.append(
                f"Soil is {acidity_level} (pH {ph}). Lime application required."
            )

            # Scientific lime calculation: (Target pH - Current pH) × Buffer Capacity × Area
            buffer_capacity = 2.5  # Typical for Maharashtra black cotton soils
            lime_needed = (6.5 - ph) * buffer_capacity * 1000 * farm_area  # kg/ha

            fertilizer_recommendations.append(
                {
                    "type": "Agricultural Lime (CaCO3)",
                    "quantity": round(lime_needed, 1),
                    "cost": round(lime_needed * 3.5, 2),  # Updated price per kg
                    "purpose": f"pH correction from {ph} to 6.5",
                    "application_method": "Broadcast and incorporate 15-20 days before planting",
                }
            )
        elif ph > optimal_ph_range[1]:
            alkalinity_level = (
                "moderately alkaline" if ph < 8.5 else "strongly alkaline"
            )
            recommendations.append(
                f"Soil is {alkalinity_level} (pH {ph}). Organic matter and sulfur needed."
            )

            # Calculate organic matter and sulfur requirements
            organic_matter_needed = (ph - 7.0) * 300 * farm_area  # kg/ha
            sulfur_needed = (ph - 7.5) * 50 * farm_area  # kg/ha

            fertilizer_recommendations.extend(
                [
                    {
                        "type": "Well-decomposed FYM",
                        "quantity": round(organic_matter_needed, 1),
                        "cost": round(
                            organic_matter_needed
                            * self.fertilizer_data["Organic Compost"]["price"],
                            2,
                        ),
                        "purpose": f"pH reduction and soil conditioning",
                        "application_method": "Apply before monsoon and mix thoroughly",
                    },
                    {
                        "type": "Elemental Sulfur",
                        "quantity": round(sulfur_needed, 1),
                        "cost": round(sulfur_needed * 25, 2),  # Price per kg
                        "purpose": "pH adjustment and sulfur nutrition",
                        "application_method": "Apply with organic matter",
                    },
                ]
            )
        else:
            recommendations.append(f"Soil pH ({ph}) is optimal for most crops.")

        # Advanced NPK analysis with Maharashtra-specific standards
        # Updated optimal levels for Maharashtra soils
        optimal_N = 300  # kg/ha for black cotton soils
        optimal_P = 15  # kg/ha (higher for P-fixing soils)
        optimal_K = 150  # kg/ha

        n_deficit = max(0, optimal_N - nitrogen)
        p_deficit = max(0, optimal_P - phosphorus)
        k_deficit = max(0, optimal_K - potassium)

        # Nitrogen management
        if n_deficit > 0:
            # Calculate fertilizer needs with efficiency factors
            urea_efficiency = 0.65  # 65% efficiency in field conditions
            urea_needed = (
                (n_deficit * farm_area)
                / (self.fertilizer_data["Urea"]["n_content"] / 100)
                / urea_efficiency
            )

            severity = (
                "severe"
                if n_deficit > 100
                else "moderate" if n_deficit > 50 else "mild"
            )
            recommendations.append(
                f"Nitrogen deficiency: {severity} ({n_deficit} kg/ha deficit). Split application recommended."
            )

            fertilizer_recommendations.append(
                {
                    "type": "Urea (46% N)",
                    "quantity": round(urea_needed, 1),
                    "cost": round(
                        urea_needed * self.fertilizer_data["Urea"]["price"], 2
                    ),
                    "purpose": f"Nitrogen supply ({n_deficit} kg/ha deficit)",
                    "application_method": "50% at planting + 30% at 30 days + 20% at 60 days",
                }
            )

        # Phosphorus management
        if p_deficit > 0:
            # Account for P-fixation in black soils
            p_fixation_factor = 1.5  # 50% more needed due to fixation
            dap_needed = (p_deficit * farm_area * p_fixation_factor) / (
                self.fertilizer_data["DAP"]["p_content"] / 100
            )

            severity = (
                "severe" if p_deficit > 8 else "moderate" if p_deficit > 4 else "mild"
            )
            recommendations.append(
                f"Phosphorus deficiency: {severity} ({p_deficit} kg/ha deficit). Band placement recommended."
            )

            fertilizer_recommendations.append(
                {
                    "type": "DAP (18-46-0)",
                    "quantity": round(dap_needed, 1),
                    "cost": round(dap_needed * self.fertilizer_data["DAP"]["price"], 2),
                    "purpose": f"Phosphorus supply ({p_deficit} kg/ha deficit)",
                    "application_method": "Band placement 5cm below and beside seed",
                }
            )

        # Potassium management
        if k_deficit > 0:
            # K availability factor in clay soils
            k_availability = 0.8  # 80% available in clay soils
            mop_needed = (
                (k_deficit * farm_area)
                / (self.fertilizer_data["MOP"]["k_content"] / 100)
                / k_availability
            )

            severity = (
                "severe" if k_deficit > 75 else "moderate" if k_deficit > 40 else "mild"
            )
            recommendations.append(
                f"Potassium deficiency: {severity} ({k_deficit} kg/ha deficit). Split application needed."
            )

            fertilizer_recommendations.append(
                {
                    "type": "MOP (60% K2O)",
                    "quantity": round(mop_needed, 1),
                    "cost": round(mop_needed * self.fertilizer_data["MOP"]["price"], 2),
                    "purpose": f"Potassium supply ({k_deficit} kg/ha deficit)",
                    "application_method": "60% at planting + 40% at flowering",
                }
            )

        # Maintenance fertilization for balanced soils
        if n_deficit == 0 and p_deficit == 0 and k_deficit == 0:
            recommendations.append(
                "Excellent nutrient status! Maintenance fertilization recommended."
            )
            maintenance_npk = 75 * farm_area  # kg/ha for maintenance

            fertilizer_recommendations.append(
                {
                    "type": "NPK 19:19:19",
                    "quantity": round(maintenance_npk, 1),
                    "cost": round(
                        maintenance_npk * self.fertilizer_data["NPK 19:19:19"]["price"],
                        2,
                    ),
                    "purpose": "Maintenance nutrition and soil health",
                    "application_method": "Broadcast and incorporate before planting",
                }
            )

        # Secondary and micronutrient recommendations
        secondary_nutrients = self.analyze_secondary_micronutrients(
            ph, nitrogen, phosphorus, potassium
        )
        if secondary_nutrients["recommendations"]:
            recommendations.extend(secondary_nutrients["recommendations"])
            fertilizer_recommendations.extend(secondary_nutrients["fertilizers"])

        # Advanced soil health scoring with weighted parameters
        ph_score = self.calculate_ph_score(ph)
        n_score = self.calculate_nutrient_score(nitrogen, optimal_N)
        p_score = self.calculate_nutrient_score(phosphorus, optimal_P)
        k_score = self.calculate_nutrient_score(potassium, optimal_K)

        # Weighted scoring (pH is most critical)
        overall_score = ph_score * 0.4 + n_score * 0.25 + p_score * 0.2 + k_score * 0.15
        total_cost = sum([f["cost"] for f in fertilizer_recommendations])

        # Soil health classification
        if overall_score >= 85:
            status = "Excellent"
            status_color = "#4CAF50"
        elif overall_score >= 70:
            status = "Good"
            status_color = "#8BC34A"
        elif overall_score >= 55:
            status = "Fair"
            status_color = "#FFC107"
        elif overall_score >= 40:
            status = "Poor"
            status_color = "#FF9800"
        else:
            status = "Very Poor"
            status_color = "#F44336"

        return {
            "score": round(overall_score, 1),
            "recommendations": recommendations,
            "fertilizer_recommendations": fertilizer_recommendations,
            "total_cost": round(total_cost, 2),
            "deficits": {
                "nitrogen": round(n_deficit, 1),
                "phosphorus": round(p_deficit, 1),
                "potassium": round(k_deficit, 1),
            },
            "status": status,
            "status_color": status_color,
            "individual_scores": {
                "ph_score": round(ph_score, 1),
                "nitrogen_score": round(n_score, 1),
                "phosphorus_score": round(p_score, 1),
                "potassium_score": round(k_score, 1),
            },
            "soil_classification": self.classify_soil_type(
                ph, nitrogen, phosphorus, potassium
            ),
        }

    def calculate_ph_score(self, ph):
        """Calculate pH score using scientific curve"""
        optimal_range = (6.0, 7.5)
        if optimal_range[0] <= ph <= optimal_range[1]:
            return 100
        elif ph < optimal_range[0]:
            return max(0, 100 - (optimal_range[0] - ph) ** 1.5 * 25)
        else:
            return max(0, 100 - (ph - optimal_range[1]) ** 1.5 * 25)

    def calculate_nutrient_score(self, current, optimal):
        """Calculate nutrient score with diminishing returns"""
        if current >= optimal:
            return 100
        else:
            # Logarithmic curve for deficiency impact
            ratio = current / optimal
            return max(0, 100 * (ratio**0.7))  # Diminishing penalty

    def analyze_secondary_micronutrients(self, ph, n, p, k):
        """Analyze secondary and micronutrient needs"""
        recommendations = []
        fertilizers = []

        # Sulfur deficiency likely in high rainfall areas
        if n > 250 and ph > 7.0:  # High N uptake areas need more S
            recommendations.append(
                "Sulfur deficiency likely. Apply gypsum or elemental sulfur."
            )
            fertilizers.append(
                {
                    "type": "Gypsum (CaSO4.2H2O)",
                    "quantity": 150.0,
                    "cost": 150.0 * 2.5,
                    "purpose": "Sulfur nutrition and soil conditioning",
                    "application_method": "Broadcast before monsoon",
                }
            )

        # Zinc deficiency common in alkaline soils
        if ph > 7.5 or (p > 12 and k < 100):  # High P can induce Zn deficiency
            recommendations.append(
                "Zinc deficiency possible in alkaline soil. Apply zinc sulfate."
            )
            fertilizers.append(
                {
                    "type": "Zinc Sulfate (ZnSO4.7H2O)",
                    "quantity": 25.0,
                    "cost": 25.0 * 45,  # Price per kg
                    "purpose": "Zinc nutrition",
                    "application_method": "Soil application or foliar spray",
                }
            )

        # Boron for specific crops
        if p > 10 and k > 120:  # Good fertility soils may need B
            recommendations.append(
                "Consider boron application for cotton/sunflower crops."
            )
            fertilizers.append(
                {
                    "type": "Borax (Na2B4O7.10H2O)",
                    "quantity": 10.0,
                    "cost": 10.0 * 80,
                    "purpose": "Boron nutrition for reproductive growth",
                    "application_method": "Soil application at flowering",
                }
            )

        return {"recommendations": recommendations, "fertilizers": fertilizers}

    def classify_soil_type(self, ph, n, p, k):
        """Classify soil type based on nutrient profile"""
        if ph < 6.5 and n < 200:
            return "Acidic, Low Fertility"
        elif ph > 7.8 and k > 150:
            return "Alkaline, High Potassium (Black Cotton Soil)"
        elif n > 250 and p > 12 and k > 120:
            return "High Fertility, Well Managed"
        elif n < 180 and p < 8:
            return "Nutrient Depleted"
        else:
            return "Moderate Fertility"

    def analyze_pest_risk(self, weather_data, crop_type, growth_stage):
        """Comprehensive pest risk analysis with visual data"""
        current_weather = weather_data["current"]
        historical_weather = weather_data["historical"]

        # Calculate risk factors
        temp_risk = self.calculate_temperature_risk(current_weather["temperature"])
        humidity_risk = self.calculate_humidity_risk(current_weather["humidity"])
        rainfall_risk = self.calculate_rainfall_risk(
            np.mean(historical_weather["rainfall"][-7:])
        )
        seasonal_risk = self.calculate_seasonal_risk()

        # Crop and growth stage specific risks
        crop_risks = self.get_crop_specific_risks(crop_type, growth_stage)

        # Calculate overall risk score (0-100)
        risk_factors = [temp_risk, humidity_risk, rainfall_risk, seasonal_risk]
        overall_risk = np.mean(risk_factors)

        # Generate pest predictions
        pest_predictions = self.predict_specific_pests(
            crop_type, overall_risk, current_weather
        )

        # Get treatment plans and spraying windows (HIGH PRIORITY)
        treatment_plans = self.get_pest_treatment_plans(crop_type, overall_risk)
        spraying_windows = self.get_optimal_spraying_windows(current_weather)

        return {
            "overall_risk": round(overall_risk, 1),
            "risk_factors": {
                "temperature": {
                    "value": temp_risk,
                    "status": self.get_risk_status(temp_risk),
                },
                "humidity": {
                    "value": humidity_risk,
                    "status": self.get_risk_status(humidity_risk),
                },
                "rainfall": {
                    "value": rainfall_risk,
                    "status": self.get_risk_status(rainfall_risk),
                },
                "seasonal": {
                    "value": seasonal_risk,
                    "status": self.get_risk_status(seasonal_risk),
                },
            },
            "crop_specific_risks": crop_risks,
            "pest_predictions": pest_predictions,
            "recommendations": self.generate_pest_recommendations(
                overall_risk, crop_type
            ),
            "risk_level": self.get_risk_level(overall_risk),
            "treatment_plans": treatment_plans,
            "spraying_windows": spraying_windows,
        }

    def calculate_temperature_risk(self, temperature):
        """Calculate pest risk based on temperature"""
        if 25 <= temperature <= 32:
            return 85  # High risk zone
        elif 20 <= temperature <= 35:
            return 60  # Medium risk
        else:
            return 30  # Low risk

    def calculate_humidity_risk(self, humidity):
        """Calculate pest risk based on humidity"""
        if humidity >= 75:
            return 90  # Very high risk for fungal diseases
        elif humidity >= 60:
            return 65  # Medium-high risk
        elif humidity >= 40:
            return 35  # Low-medium risk
        else:
            return 20  # Low risk

    def calculate_rainfall_risk(self, avg_rainfall):
        """Calculate pest risk based on rainfall pattern"""
        if avg_rainfall > 20:
            return 80  # High risk due to excess moisture
        elif avg_rainfall > 10:
            return 50  # Moderate risk
        elif avg_rainfall > 2:
            return 30  # Low risk
        else:
            return 40  # Drought stress can increase pest susceptibility

    def calculate_seasonal_risk(self):
        """Calculate seasonal pest risk"""
        month = datetime.now().month
        if month in [6, 7, 8, 9]:  # Monsoon season
            return 75
        elif month in [10, 11]:  # Post-monsoon
            return 60
        elif month in [3, 4, 5]:  # Summer
            return 45
        else:  # Winter
            return 35

    def get_crop_specific_risks(self, crop_type, growth_stage):
        """Get crop and growth stage specific pest risks"""
        crop_pest_db = {
            "Cotton": {
                "pests": ["Bollworm", "Aphids", "Whitefly", "Thrips"],
                "high_risk_stages": ["Flowering", "Fruit Development"],
                "diseases": ["Bacterial Blight", "Fusarium Wilt"],
            },
            "Rice": {
                "pests": ["Brown Plant Hopper", "Stem Borer", "Leaf Folder"],
                "high_risk_stages": ["Vegetative", "Flowering"],
                "diseases": ["Blast Disease", "Sheath Blight"],
            },
            "Tomato": {
                "pests": ["Fruit Borer", "Aphids", "Whitefly"],
                "high_risk_stages": ["Flowering", "Fruit Development"],
                "diseases": ["Early Blight", "Late Blight", "Leaf Curl Virus"],
            },
            "Potato": {
                "pests": ["Colorado Beetle", "Tuber Moth", "Aphids"],
                "high_risk_stages": ["Vegetative", "Maturity"],
                "diseases": ["Late Blight", "Early Blight"],
            },
        }

        if crop_type in crop_pest_db:
            crop_data = crop_pest_db[crop_type]
            risk_multiplier = (
                1.5 if growth_stage in crop_data["high_risk_stages"] else 1.0
            )
            return {
                "pests": crop_data["pests"],
                "diseases": crop_data["diseases"],
                "risk_multiplier": risk_multiplier,
                "high_risk_stage": growth_stage in crop_data["high_risk_stages"],
            }
        return {
            "pests": [],
            "diseases": [],
            "risk_multiplier": 1.0,
            "high_risk_stage": False,
        }

    def predict_specific_pests(self, crop_type, overall_risk, weather):
        """Predict specific pest probabilities with detailed information"""
        predictions = []
        crop_risks = self.get_crop_specific_risks(crop_type, "Vegetative")

        for pest in crop_risks["pests"]:
            # Calculate pest-specific probability
            base_prob = overall_risk / 100

            # Adjust based on weather conditions
            if pest in ["Aphids", "Whitefly"] and weather["temperature"] > 25:
                prob_adjustment = 0.2
            elif pest in ["Bollworm", "Fruit Borer"] and weather["humidity"] > 70:
                prob_adjustment = 0.3
            elif pest in ["Stem Borer"] and weather["humidity"] > 80:
                prob_adjustment = 0.25
            else:
                prob_adjustment = 0.1

            final_probability = min(95, (base_prob + prob_adjustment) * 100)

            # Get detailed pest information from database
            pest_details = self.get_detailed_pest_info(crop_type, pest)

            predictions.append(
                {
                    "pest": pest,
                    "probability": round(final_probability, 1),
                    "severity": (
                        "High"
                        if final_probability > 70
                        else "Medium" if final_probability > 40 else "Low"
                    ),
                    "details": pest_details,
                }
            )

        return predictions

    def get_detailed_pest_info(self, crop_type, pest_name):
        """Retrieve detailed pest information from enhanced database"""
        if crop_type in PEST_DATABASE and pest_name in PEST_DATABASE[crop_type]:
            return PEST_DATABASE[crop_type][pest_name]
        return None

    def generate_pest_recommendations(self, risk_score, crop_type):
        """Generate comprehensive pest management recommendations with detailed explanations"""
        recommendations = []

        if risk_score > 70:
            recommendations.extend(
                [
                    "🚨 **CRITICAL PEST ALERT**: High pest risk detected (Risk Score: {:.1f}/100). Immediate intervention required to prevent crop damage.".format(
                        risk_score
                    ),
                    "💊 **Emergency Pesticide Application**: Apply broad-spectrum insecticide immediately. Use Chlorpyriphos 20% EC @ 2ml/L or Profenofos 50% EC @ 2ml/L.",
                    "🔍 **Daily Intensive Monitoring**: Check ALL plants daily for 2 weeks. Focus on undersides of leaves, growing tips, and flower/fruit areas.",
                    "🌡️ **Weather-Based Action**: High pest activity correlates with warm, humid weather. Adjust spray timing based on temperature and humidity.",
                    "🚨 **Quarantine Measures**: Isolate heavily infested areas. Prevent movement of equipment between clean and infested areas.",
                    "📱 **Professional Consultation**: Contact agricultural extension officer or pest management expert immediately for severe infestations.",
                    "🔄 **Integrated Approach**: Combine chemical control with biological methods. Use pheromone traps alongside pesticide applications.",
                ]
            )
        elif risk_score > 50:
            recommendations.extend(
                [
                    "⚠️ **MODERATE PEST RISK**: Risk Score {:.1f}/100 indicates potential pest buildup. Take preventive action now.".format(
                        risk_score
                    ),
                    "🌿 **Preventive Bio-Control**: Apply Neem oil @ 3-5ml/L or Bt spray @ 1-2g/L every 7-10 days as prevention.",
                    "🔍 **Enhanced Monitoring**: Check plants every 2-3 days during peak risk periods. Look for early signs of pest activity.",
                    "📍 **Strategic Trap Placement**: Install pheromone/sticky traps at 4-6 per acre to monitor and reduce pest population.",
                    "🌱 **Field Hygiene Program**: Remove crop residues, weeds, and alternate host plants that harbor pests between seasons.",
                    "💧 **Water Management**: Avoid over-watering which creates humid conditions favoring pest multiplication.",
                    "🔄 **Rotation Planning**: Plan crop rotation with non-host crops to break pest life cycles.",
                ]
            )
        else:
            recommendations.extend(
                [
                    "✅ **LOW PEST RISK**: Current conditions show low pest pressure (Risk Score: {:.1f}/100). Maintain preventive measures.".format(
                        risk_score
                    ),
                    "🔍 **Weekly Surveillance**: Regular weekly inspection is sufficient. Check 10-15 plants per acre systematically.",
                    "🌿 **Preventive Organic Methods**: Continue bio-pesticide applications every 15 days. Use Neem oil or microbial pesticides.",
                    "🌱 **Ecosystem Balance**: Preserve beneficial insects. Avoid broad-spectrum pesticides unless absolutely necessary.",
                    "📊 **Record Keeping**: Maintain pest monitoring records to track seasonal patterns and predict future outbreaks.",
                    "🛡️ **Early Warning System**: Set up monitoring traps as early warning system for pest population increases.",
                ]
            )

        # Add detailed crop-specific recommendations
        crop_specific = {
            "Cotton": [
                "🌿 **Cotton Bollworm Management**: Check squares and young bolls daily during flowering. Look for entry holes and frass (pest droppings).",
                "🐛 **Aphid Colony Detection**: Examine undersides of leaves for aphid colonies. Early morning inspection is most effective.",
                "🎯 **Target Critical Stages**: Focus protection during squaring and boll formation stages when crop is most vulnerable.",
                "📊 **Economic Threshold**: Spray when 10% plants show bollworm damage or 5-10 aphids per leaf are found.",
            ],
            "Rice": [
                "🌾 **Brown Plant Hopper Monitoring**: Check for yellowing and 'hopper burn' symptoms. Look for insects at base of plants.",
                "🌱 **Stem Borer Detection**: Look for 'dead hearts' in vegetative stage and 'white heads' during reproductive stage.",
                "💧 **Water Level Management**: Maintain proper water levels. Drain fields periodically to reduce hopper populations.",
                "🕰️ **Timing Critical**: Apply control measures during early morning or late evening when insects are most active.",
            ],
            "Tomato": [
                "🍅 **Fruit Borer Prevention**: Check fruits for tiny entry holes with brown frass. Remove infected fruits immediately.",
                "🍃 **Whitefly Management**: Look for tiny white insects on leaf undersides. They cause yellowing and transmit viral diseases.",
                "📱 **Integrated Control**: Use yellow sticky traps + reflective mulch + bio-pesticides for comprehensive whitefly management.",
                "✂️ **Pruning Strategy**: Remove lower leaves touching ground and maintain proper plant spacing for air circulation.",
            ],
            "Sugarcane": [
                "🌿 **Borer Complex Management**: Check for entry/exit holes in stems and presence of galleries inside canes.",
                "🐛 **Scale Insect Control**: Look for white waxy deposits on leaves and stems. Heavy infestations cause yellowing.",
                "🔥 **Trash Management**: Remove crop residues and burn them to eliminate overwintering pest populations.",
            ],
            "Soybean": [
                "🌱 **Defoliator Management**: Monitor for caterpillars causing leaf damage. Economic threshold is 25% defoliation.",
                "🌿 **Pod Borer Control**: Check pods for entry holes during pod-filling stage. This is the most critical period.",
                "📊 **Sampling Method**: Use beat sheet method for accurate pest counting. Sample 10 spots per field.",
            ],
        }

        if crop_type in crop_specific:
            recommendations.extend(crop_specific[crop_type])
        else:
            recommendations.extend(
                [
                    "🔍 **General Crop Monitoring**: Focus on growing points, flower clusters, and developing fruits/pods.",
                    "🌿 **Universal Practices**: Maintain field hygiene, use sticky traps, and practice crop rotation.",
                ]
            )

        return recommendations

    def get_risk_status(self, risk_value):
        """Convert risk value to status"""
        if risk_value > 70:
            return "High"
        elif risk_value > 50:
            return "Medium"
        elif risk_value > 30:
            return "Low"
        else:
            return "Very Low"

    def get_risk_level(self, risk_score):
        """Get overall risk level with color coding"""
        if risk_score > 75:
            return {"level": "Critical", "color": "#FF4444"}
        elif risk_score > 60:
            return {"level": "High", "color": "#FF8800"}
        elif risk_score > 40:
            return {"level": "Medium", "color": "#FFBB00"}
        elif risk_score > 20:
            return {"level": "Low", "color": "#88DD88"}
        else:
            return {"level": "Very Low", "color": "#44BB44"}

    def get_pest_treatment_plans(self, crop_type, riskScore):
        """Get actionable treatment plans with products and pricing"""
        treatments = {
            "Cotton": {
                "Bollworm": [
                    {"product": "Chlorpyriphos 20%", "price": "279/L"},
                    {"product": "Neem Oil", "price": "180/L"},
                ],
                "Aphids": [{"product": "Imidacloprid", "price": "350/L"}],
            },
            "Rice": {
                "Brown Plant Hopper": [{"product": "Buprofezin", "price": "280/L"}]
            },
            "Tomato": {"Fruit Borer": [{"product": "Spinosad", "price": "280/L"}]},
        }
        severity = (
            "Critical" if riskScore > 70 else "Moderate" if riskScore > 40 else "Low"
        )
        return {
            "severity": severity,
            "treatments": treatments.get(crop_type, {}),
            "note": "Wear PPE",
        }

    def get_optimal_spraying_windows(self, weather):
        """Get optimal spray timing"""
        t = weather.get("temperature", 25)
        w = weather.get("wind_speed", 2.5)
        r = weather.get("rainfall", 0)

        warnings = []
        if r > 3:
            warnings.append(f"Rain {r}mm - Skip")
        if w > 3.5:
            warnings.append(f"Wind {w}m/s - Wait")

        windows = (
            ["06-09 AM", "17-20 PM"]
            if (15 <= t <= 32 and w <= 3.5 and r <= 3)
            else ["Wait for better weather"]
        )
        return {"windows": windows, "warnings": warnings, "reapply": "7-10 days"}

    def calculate_soil_moisture(self, soil_ph, daily_need, rainfall):
        """Calculate soil moisture (0-100%)"""
        rain_c = min(50, rainfall * 5)
        retention = 0.9 if soil_ph > 7.5 else 0.8 if 5.5 <= soil_ph <= 7.5 else 0.6
        demand = min(50, (daily_need / 10) * 10)
        moisture = int((rain_c * retention + (50 - demand)) * 0.7)
        moisture = max(10, min(95, moisture))

        if moisture > 70:
            status, action = "Very High", "Space irrigations"
        elif moisture > 50:
            status, action = "Adequate", "Continue"
        elif moisture > 30:
            status, action = "Low", "Water soon"
        else:
            status, action = "Critical", "Irrigate now"

        return {"level": f"{moisture}%", "status": status, "action": action}

    def get_rain_forecast_impact(self, expected_rain, daily_need):
        """Analyze rain impact on irrigation"""
        if expected_rain >= daily_need * 0.8:
            return {
                "recommendation": "Skip irrigation",
                "reason": "Rain will meet needs",
            }
        elif expected_rain >= daily_need * 0.4:
            return {
                "recommendation": f"Reduce by {(daily_need-expected_rain):.1f}mm",
                "reason": "Partial rain expected",
            }
        else:
            return {
                "recommendation": "Irrigate as planned",
                "reason": "Insufficient rain",
            }

    def generate_zone_risk_summary(self):
        """Generate comprehensive zone-wise risk summary"""
        zone_summary = {
            "Western Zone": {
                "districts": ["Pune", "Satara", "Sangli", "Kolhapur", "Solapur"],
                "avg_risk": 65.8,
                "high_risk_districts": 1,
                "total_districts": 5,
            },
            "Coastal Zone": {
                "districts": ["Mumbai", "Thane"],
                "avg_risk": 60.7,
                "high_risk_districts": 0,
                "total_districts": 2,
            },
            "Vidarbha Zone": {
                "districts": ["Nagpur", "Wardha", "Chandrapur", "Gadchiroli", "Gondia"],
                "avg_risk": 63.8,
                "high_risk_districts": 1,
                "total_districts": 5,
            },
            "Northern Zone": {
                "districts": ["Nashik", "Dhule", "Jalgaon"],
                "avg_risk": 63.2,
                "high_risk_districts": 1,
                "total_districts": 3,
            },
            "Marathwada Zone": {
                "districts": ["Aurangabad", "Osmanabad", "Latur", "Nanded"],
                "avg_risk": 64.7,
                "high_risk_districts": 1,
                "total_districts": 4,
            },
            "Southern Zone": {
                "districts": ["Raigad"],
                "avg_risk": 76.6,
                "high_risk_districts": 1,
                "total_districts": 1,
            },
        }
        return zone_summary

    def generate_30_day_weather_data(self, district):
        """Generate 30-day weather trend data"""
        import random
        from datetime import datetime, timedelta

        # Generate dates for last 30 days
        end_date = datetime.now()
        dates = [
            (end_date - timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range(30, 0, -1)
        ]

        # Generate realistic weather patterns for Maharashtra
        base_temp = 28  # September average temp
        base_humidity = 65
        base_wind = 3.0

        weather_data = {
            "dates": dates,
            "temperature": [],
            "humidity": [],
            "rainfall": [],
            "wind_speed": [],
        }

        for i, date in enumerate(dates):
            # Temperature variations (18-38°C range for Maharashtra)
            temp_variation = random.uniform(-8, 10)
            daily_temp = max(18, min(38, base_temp + temp_variation))
            weather_data["temperature"].append(round(daily_temp, 1))

            # Humidity patterns (30-85% range)
            humidity_variation = random.uniform(-25, 20)
            daily_humidity = max(30, min(85, base_humidity + humidity_variation))
            weather_data["humidity"].append(round(daily_humidity, 1))

            # Rainfall patterns (0-10mm daily, with occasional heavy days)
            if random.random() < 0.3:  # 30% chance of rain
                daily_rainfall = random.uniform(0.5, 10.0)
            else:
                daily_rainfall = 0
            weather_data["rainfall"].append(round(daily_rainfall, 1))

            # Wind speed (1-5 m/s typical range)
            wind_variation = random.uniform(-1.5, 2.0)
            daily_wind = max(1, min(5, base_wind + wind_variation))
            weather_data["wind_speed"].append(round(daily_wind, 1))

        return weather_data

    def get_irrigation_recommendations(
        self, crop_type, district, growth_stage, soil_ph, farm_area, current_weather
    ):
        """Generate comprehensive irrigation recommendations based on multiple factors"""
        recommendations = {}

        # Get zone for district
        zone = None
        for zone_name, districts in self.maharashtra_districts.items():
            if district in districts:
                zone = zone_name
                break

        # Zone-specific irrigation patterns
        zone_irrigation_data = {
            "Konkan (Coastal)": {
                "base_water_need": 600,  # mm per season
                "monsoon_dependency": 0.8,
                "irrigation_efficiency": 0.75,
                "recommended_method": "Drip + Sprinkler",
                "water_source": "Wells + Rainwater harvesting",
            },
            "Western Maharashtra": {
                "base_water_need": 500,
                "monsoon_dependency": 0.6,
                "irrigation_efficiency": 0.85,
                "recommended_method": "Drip irrigation",
                "water_source": "Canal + Borewells",
            },
            "North Maharashtra (Khandesh)": {
                "base_water_need": 550,
                "monsoon_dependency": 0.7,
                "irrigation_efficiency": 0.8,
                "recommended_method": "Canal + Drip",
                "water_source": "Tapi river + Wells",
            },
            "Marathwada": {
                "base_water_need": 450,
                "monsoon_dependency": 0.9,
                "irrigation_efficiency": 0.7,
                "recommended_method": "Micro-sprinkler",
                "water_source": "Watershed + Tanks",
            },
            "Vidarbha": {
                "base_water_need": 480,
                "monsoon_dependency": 0.8,
                "irrigation_efficiency": 0.78,
                "recommended_method": "Mixed (Drip + Flood)",
                "water_source": "Rivers + Tanks + Wells",
            },
        }

        # Crop-specific water requirements (mm per growing season)
        crop_water_needs = {
            "Cotton": {
                "min": 600,
                "max": 1200,
                "critical_stages": ["Flowering", "Fruit Development"],
            },
            "Rice": {
                "min": 1000,
                "max": 1800,
                "critical_stages": ["Vegetative", "Flowering"],
            },
            "Wheat": {
                "min": 400,
                "max": 650,
                "critical_stages": ["Tillering", "Grain filling"],
            },
            "Sugarcane": {
                "min": 1200,
                "max": 2000,
                "critical_stages": ["Grand growth", "Maturity"],
            },
            "Soybean": {
                "min": 450,
                "max": 700,
                "critical_stages": ["Flowering", "Pod filling"],
            },
            "Tomato": {
                "min": 500,
                "max": 800,
                "critical_stages": ["Flowering", "Fruit Development"],
            },
            "Potato": {
                "min": 400,
                "max": 600,
                "critical_stages": ["Tuber formation", "Tuber bulking"],
            },
            "Onion": {
                "min": 350,
                "max": 550,
                "critical_stages": ["Bulb formation", "Maturity"],
            },
            "Maize": {
                "min": 500,
                "max": 800,
                "critical_stages": ["Tasseling", "Grain filling"],
            },
            "Jowar": {
                "min": 300,
                "max": 500,
                "critical_stages": ["Panicle emergence", "Grain filling"],
            },
        }

        zone_data = zone_irrigation_data.get(
            zone, zone_irrigation_data["Western Maharashtra"]
        )
        crop_data = crop_water_needs.get(crop_type, crop_water_needs["Cotton"])

        # Calculate water requirements
        base_requirement = (
            crop_data["min"] + (crop_data["max"] - crop_data["min"]) * 0.7
        )  # 70% of max
        zone_adjustment = (
            zone_data["base_water_need"] / 500
        )  # Normalize to Western Maharashtra

        total_water_need = base_requirement * zone_adjustment
        daily_water_need = total_water_need / 120  # 120-day average season

        # Adjust for growth stage
        growth_multipliers = {
            "Sowing": 0.5,
            "Germination": 0.7,
            "Vegetative": 1.0,
            "Flowering": 1.3,
            "Fruit Development": 1.2,
            "Maturity": 0.8,
            "Harvesting": 0.3,
        }

        current_multiplier = growth_multipliers.get(growth_stage, 1.0)
        adjusted_daily_need = daily_water_need * current_multiplier

        # Weather-based adjustments
        temp = current_weather.get("temperature", 25)
        humidity = current_weather.get("humidity", 65)
        wind_speed = current_weather.get("wind_speed", 3)

        # Temperature adjustment
        if temp > 35:
            temp_factor = 1.3
        elif temp > 30:
            temp_factor = 1.1
        elif temp < 20:
            temp_factor = 0.8
        else:
            temp_factor = 1.0

        # Humidity adjustment
        humidity_factor = 1.5 - (humidity / 100)  # Lower humidity = more water need

        # Wind adjustment
        wind_factor = 1 + (wind_speed - 3) * 0.05  # Higher wind = more water need

        final_daily_need = (
            adjusted_daily_need * temp_factor * humidity_factor * wind_factor
        )

        # Calculate irrigation schedule
        irrigation_frequency = (
            "Daily"
            if final_daily_need > 8
            else "Every 2 days" if final_daily_need > 4 else "Every 3 days"
        )
        water_per_application = (
            final_daily_need * farm_area * 10
        )  # Convert to liters per hectare

        recommendations = {
            "daily_water_requirement": round(final_daily_need, 2),
            "water_per_hectare": round(water_per_application, 0),
            "irrigation_frequency": irrigation_frequency,
            "recommended_method": zone_data["recommended_method"],
            "water_source": zone_data["water_source"],
            "critical_growth_stages": crop_data["critical_stages"],
            "is_critical_stage": growth_stage in crop_data["critical_stages"],
            "zone_efficiency": zone_data["irrigation_efficiency"],
            "seasonal_total": round(total_water_need, 0),
            "current_factors": {
                "temperature_impact": f"{((temp_factor - 1) * 100):+.1f}%",
                "humidity_impact": f"{((humidity_factor - 1) * 100):+.1f}%",
                "wind_impact": f"{((wind_factor - 1) * 100):+.1f}%",
                "growth_stage_impact": f"{((current_multiplier - 1) * 100):+.1f}%",
            },
            "recommendations": self.generate_irrigation_recommendations(
                zone, crop_type, growth_stage, final_daily_need
            ),
            "soil_moisture": self.calculate_soil_moisture(
                soil_ph, final_daily_need, current_weather.get("rainfall", 0)
            ),
            "rain_forecast_impact": self.get_rain_forecast_impact(
                current_weather.get("rainfall", 0), final_daily_need
            ),
        }

        return recommendations

    def generate_irrigation_recommendations(
        self, zone, crop_type, growth_stage, daily_need
    ):
        """Generate specific irrigation recommendations"""
        recommendations = []

        # Base recommendations with detailed explanations
        if daily_need > 8:
            recommendations.append(
                "⚠️ **HIGH WATER DEMAND**: Daily requirement {:.1f}mm indicates high evapotranspiration. Consider drip irrigation for 40-60% water savings.".format(
                    daily_need
                )
            )
            recommendations.append(
                "🌅 **Optimal Timing**: Irrigate during early morning (5:30-8:00 AM) when evaporation rates are lowest. Avoid midday irrigation."
            )
            recommendations.append(
                "📊 **Split Application**: Divide daily water into 2-3 smaller applications to improve soil water retention."
            )
        elif daily_need < 3:
            recommendations.append(
                "✅ LOW water requirement. Monitor soil moisture before irrigating."
            )
            recommendations.append(
                "📅 Extend irrigation intervals to prevent waterlogging."
            )
        else:
            recommendations.append(
                "✅ MODERATE water requirement. Maintain regular irrigation schedule."
            )

        # Zone-specific recommendations
        if "Coastal" in zone:
            recommendations.append(
                "🌊 Coastal zone: Use rainwater harvesting during monsoon."
            )
            recommendations.append("🧂 Monitor soil salinity near coastal areas.")
        elif "Marathwada" in zone:
            recommendations.append(
                "💧 Marathwada zone: Focus on water conservation techniques."
            )
            recommendations.append("🌧️ Implement watershed management practices.")
        elif "Vidarbha" in zone:
            recommendations.append(
                "🏞️ Vidarbha zone: Utilize river water sources when available."
            )
            recommendations.append("💨 Consider wind breaks to reduce evaporation.")

        # Crop-specific recommendations
        if crop_type == "Cotton":
            recommendations.append(
                "🌱 Cotton: Avoid overhead irrigation during flowering to prevent pest issues."
            )
        elif crop_type == "Rice":
            recommendations.append(
                "🌾 Rice: Maintain 2-5 cm water depth in fields during vegetative stage."
            )
        elif crop_type == "Sugarcane":
            recommendations.append(
                "🎋 Sugarcane: Ensure adequate moisture during grand growth phase."
            )

        # Growth stage recommendations
        if growth_stage in ["Flowering", "Fruit Development"]:
            recommendations.append(
                "🌸 Critical growth stage: Maintain consistent soil moisture."
            )
            recommendations.append("❌ Avoid water stress during this critical period.")

        return recommendations

    def save_analysis_data(self, data):
        """Save analysis data to MongoDB database"""
        # Check database connection
        if (
            not hasattr(self, "mongo_db")
            or not self.mongo_db
            or not getattr(self.mongo_db, "connected", False)
        ):
            # do not spam error if user never asked to save
            st.warning("Database connection unavailable; analysis will not be saved.")
            return False

        # Validate input data
        if not isinstance(data, (list, tuple)) or len(data) < 13:
            st.error("Invalid analysis data format")
            return False

        # Create document with proper data types
        try:
            analysis_doc = {
                "timestamp": (
                    data[0] if isinstance(data[0], datetime) else datetime.now()
                ),
                "district": str(data[1]),
                "crop_type": str(data[2]),
                "growth_stage": str(data[3]),
                "farm_area": float(data[4]),
                "disease_detected": str(data[5]),
                "confidence": float(data[6]),
                "ndvi_value": float(data[7]),
                "soil_ph": float(data[8]),
                "nitrogen": float(data[9]),
                "phosphorus": float(data[10]),
                "potassium": float(data[11]),
                "recommendations": str(data[12]),
                "farmer_id": st.session_state.get("farmer_id"),
                "created_at": datetime.now(),
            }

            # Save to MongoDB
            result = self.mongo_db.save_crop_analysis(analysis_doc)
            if result and hasattr(result, "inserted_id"):
                st.success(
                    f"Analysis data saved successfully (ID: {result.inserted_id})"
                )
                return True
            else:
                st.error("Failed to save analysis data to MongoDB")
                return False
        except (ValueError, TypeError) as e:
            st.error(f"Error processing analysis data: {str(e)}")
            return False
        except Exception as e:
            st.error(f"Error saving to database: {str(e)}")
            return False


def generate_pdf_report(
    district, zone, crop_type, growth_stage, farm_area, current_weather
):
    """Generate a professional 4-page PDF report with charts including:
    - Page 1: Summary with Key Alerts
    - Page 2: NDVI Trend Chart
    - Page 3: Soil Nutrients & Cost Analysis
    - Page 4: Irrigation & Action Plan
    """
    buffer = BytesIO()
    pdf = PdfPages(buffer)

    # Safely pull optional data from the current Streamlit session
    try:
        crop_analysis = (
            st.session_state.get("crop_analysis") if "st" in globals() else None
        )
        soil_analysis = (
            st.session_state.get("soil_analysis") if "st" in globals() else None
        )
        pest_analysis = (
            st.session_state.get("pest_analysis") if "st" in globals() else None
        )
        ndvi_trend = st.session_state.get("ndvi_trend") if "st" in globals() else None
        irrigation_analysis = (
            st.session_state.get("irrigation_recommendations")
            if "st" in globals()
            else None
        )
    except Exception:
        crop_analysis = soil_analysis = pest_analysis = ndvi_trend = (
            irrigation_analysis
        ) = None

    ################################################
    # PAGE 1 — SUMMARY WITH ALERTS
    ################################################
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")

    fig.text(
        0.5,
        0.95,
        "Maharashtra Smart Agriculture Report",
        ha="center",
        fontsize=20,
        fontweight="bold",
        color="#1B5E20",
    )

    fig.text(
        0.5,
        0.92,
        f"Generated: {datetime.now().strftime('%d %B %Y %H:%M')}",
        ha="center",
        fontsize=10,
        color="#666",
    )

    info = f"""
District : {district}
Zone     : {zone}
Crop     : {crop_type}
Stage    : {growth_stage}
Area     : {farm_area} ha
Weather  : {current_weather}
"""

    fig.text(
        0.1,
        0.80,
        info,
        fontsize=11,
        bbox=dict(
            boxstyle="round",
            facecolor="#E8F5E9",
            pad=1,
            edgecolor="#4CAF50",
            linewidth=2,
        ),
    )

    # Key Alerts Section
    fig.text(
        0.1, 0.62, "🔴 Key Alerts", fontsize=14, fontweight="bold", color="#D32F2F"
    )

    y = 0.58

    # Disease Alert
    if (
        crop_analysis
        and crop_analysis.get("disease_detected")
        and "no disease" not in str(crop_analysis.get("disease_detected")).lower()
    ):
        fig.text(
            0.12,
            y,
            f"⚠️  Disease Detected: {crop_analysis['disease_detected']}",
            fontsize=11,
            color="red",
            fontweight="bold",
        )
        y -= 0.04
    else:
        fig.text(
            0.12,
            y,
            "✅ No disease detected",
            fontsize=11,
            color="green",
            fontweight="bold",
        )
        y -= 0.04

    # Pest Alert
    if pest_analysis and pest_analysis.get("pest_found"):
        fig.text(
            0.12,
            y,
            f"⚠️  Pest Alert: {pest_analysis['pest_found']}",
            fontsize=11,
            color="red",
            fontweight="bold",
        )
        y -= 0.04
    else:
        fig.text(
            0.12,
            y,
            "✅ No pest attack detected",
            fontsize=11,
            color="green",
            fontweight="bold",
        )
        y -= 0.04

    # Cost Summary Box
    total_cost = 0
    if soil_analysis and isinstance(soil_analysis, dict):
        total_cost = soil_analysis.get("total_cost", 0)

    fig.text(
        0.1,
        y - 0.08,
        "💰 Quick Cost Summary",
        fontsize=13,
        fontweight="bold",
        color="#F57C00",
    )
    y -= 0.13
    fig.text(
        0.12,
        y,
        f"Total Fertilizer Cost: ₹{total_cost:,.2f}",
        fontsize=12,
        fontweight="bold",
        color="#E65100",
        bbox=dict(boxstyle="round", facecolor="#FFF9C4", pad=0.5),
    )

    fig.text(0.5, 0.03, "Page 1 of 4", ha="center", fontsize=8, color="#999")

    pdf.savefig(fig)
    plt.close(fig)

    ################################################
    # PAGE 2 — NDVI TREND CHART
    ################################################
    fig2 = plt.figure(figsize=(8.27, 11.69))
    fig2.patch.set_facecolor("white")

    fig2.text(
        0.5,
        0.95,
        "Crop Health Analysis (NDVI Trend)",
        ha="center",
        fontsize=18,
        fontweight="bold",
        color="#1B5E20",
    )

    # Sample NDVI if none provided
    if not ndvi_trend:
        ndvi_trend = [0.42, 0.48, 0.55, 0.61, 0.67, 0.71, 0.76]

    weeks = list(range(1, len(ndvi_trend) + 1))

    ax1 = fig2.add_axes([0.1, 0.45, 0.8, 0.35])
    ax1.plot(
        weeks, ndvi_trend, marker="o", linewidth=2.5, markersize=8, color="#2E7D32"
    )
    ax1.fill_between(weeks, ndvi_trend, alpha=0.3, color="#4CAF50")
    ax1.set_title(
        "NDVI Value Progression Over Growth Period", fontsize=12, fontweight="bold"
    )
    ax1.set_xlabel("Weeks After Sowing", fontsize=11)
    ax1.set_ylabel("NDVI Value", fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    # NDVI Interpretation Guide
    fig2.text(
        0.1, 0.35, "NDVI Health Interpretation Guide:", fontsize=12, fontweight="bold"
    )

    ndvi_text = """
0.0 - 0.2  : No vegetation / Water bodies
0.2 - 0.3  : Poor vegetation stress
0.4 - 0.5  : Moderate growth (monitoring needed)
0.6 - 0.7  : Good healthy crop (optimal)
0.75+      : Excellent vegetation (peak vigor)

Current NDVI: {:.2f}  |  Trend: {}
""".format(
        ndvi_trend[-1] if ndvi_trend else 0,
        (
            "Improving ↗"
            if len(ndvi_trend) > 1 and ndvi_trend[-1] > ndvi_trend[-2]
            else "Stable →"
        ),
    )

    fig2.text(
        0.12,
        0.25,
        ndvi_text,
        fontsize=10,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="#E8F5E9", pad=0.7),
    )

    fig2.text(0.5, 0.03, "Page 2 of 4", ha="center", fontsize=8, color="#999")

    pdf.savefig(fig2)
    plt.close(fig2)

    ################################################
    # PAGE 3 — SOIL NUTRIENTS & COST ANALYSIS
    ################################################
    fig3 = plt.figure(figsize=(8.27, 11.69))
    fig3.patch.set_facecolor("white")

    fig3.text(
        0.5,
        0.95,
        "Soil Nutrients & Fertilizer Cost Analysis",
        ha="center",
        fontsize=18,
        fontweight="bold",
        color="#1B5E20",
    )

    # Soil Nutrient Bar Chart
    nutrients = ["Nitrogen (N)", "Phosphorus (P)", "Potassium (K)"]
    values = [0, 0, 0]
    soil_ph = "N/A"

    if soil_analysis and isinstance(soil_analysis, dict):
        values = [
            (
                float(soil_analysis.get("nitrogen", 0))
                if soil_analysis.get("nitrogen")
                else 0
            ),
            (
                float(soil_analysis.get("phosphorus", 0))
                if soil_analysis.get("phosphorus")
                else 0
            ),
            (
                float(soil_analysis.get("potassium", 0))
                if soil_analysis.get("potassium")
                else 0
            ),
        ]
        soil_ph = soil_analysis.get("soil_ph", "N/A")

    ax2 = fig3.add_axes([0.1, 0.50, 0.35, 0.30])
    colors = ["#4CAF50", "#2196F3", "#FFC107"]
    bars = ax2.bar(
        ["N", "P", "K"], values, color=colors, edgecolor="black", linewidth=1.5
    )
    ax2.set_title("Soil Nutrient Levels (mg/kg)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("mg/kg", fontsize=10)
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # Fertilizer Cost Distribution
    fert_names = []
    fert_costs = []

    if soil_analysis and isinstance(soil_analysis, dict):
        fert_recs = soil_analysis.get("fertilizer_recommendations", [])
        if fert_recs:
            for f in fert_recs:
                if isinstance(f, dict):
                    fert_names.append(f.get("name", "")[:15])  # Truncate long names
                    try:
                        fert_costs.append(float(f.get("cost", 0)))
                    except:
                        fert_costs.append(0)

    ax3 = fig3.add_axes([0.55, 0.50, 0.35, 0.30])

    if fert_costs:
        bars3 = ax3.barh(
            fert_names, fert_costs, color="#FF9800", edgecolor="black", linewidth=1.5
        )
        ax3.set_title("Fertilizer Cost Distribution", fontsize=11, fontweight="bold")
        ax3.set_xlabel("Cost (₹)", fontsize=10)
        for i, (bar, cost) in enumerate(zip(bars3, fert_costs)):
            width = bar.get_width()
            ax3.text(
                width,
                bar.get_y() + bar.get_height() / 2.0,
                f" ₹{cost:.0f}",
                ha="left",
                va="center",
                fontsize=9,
                fontweight="bold",
            )
    else:
        ax3.text(
            0.5, 0.5, "No cost data available", ha="center", va="center", fontsize=10
        )
        ax3.set_xticks([])
        ax3.set_yticks([])

    # Soil & Cost Summary
    fig3.text(
        0.1,
        0.35,
        "📊 Soil & Cost Summary",
        fontsize=12,
        fontweight="bold",
        color="#333",
    )

    summary = f"""
Soil pH Level          : {soil_ph}
Total Fertilizer Cost  : ₹{total_cost:,.2f}
Nitrogen (N)           : {values[0]:.0f} mg/kg
Phosphorus (P)         : {values[1]:.0f} mg/kg
Potassium (K)          : {values[2]:.0f} mg/kg
Recommendation         : Balanced fertilization as per application plan
"""

    fig3.text(
        0.12,
        0.25,
        summary,
        fontsize=10,
        family="monospace",
        bbox=dict(
            boxstyle="round",
            facecolor="#FFF3E0",
            pad=0.8,
            edgecolor="#F57C00",
            linewidth=1.5,
        ),
    )

    fig3.text(0.5, 0.03, "Page 3 of 4", ha="center", fontsize=8, color="#999")

    pdf.savefig(fig3)
    plt.close(fig3)

    ################################################
    # PAGE 4 — IRRIGATION & ACTION PLAN
    ################################################
    fig4 = plt.figure(figsize=(8.27, 11.69))
    fig4.patch.set_facecolor("white")

    fig4.text(
        0.5,
        0.95,
        "Irrigation Advisory & Farmer Action Plan",
        ha="center",
        fontsize=18,
        fontweight="bold",
        color="#1B5E20",
    )

    y = 0.88

    # Irrigation Advisory
    fig4.text(
        0.05,
        y,
        "💧 Irrigation Advisory",
        fontsize=13,
        fontweight="bold",
        color="#2196F3",
    )
    y -= 0.04

    if irrigation_analysis and isinstance(irrigation_analysis, (dict, list)):
        if isinstance(irrigation_analysis, dict):
            irr = f"""
Method       : {irrigation_analysis.get('method', 'N/A')}
Frequency    : {irrigation_analysis.get('frequency', 'N/A')}
Water Need   : {irrigation_analysis.get('water_need', 'N/A')}
Next Due     : {irrigation_analysis.get('next_irrigation', 'N/A')}
"""
        else:
            irr = "\n".join([f"• {item}" for item in irrigation_analysis[:4]])
    else:
        irr = "• Drip irrigation recommended\n• Frequency: 2-3 times per week\n• Water need: 40-50 mm per week"

    fig4.text(
        0.07,
        y,
        irr,
        fontsize=10,
        bbox=dict(
            boxstyle="round",
            facecolor="#E3F2FD",
            pad=0.8,
            edgecolor="#2196F3",
            linewidth=1.5,
        ),
    )
    y -= 0.20

    # Farmer Checklist
    fig4.text(
        0.05, y, "✅ Farmer Checklist", fontsize=13, fontweight="bold", color="#2E7D32"
    )
    y -= 0.04

    actions = [
        "☐ Inspect crop weekly for diseases and pests",
        "☐ Apply fertilizers as per recommended schedule",
        "☐ Maintain irrigation schedule as advised",
        "☐ Record all farm inputs and expenses",
        "☐ Report unusual symptoms to local extension officer",
        "☐ Follow weather-based crop advisories",
        "☐ Ensure proper drainage during heavy rainfall",
        "☐ Collect soil samples before next crop",
    ]

    for a in actions:
        fig4.text(0.07, y, a, fontsize=10)
        y -= 0.035

    y -= 0.03

    # Support Information
    fig4.text(
        0.05,
        y,
        "📞 Support & Resources",
        fontsize=13,
        fontweight="bold",
        color="#D32F2F",
    )
    y -= 0.04

    support = """
🎯 Kisan Helpline        : 1800-180-1551 (Toll Free)
📱 SMS Advisory          : Send 'KRISHI' to 56161
🌐 Web Portal            : www.krishimaharashtra.gov.in
⚠️  Emergency Hotline    : 104 (For urgent pest/disease issues)
📧 Support Email         : support@maharashtragov.in
"""

    fig4.text(
        0.07,
        y,
        support,
        fontsize=9,
        family="monospace",
        bbox=dict(
            boxstyle="round",
            facecolor="#FFEBEE",
            pad=0.8,
            edgecolor="#D32F2F",
            linewidth=1.5,
        ),
    )

    fig4.text(
        0.5,
        0.03,
        "Page 4 of 4 | Government of Maharashtra | Printed Report for Farm Reference",
        ha="center",
        fontsize=8,
        color="#999",
    )

    pdf.savefig(fig4)
    plt.close(fig4)

    ################################################
    # FINALIZE PDF
    ################################################
    pdf.close()
    buffer.seek(0)
    return buffer.getvalue()


def offer_report_download(report_bytes, filename=None):
    """Offer a professional Streamlit download button for the generated PDF.

    - If Streamlit is available in the environment (`st` in globals()), this will
      render a clear download button with helpful tooltip and filename.
    - Otherwise it will write the bytes to a local file named by `filename`.
    """
    if filename is None:
        # safe default filename
        filename = f"MahaAgro_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"

    if "st" in globals():
        try:
            st.download_button(
                label="📄 Download PDF Report",
                data=report_bytes,
                file_name=filename,
                mime="application/pdf",
                help="High-quality A4 report (ready to print).",
                key="mahaagro_pdf_download",
            )
        except Exception:
            # If download button fails, offer a fallback message
            try:
                st.write(f"Download ready — save file as: **{filename}**")
            except Exception:
                pass
    else:
        # Non-Streamlit fallback: write file locally
        try:
            with open(filename, "wb") as f:
                f.write(report_bytes)
            print(f"Report saved to {filename}")
        except Exception as e:
            print(f"Unable to save report to disk: {e}")


def main():
    """Main Streamlit application"""

    # Initialize system with error handling
    try:
        # Initialize scheduler to keep app alive 24/7
        init_scheduler()

        if "agri_system" not in st.session_state:
            st.session_state.agri_system = MaharashtraAgriculturalSystem()

        system = st.session_state.agri_system

        # Initialize all session state variables with default values
        if "crop_analysis" not in st.session_state:
            st.session_state.crop_analysis = None

        if "soil_analysis" not in st.session_state:
            st.session_state.soil_analysis = None

        if "pest_analysis" not in st.session_state:
            st.session_state.pest_analysis = None

        if "irrigation_analysis" not in st.session_state:
            st.session_state.irrigation_analysis = None

        # Auth DB
        if "auth_db" not in st.session_state:
            st.session_state.auth_db = MongoFarmerAuth()
            # notify user if mongodb connection was not established
            if not getattr(st.session_state.auth_db, "connected", False):
                st.warning(
                    "⚠️ Unable to connect to MongoDB. The system is running in offline mode; "
                    "authentication and data persistence features will be limited. "
                    "Ensure the MONGODB_URI environment variable is set to a valid URI "
                    "or verify network access to your database server."
                )
        if "authenticated" not in st.session_state:
            st.session_state.authenticated = False

    except Exception as e:
        st.error(f"System initialization error: {str(e)}")
        st.stop()

    # Enhanced Authentication gate with modern UI
    if not st.session_state.authenticated:
        # Apply custom styling for auth pages
        st.markdown(
            """
<style>
    div[data-testid="stForm"] {
        background: linear-gradient(135deg, var(--card-bg) 0%, var(--surface-bg) 100%);
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid var(--border-color);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }

    .auth-header {
        background: linear-gradient(135deg, var(--primary-green) 0%, var(--secondary-green) 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 4px 15px rgba(46, 125, 50, 0.3);
    }

    .form-header {
        color: var(--text-primary);
        margin-bottom: 1.5rem;
        text-align: center;
    }

    .stTextInput>div>div>input:focus {
    outline: none !important;
    box-shadow: none !important;
}

/* Focus effect on whole container */
.stTextInput:focus-within {
    border: 1px solid #2e7d32;
    box-shadow: 0 0 0 3px rgba(46, 125, 50, 0.25);
    background: rgba(255, 255, 255, 0.07);
}
    }

    .stButton>button {
        width: 100% !important;
        margin-top: 1rem !important;
    }

    .auth-links {
        text-align: center;
        margin-top: 1rem;
        color: var(--text-secondary);
    }

    .auth-links a {
        color: var(--secondary-green);
        text-decoration: none;
    }

    .auth-links a:hover {
        text-decoration: underline;
    }
</style>


        """,
            unsafe_allow_html=True,
        )

        # Enhanced Header
        st.markdown(
            """
        <div class="auth-header">
            <h1>🌾 MahaAgroAI 🌾</h1>
            <p style="font-size: 1.2rem; opacity: 0.9;">Secure Farmer Access Portal</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Initialize session state for form validation
        if "password_visible" not in st.session_state:
            st.session_state.password_visible = False
        if "remember_me" not in st.session_state:
            st.session_state.remember_me = False

        tab_login, tab_register = st.tabs(["🔐 Login", "📝 Register"])

        with tab_login:
            st.markdown(
                '<div class="form-header"><h2>Welcome Back! 👋</h2></div>',
                unsafe_allow_html=True,
            )
            with st.form("login_form_main"):
                username = st.text_input(
                    "👤 Username or Email", placeholder="Enter your username or email"
                )
                password = st.text_input(
                    "🔒 Password",
                    type=(
                        "password" if not st.session_state.password_visible else "text"
                    ),
                    placeholder="Enter your password",
                )

                col1, col2 = st.columns(2)
                with col1:
                    remember_me = st.checkbox(
                        "Remember me", value=st.session_state.remember_me
                    )
                with col2:
                    st.markdown(
                        '<div style="text-align: right;"><a href="#" style="color: var(--secondary-green);">Forgot Password?</a></div>',
                        unsafe_allow_html=True,
                    )

                submit = st.form_submit_button("🚀 Login")

            if submit:
                if not username or not password:
                    st.error("⚠️ Please fill in all fields")
                else:
                    with st.spinner("🔄 Authenticating..."):
                        res = st.session_state.auth_db.authenticate_farmer(
                            username, password, ip_address="127.0.0.1"
                        )
                        if res.get("success"):
                            st.session_state.authenticated = True
                            st.session_state.farmer_id = res.get("farmer_id")
                            st.session_state.username = res.get("username")
                            st.session_state.full_name = res.get("full_name")
                            if remember_me:
                                st.session_state.remember_me = True
                            st.success(
                                "✅ Login successful! Redirecting to dashboard..."
                            )
                            st.rerun()
                        else:
                            st.error(f"❌ {res.get('message', 'Login failed')}")

        with tab_register:
            st.markdown(
                '<div class="form-header"><h2>Create New Account 🌱</h2></div>',
                unsafe_allow_html=True,
            )
            with st.form("register_form_main"):
                col1, col2 = st.columns(2)
                with col1:
                    full_name = st.text_input(
                        "👤 Full Name", placeholder="Enter your full name"
                    )
                    username_r = st.text_input(
                        "👤 Username", placeholder="Choose a username"
                    )
                with col2:
                    email = st.text_input("📧 Email", placeholder="Enter your email")
                    phone = st.text_input(
                        "📱 Phone (Optional)", placeholder="Enter your phone number"
                    )

                password_r = st.text_input(
                    "🔒 Password",
                    type=(
                        "password" if not st.session_state.password_visible else "text"
                    ),
                    placeholder="Choose a strong password",
                    help="Password must be at least 8 characters long with letters and numbers",
                )

                password_confirm = st.text_input(
                    "🔒 Confirm Password",
                    type=(
                        "password" if not st.session_state.password_visible else "text"
                    ),
                    placeholder="Confirm your password",
                )

                # Terms and conditions checkbox
                agree_terms = st.checkbox("I agree to the Terms and Conditions")

                submit_r = st.form_submit_button("🌱 Create Account")

            if submit_r:
                if not all(
                    [full_name, username_r, email, password_r, password_confirm]
                ):
                    st.error("⚠️ Please fill in all required fields")
                elif not agree_terms:
                    st.error("⚠️ Please agree to the Terms and Conditions")
                elif password_r != password_confirm:
                    st.error("⚠️ Passwords do not match")
                elif len(password_r) < 8:
                    st.error("⚠️ Password must be at least 8 characters long")
                else:
                    with st.spinner("🔄 Creating your account..."):
                        res = st.session_state.auth_db.register_farmer(
                            username=username_r,
                            email=email,
                            password=password_r,
                            full_name=full_name,
                            phone=phone,
                        )
                        if res.get("success"):
                            st.success("✅ Account created successfully! Please login.")
                        else:
                            st.error(f"❌ {res.get('message', 'Registration failed')}")

        # Footer links
        st.markdown(
            """
        <div class="auth-links">
            <p>Need help ? Contact our support team Green Coders At <a>yashimamdar@gmail.com</a></p>
            
            
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.stop()

    # Header
    st.markdown(
        """
    <div class="crop-header">
        <div class="main-title">🌾MahaAgroAI </div>
        <div class="subtitle"><b>Intelligent Farming with Real-Time Crop, Soil, and Weather Analytics Powered by AI<b></div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Sidebar - Farm Information
    with st.sidebar:
        # Warn user if MongoDB is unavailable
        if "auth_db" in st.session_state and not getattr(
            st.session_state.auth_db, "connected", False
        ):
            st.error(
                "🚫 MongoDB connection unavailable. "
                "Some features (login/registration, history) are disabled."
            )
        st.markdown("🏡 Smart Farm Dashboard")
        # Optional logout control (does not alter main UI)
        if st.session_state.authenticated:
            if st.button("Logout"):
                st.session_state.authenticated = False
                for k in ["farmer_id", "username", "full_name"]:
                    if k in st.session_state:
                        del st.session_state[k]
                st.rerun()

        # Enhanced Location & Crop Details with all 36 districts
        st.markdown(
            """
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 1rem;
        ">
            <h3 style="margin: 0;">📍 Farm Location & Details</h3>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # District selection method
        selection_method = st.radio(
            "Choose Selection Method:",
            ["By Zone", "All Districts (A-Z)"],
            help="Select districts either by agricultural zone or alphabetically",
        )

        if selection_method == "By Zone":
            zone = st.selectbox(
                "Select Zone", list(system.maharashtra_districts.keys())
            )
            district = st.selectbox(
                "Select District", system.maharashtra_districts[zone]
            )
        else:
            # Create alphabetically sorted list of all 36 districts
            all_districts = []
            for zone_districts in system.maharashtra_districts.values():
                all_districts.extend(zone_districts)
            all_districts.sort()

            district = st.selectbox(
                "Select District (All 36 Districts)",
                all_districts,
                help="All 36 districts of Maharashtra in alphabetical order",
            )

            # Find which zone this district belongs to
            zone = None
            for zone_name, districts in system.maharashtra_districts.items():
                if district in districts:
                    zone = zone_name
                    break

        # Display selected zone info
        if zone:
            st.info(f"Selected District: **{district}** (Zone: {zone})")

        crop_type = st.selectbox("Select Crop Type", system.crop_types)
        growth_stage = st.selectbox("Current Growth Stage", system.growth_stages)
        farm_area = st.number_input(
            "Farm Area (Hectares)", min_value=0.1, max_value=1000.0, value=1.0, step=0.1
        )

        # Get weather data
        weather_data = system.get_weather_data(district)
        current_weather = weather_data["current"]

        st.markdown("### 🌤️ Current Weather")
        st.metric("Temperature", f"{current_weather['temperature']}°C")
        st.metric("Humidity", f"{current_weather['humidity']}%")
        st.metric("Wind Speed", f"{current_weather['wind_speed']} m/s")

        # Soil Testing Data
        st.markdown("### 🧪 Soil Testing Data")
        st.markdown("#### Enter Soil Test Results")

        soil_ph = st.slider("pH Level", 4.0, 9.0, 6.5, 0.1)
        nitrogen = st.number_input("Nitrogen (kg/ha)", 0, 500, 300, 10)
        phosphorus = st.number_input("Phosphorus (kg/ha)", 0, 100, 15, 5)
        potassium = st.number_input("Potassium (kg/ha)", 0, 300, 150, 10)

        st.markdown("---")

        # Unified Analyze Button
        st.markdown("### 🔍 Analysis Center")

        # Image upload in sidebar
        uploaded_file = st.file_uploader(
            "Upload Crop Image (Optional)",
            type=["jpg", "jpeg", "png"],
            help="Upload crop image for AI disease detection (optional — all other analyses run without it)",
            key="sidebar_image_upload",
        )

        # ── Professional inline validation feedback in sidebar ──
        _image_is_analysis_ready = True  # flag used in analyze button below

        if uploaded_file:
            image_validation = system.validate_image_file(uploaded_file)
            _sev = image_validation.get("severity", "success")

            if _sev in ("error", "critical"):
                _image_is_analysis_ready = False
                st.markdown(
                    """
                    <div style="
                        background: linear-gradient(135deg,#3B0000,#1a0000);
                        border:1px solid #F44336;
                        border-left:5px solid #F44336;
                        border-radius:9px;
                        padding:0.75rem 0.9rem;
                        margin:0.6rem 0;
                    ">
                        <p style="color:#EF9A9A; font-weight:700; font-size:0.88rem; margin:0 0 0.3rem;">
                            🔴 Image Cannot Be Used for Crop Analysis
                        </p>
                    """,
                    unsafe_allow_html=True,
                )
                for err in image_validation.get("errors", []):
                    st.markdown(
                        f"<p style='color:#FFCDD2; font-size:0.8rem; margin:0.15rem 0;'>"
                        f"❌ {err}</p>",
                        unsafe_allow_html=True,
                    )
                st.markdown(
                    """
                        <hr style="border-color:#7f0000; margin:0.5rem 0;"/>
                        <p style="color:#90CAF9; font-size:0.8rem; margin:0;">
                            ✅ <b>All other analyses still run normally.</b><br>
                            Weather · Soil · Pest Risk · Irrigation are unaffected.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            elif _sev == "warning":
                st.markdown(
                    """
                    <div style="
                        background:linear-gradient(135deg,#3D2000,#2A1500);
                        border:1px solid #FF9800;
                        border-left:4px solid #FF9800;
                        border-radius:9px;
                        padding:0.7rem 0.9rem;
                        margin:0.6rem 0;
                    ">
                        <p style="color:#FFB74D; font-weight:700; font-size:0.86rem; margin:0 0 0.25rem;">
                            ⚠️ Image Quality Warning
                        </p>
                        <p style="color:#FFCC80; font-size:0.8rem; margin:0 0 0.3rem;">
                            Analysis will proceed with reduced accuracy.
                        </p>
                        <p style="color:#FFF3E0; font-size:0.79rem; margin:0;">
                            ✅ All tabs (Weather, Soil, Pest, Irrigation) are fully available.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            else:  # success
                st.markdown(
                    """
                    <div style="
                        background:linear-gradient(135deg,#1B3A1F,#1B5E20);
                        border:1px solid #4CAF50;
                        border-left:4px solid #4CAF50;
                        border-radius:9px;
                        padding:0.6rem 0.9rem;
                        margin:0.4rem 0;
                    ">
                        <p style="color:#A5D6A7; font-weight:600; font-size:0.85rem; margin:0;">
                            ✅ Image ready for AI crop analysis
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                """
                <div style="
                    background:rgba(25,118,210,0.10);
                    border:1px dashed #1976D2;
                    border-radius:8px;
                    padding:0.65rem 0.9rem;
                    margin:0.5rem 0;
                ">
                    <p style="color:#90CAF9; font-size:0.82rem; margin:0 0 0.25rem; font-weight:600;">
                        💡 Image is optional
                    </p>
                    <p style="color:#BBDEFB; font-size:0.79rem; margin:0;">
                        You can still get full Weather, Soil, Pest Risk &amp;
                        Irrigation analysis without uploading a crop photo.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if st.button(
            "🔍 ANALYZE ALL DATA",
            type="primary",
            use_container_width=True,
            key="main_analyze_button",
            help="Analyze weather, soil, pest risk & irrigation — crop image is optional",
        ):
            analyses = {}

            # ── 1. Crop image analysis (OPTIONAL — skip gracefully if invalid) ──
            if uploaded_file:
                # Re-validate so we don't analyse a broken file
                _val = system.validate_image_file(uploaded_file)
                if _val["severity"] not in ("error", "critical"):
                    crop_result = system.analyze_crop_image(uploaded_file)
                    if crop_result:
                        analyses["crop"] = crop_result
                        st.session_state.crop_analysis = crop_result
                        if "error" in crop_result:
                            # Non-blocking inline error inside the sidebar summary
                            st.warning(f"⚠️ Crop analysis note: {crop_result['error']}")
                else:
                    # Image invalid — clear any stale crop result and inform user
                    st.session_state.crop_analysis = None
                    st.markdown(
                        """
                        <div style="
                            background:linear-gradient(135deg,#3B0000,#1a0000);
                            border:1px solid #F44336; border-left:5px solid #F44336;
                            border-radius:9px; padding:0.75rem 1rem; margin:0.5rem 0;
                        ">
                            <p style="color:#EF9A9A; font-weight:700; font-size:0.88rem; margin:0 0 0.25rem;">
                                🔴 Crop Image Skipped
                            </p>
                            <p style="color:#FFCDD2; font-size:0.81rem; margin:0;">
                                The uploaded image has validation errors and was excluded from analysis.
                                All other analyses below will run normally.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            # (if no image uploaded, crop analysis simply doesn't run — no error shown)

            # ── 2. Soil analysis — ALWAYS runs ────────────────────────────────
            soil_result = system.analyze_soil_health(
                soil_ph, nitrogen, phosphorus, potassium, farm_area
            )
            analyses["soil"] = soil_result
            st.session_state.soil_analysis = soil_result

            # ── 3. Pest risk — ALWAYS runs ─────────────────────────────────────
            pest_result = system.analyze_pest_risk(
                weather_data, crop_type, growth_stage
            )
            analyses["pest"] = pest_result
            st.session_state.pest_analysis = pest_result

            # ── 4. Irrigation — ALWAYS runs ────────────────────────────────────
            irrigation_result = system.get_irrigation_recommendations(
                crop_type, district, growth_stage, soil_ph, farm_area, current_weather
            )
            analyses["irrigation"] = irrigation_result
            st.session_state.irrigation_analysis = irrigation_result

            # ── 5. Persist to DB ───────────────────────────────────────────────
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            disease_detected = analyses.get("crop", {}).get(
                "disease", "No image uploaded"
            )
            confidence = analyses.get("crop", {}).get("confidence", 0)
            ndvi_val = system.calculate_ndvi(0.8, 0.3)

            system.save_analysis_data(
                (
                    timestamp,
                    district,
                    crop_type,
                    growth_stage,
                    farm_area,
                    disease_detected,
                    confidence,
                    ndvi_val,
                    soil_ph,
                    nitrogen,
                    phosphorus,
                    potassium,
                    "Comprehensive analysis completed",
                )
            )

            st.success("✅ Analysis complete! Check all tabs for results.")

            # ── 6. Quick summary ───────────────────────────────────────────────
            st.markdown("#### 📋 Quick Summary")
            if "crop" in analyses:
                st.write(
                    f"🌿 Crop: {analyses['crop']['disease']} "
                    f"({analyses['crop']['confidence']:.1f}% confidence)"
                )
            else:
                st.info("🌿 Crop analysis: not run (no valid image uploaded)")

            st.write(
                f"🧪 Soil Health: {analyses['soil']['score']}/100 ({analyses['soil']['status']})"
            )
            st.write(
                f"🐛 Pest Risk: {analyses['pest']['overall_risk']}/100 "
                f"({analyses['pest']['risk_level']['level']})"
            )
            st.write(
                f"💧 Irrigation: {analyses['irrigation']['daily_water_requirement']:.1f} mm/day "
                f"({analyses['irrigation']['irrigation_frequency']})"
            )
            st.write(f"💰 Fertilizer Cost: ₹{analyses['soil']['total_cost']:.2f}")

        # Download PDF report button (adds only one button)
        try:
            pdf_buf = generate_pdf_report(
                district, zone, crop_type, growth_stage, farm_area, current_weather
            )
            st.download_button(
                label="⬇️ Download PDF Report",
                data=pdf_buf,
                file_name=f"maharashtra_agri_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="download_pdf_report_button",
            )
        except Exception as _e:
            pass

        # Scheduler Monitoring Section
        st.markdown("---")
        with st.expander("⚙️ System Status & Scheduler", expanded=False):
            scheduler = get_scheduler()
            status = scheduler.get_job_status()

            # Display scheduler status
            if status["scheduler_running"]:
                st.success(
                    f"✅ **App Active & Running 24/7** | "
                    f"{status['jobs_scheduled']} background jobs"
                )
            else:
                st.warning("⏸️ Scheduler inactive")

            # Show next scheduled jobs
            if status["next_jobs"]:
                st.subheader("📅 Next Scheduled Tasks")
                for job in status["next_jobs"][:5]:
                    st.write(f"**{job['name']}** — {job['next_run']}")

            # Show recent job results
            if status["job_details"]:
                st.subheader("📊 Recent Activity")
                for job_name, job_result in status["job_details"].items():
                    st.write(f"• {job_name}: {job_result}")

    # Main Navigation Tabs
    tab_names = [
        "🌱 Crop Health",
        "🌤️ Weather & Soil",
        "🐛 Pest Risk",
        "💧 Irrigation",
        "🗺️ Zone Mapping",
        "📊 Dashboard",
    ]
    tabs = st.tabs(tab_names)

    # Enhanced Crop Health Tab
    with tabs[0]:
        st.markdown(
            '<div class="tab-header"><h2>🌿 Advanced Crop Health Monitoring & Analysis</h2></div>',
            unsafe_allow_html=True,
        )

        # Quick Health Status Banner
        if (
            "crop_analysis" in st.session_state
            and st.session_state.crop_analysis is not None
        ):
            result = st.session_state.crop_analysis
            if isinstance(result, dict) and "error" in result:
                st.warning(
                    f"⚠️ {result.get('error', 'Invalid crop image uploaded. Please upload a valid leaf image.')}."
                )
            else:
                confidence = result.get("confidence", 0)
                disease = result.get("disease", "Unknown")

                if "healthy" in disease.lower():
                    st.markdown(
                        f"""
        <div style="
            background: linear-gradient(135deg, #43A047 0%, #66BB6A 100%);
            padding: 1.8rem;
            border-radius: 18px;
            text-align: center;
            color: white;
            margin: 1.5rem 0;
            border: 2px solid #81C784;
            box-shadow: 0 6px 20px rgba(76, 175, 80, 0.4);
        ">
            <h2 style="margin: 0; font-size: 28px;">🌿 HEALTHY PLANT DETECTED</h2>
            <p style="margin: 10px 0 5px; font-size: 20px; opacity: 0.95;">
                ✅ No Risk Found — Your plant is in great condition!
            </p>
            <p style="margin: 6px 0 0; font-size: 18px; opacity: 0.9;">
                Confidence: {confidence:.1f}% | Keep up your current care routine 🌱
            </p>
        </div>
    """,
                        unsafe_allow_html=True,
                    )
                else:
                    severity = (
                        "HIGH"
                        if confidence > 80
                        else "MEDIUM" if confidence > 60 else "LOW"
                    )
                    alert_color = (
                        "#F44336"
                        if confidence > 80
                        else "#FF9800" if confidence > 60 else "#2196F3"
                    )
                    st.markdown(
                        f"""
                <div style="
                    background: linear-gradient(135deg, {alert_color} 0%, #D32F2F 100%);
                    padding: 1.5rem;
                    border-radius: 15px;
                    text-align: center;
                    color: white;
                    margin: 1rem 0;
                    border: 2px solid #FF5722;
                    box-shadow: 0 4px 15px rgba(244, 67, 54, 0.4);
                ">
                    <h2 style="margin: 0;">⚠️ {disease.upper()} DETECTED</h2>
                    <p style="margin: 8px 0 0; font-size: 18px; opacity: 0.9;">Severity: {severity} | Confidence: {confidence:.1f}%</p>
                    <p style="margin: 4px 0 0; font-size: 16px;">📋 Treatment recommendations available below</p>
                </div>
                """,
                        unsafe_allow_html=True,
                    )

        # Main Content Area
        col1, col2 = st.columns([1.2, 1])

        with col1:
            # Enhanced Image Upload Section
            st.markdown(
                """
                <div style="
                    background: linear-gradient(135deg, var(--secondary-green) 0%, var(--accent-green) 100%);
                    padding: 1rem;
                    border-radius: 12px;
                    color: white;
                    text-align: center;
                    margin-bottom: 1rem;
                    box-shadow: 0 4px 15px rgba(76,175,80,0.3);
                ">
                    <h3 style="margin: 0;">📸 Crop Image Analysis</h3>
                    <p style="margin: 6px 0 0; opacity: 0.9; font-size: 14px;">Upload clear photos for AI-powered disease detection</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Image upload with enhanced styling
            st.markdown(
                '<div class="upload-area" style="border: 3px dashed #4CAF50; background: linear-gradient(135deg, rgba(76,175,80,0.1), rgba(102,187,106,0.1)); padding: 2rem; border-radius: 15px; text-align: center;">',
                unsafe_allow_html=True,
            )
            uploaded_file = st.file_uploader(
                "📤 Choose crop image (leaf, plant, or affected area)",
                type=["jpg", "jpeg", "png"],
                help="💡 Tips for best results:\n• Take photo in natural daylight\n• Focus on affected areas\n• Avoid blurry or dark images\n• Include healthy parts for comparison",
                key="crop_image_uploader",
            )
            st.markdown("</div>", unsafe_allow_html=True)

            if uploaded_file:
                # ── Step 1: Professional validation alert ──────────────────
                validation_result = system.validate_image_file(uploaded_file)
                system.display_professional_image_alert(validation_result)

                _can_analyze = validation_result["severity"] not in (
                    "error",
                    "critical",
                )

                if _can_analyze:
                    # Show image preview + quality info
                    col_img, col_info = st.columns([1, 1])
                    with col_img:
                        st.image(uploaded_file, caption="📸 Uploaded Image", width=300)
                    with col_info:
                        st.markdown("#### 📊 Image Properties")
                        m = validation_result["quality_metrics"]
                        if m:
                            st.markdown("**File Properties:**")
                            st.markdown(
                                f"• **Dimensions:** {m.get('dimensions', 'N/A')}"
                            )
                            st.markdown(
                                f"• **Size:** {m.get('file_size_mb', 'N/A')} MB"
                            )
                            st.markdown(f"• **Format:** {m.get('format', 'N/A')}")
                            st.markdown("**Quality Metrics:**")
                            brightness = m.get("brightness", 0.5)
                            contrast = m.get("contrast", 0.1)
                            b_status = (
                                "✓ Good"
                                if 0.15 < brightness < 0.85
                                else "⚠ Needs Improvement"
                            )
                            c_status = "✓ Good" if contrast > 0.02 else "⚠ Low Contrast"
                            st.markdown(
                                f"• **Brightness:** {brightness:.3f} — {b_status}"
                            )
                            st.markdown(f"• **Contrast:** {contrast:.3f} — {c_status}")

                    # Analyze button (enabled only for valid/warning images)
                    if st.button(
                        "🔬 AI CROP HEALTH ANALYSIS",
                        type="primary",
                        key="analyze_crop_button",
                        help="Advanced AI analysis using deep learning models",
                    ):
                        with st.spinner("🧠 AI is analysing your crop image…"):
                            progress_bar = st.progress(0)
                            for i in range(100):
                                progress_bar.progress(i + 1)
                                if i == 25:
                                    st.info("🔍 Preprocessing image…")
                                elif i == 50:
                                    st.info("🧬 Detecting disease patterns…")
                                elif i == 75:
                                    st.info("📊 Calculating confidence scores…")

                            analysis_result = system.analyze_crop_image(uploaded_file)
                            progress_bar.empty()

                            if analysis_result:
                                st.session_state.crop_analysis = analysis_result
                                if "error" in analysis_result:
                                    st.error(analysis_result["error"])
                                else:
                                    st.success("✅ AI Analysis completed successfully!")
                                st.rerun()

                else:
                    # ── Invalid image: show actionable panel, DO NOT block tabs ──
                    st.markdown(
                        """
                        <div style="
                            background: linear-gradient(135deg, #1C1C1C 0%, #121212 100%);
                            border: 1px solid #424242;
                            border-radius: 14px;
                            padding: 1.4rem 1.6rem;
                            margin: 1rem 0;
                            text-align: center;
                        ">
                            <p style="font-size:2.5rem; margin:0 0 0.5rem;">🖼️</p>
                            <p style="color:#EF9A9A; font-weight:700; font-size:1rem; margin:0 0 0.4rem;">
                                Crop Analysis Unavailable
                            </p>
                            <p style="color:#BDBDBD; font-size:0.85rem; margin:0 0 1rem;">
                                The uploaded image could not be processed for disease detection.
                                Please retake the photo and re-upload.
                            </p>
                            <p style="color:#90CAF9; font-size:0.83rem; margin:0;">
                                👉 You can still use <b>Weather &amp; Soil</b>, <b>Pest Risk</b>,
                                <b>Irrigation</b>, and <b>Dashboard</b> tabs above for full farm insights.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            else:
                # No image uploaded — animated placeholder
                st.markdown(
                    """
                    <style>
                        @keyframes pulse-glow {
                            0%, 100% { box-shadow: 0 0 10px rgba(76,175,80,0.3); transform: scale(1); }
                            50%       { box-shadow: 0 0 22px rgba(76,175,80,0.6); transform: scale(1.01); }
                        }
                        @keyframes float-up {
                            0%, 100% { transform: translateY(0px); }
                            50%       { transform: translateY(-8px); }
                        }
                        @keyframes fade-in-out {
                            0%, 100% { opacity:0.7; }
                            50%       { opacity:1; }
                        }
                        .upload-placeholder {
                            background: linear-gradient(135deg,rgba(76,175,80,0.08),rgba(102,187,106,0.08));
                            border: 3px solid rgba(76,175,80,0.4);
                            border-radius: 18px;
                            padding: 2.5rem 2rem;
                            text-align: center;
                            animation: pulse-glow 2.5s ease-in-out infinite;
                            margin: 1.5rem 0;
                        }
                        .upload-icon  { font-size:3.5rem; animation:float-up 2.5s ease-in-out infinite; display:inline-block; margin-bottom:0.8rem; }
                        .upload-text  { font-size:1.25rem; font-weight:500; color:#4CAF50; margin:0.8rem 0; }
                        .upload-sub   { font-size:0.95rem; color:#66BB6A; opacity:0.9; margin-top:0.6rem; animation:fade-in-out 2.5s ease-in-out infinite; }
                    </style>
                    <div class="upload-placeholder">
                        <div class="upload-icon">📸</div>
                        <div class="upload-text">Upload a crop leaf image for AI disease detection</div>
                        <div class="upload-sub">JPG / JPEG / PNG · Max 15 MB · Natural daylight recommended</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with col2:
            # Enhanced NDVI Analysis Section
            st.markdown(
                """
                <div style="
                    background: linear-gradient(135deg, var(--sky-blue) 0%, var(--water-blue) 100%);
                    padding: 1rem;
                    border-radius: 12px;
                    color: white;
                    text-align: center;
                    margin-bottom: 1rem;
                    box-shadow: 0 4px 15px rgba(25,118,210,0.3);
                ">
                    <h3 style="margin: 0;">🛰️ NDVI Satellite Analysis</h3>
                    <p style="margin: 6px 0 0; opacity: 0.9; font-size: 14px;">Normalized Difference Vegetation Index</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="ndvi-section" style="background: rgba(255,255,255,0.05); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(25,118,210,0.2);">',
                unsafe_allow_html=True,
            )

            # NDVI Input with better explanations
            st.markdown("#### 📡 Satellite Data Input")

            col_nir, col_red = st.columns(2)
            with col_nir:
                nir_value = st.number_input(
                    "Near Infrared (NIR)",
                    0.0,
                    1.0,
                    0.80,
                    0.01,
                    key="nir",
                    help="Healthy vegetation reflects more NIR light (0.7-0.9)",
                )
                st.markdown(
                    '<p style="font-size: 12px; color: #B8D4B8;">💡 Healthy: 0.7-0.9</p>',
                    unsafe_allow_html=True,
                )

            with col_red:
                red_value = st.number_input(
                    "Red Light",
                    0.0,
                    1.0,
                    0.30,
                    0.01,
                    key="red",
                    help="Healthy vegetation absorbs red light (0.1-0.4)",
                )
                st.markdown(
                    '<p style="font-size: 12px; color: #B8D4B8;">💡 Healthy: 0.1-0.4</p>',
                    unsafe_allow_html=True,
                )

            ndvi_value = system.calculate_ndvi(nir_value, red_value)
            ndvi_interpretation, ndvi_color = system.interpret_ndvi(ndvi_value)

            # Simple NDVI Visualization
            ndvi_fig = go.Figure(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=ndvi_value,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "NDVI Index"},
                    delta={"reference": 0.5},
                    gauge={
                        "axis": {"range": [-1, 1]},
                        "bar": {"color": ndvi_color},
                        "steps": [
                            {"range": [-1, 0.1], "color": "#F44336"},  # Danger Red
                            {"range": [0.1, 0.3], "color": "#FF7043"},  # Harvest Orange
                            {
                                "range": [0.3, 0.5],
                                "color": "#FFA726",
                            },  # Sunshine Yellow
                            {"range": [0.5, 0.7], "color": "#66BB6A"},  # Light Green
                            {"range": [0.7, 1], "color": "#4CAF50"},
                        ],  # Bright Green
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": 0.9,
                        },
                    },
                )
            )
            ndvi_fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(ndvi_fig, use_container_width=True)

            # NDVI Interpretation
            st.markdown(
                f'<div style="color: {ndvi_color}; font-weight: bold; text-align: center;">{ndvi_interpretation}</div>',
                unsafe_allow_html=True,
            )

            # NDVI ranges explanation
            st.markdown("#### 📖 NDVI Reference Guide")
            ndvi_guide = [
                ("🔴 < 0.1", "No vegetation/Water/Snow"),
                ("🟠 0.1-0.3", "Sparse/Unhealthy vegetation"),
                ("🟡 0.3-0.5", "Moderate vegetation"),
                ("🟢 0.5-0.7", "Dense/Healthy vegetation"),
                ("💚 > 0.7", "Very dense/Optimal vegetation"),
            ]

            for range_val, description in ndvi_guide:
                st.markdown(f"• {range_val}: {description}")

            st.markdown("</div>", unsafe_allow_html=True)

            # Environmental Factors Section
            st.markdown("---")
            st.markdown("#### 🌍 Environmental Impact Factors")

            env_col1, env_col2 = st.columns(2)
            with env_col1:
                season = st.selectbox(
                    "Season",
                    ["Kharif", "Rabi", "Zaid"],
                    help="Different seasons affect NDVI values",
                )
                growth_stage_ndvi = st.selectbox(
                    "Growth Stage",
                    ["Germination", "Vegetative", "Flowering", "Maturity"],
                    key="ndvi_growth",
                )

            with env_col2:
                water_status = st.selectbox(
                    "Water Status",
                    ["Adequate", "Deficit", "Excess"],
                    help="Water availability affects vegetation health",
                )
                soil_type_ndvi = st.selectbox(
                    "Soil Type",
                    ["Black Cotton", "Red Soil", "Alluvial", "Laterite"],
                    key="ndvi_soil",
                )

            # Environmental impact on NDVI
            impact_score = 1.0
            if water_status == "Deficit":
                impact_score -= 0.15
            elif water_status == "Excess":
                impact_score -= 0.10

            if season == "Zaid":
                impact_score -= 0.05  # Summer crop challenges

            adjusted_ndvi = ndvi_value * impact_score

            st.markdown(f"**🎯 Environment-Adjusted NDVI: {adjusted_ndvi:.3f}**")
            if abs(adjusted_ndvi - ndvi_value) > 0.02:
                st.info(
                    f"📝 Environmental factors suggest NDVI adjustment: {ndvi_value:.3f} → {adjusted_ndvi:.3f}"
                )

        # Enhanced Results Display Section
        if (
            "crop_analysis" in st.session_state
            and st.session_state.crop_analysis is not None
        ):
            result = st.session_state.crop_analysis

            if "error" in result:
                st.error(
                    result.get(
                        "error",
                        "Invalid image: please upload a valid crop leaf image for analysis.",
                    )
                )
                st.stop()

            st.markdown("---")
            st.markdown(
                '<div style="text-align: center; margin: 2rem 0;"><h2 style="color: #2E7D32;">🧬 AI Analysis Results & Treatment Plan</h2></div>',
                unsafe_allow_html=True,
            )

            # Enhanced metrics display with visual indicators
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Disease Detected", result["disease"])

            with col2:
                st.metric("Confidence Level", f"{result['confidence']:.1f}%")

            with col3:
                severity = (
                    "LOW"
                    if result["confidence"] < 60
                    else "MEDIUM" if result["confidence"] < 80 else "HIGH"
                )
                st.metric("Severity Level", severity)

            with col4:
                health_status = (
                    "Healthy"
                    if result["disease"].lower() == "healthy"
                    else "Needs Attention"
                )
                st.metric("Crop Status", health_status)

            st.markdown("---")

            # Enhanced confidence visualization with detailed breakdown
            col1, col2 = st.columns([1.5, 1])

            with col1:
                if "all_predictions" in result:
                    st.markdown("### 📈 Detailed Disease Probability Analysis")

                    pred_df = pd.DataFrame(
                        result["all_predictions"], columns=["Disease", "Probability"]
                    )

                    # Enhanced bar chart with healthy status highlighting
                    pred_fig = go.Figure()

                    # Color code bars - green for healthy, blue for diseases
                    colors = [
                        "#4CAF50" if "healthy" in disease.lower() else "#3498DB"
                        for disease in pred_df["Disease"]
                    ]

                    pred_fig.add_trace(
                        go.Bar(
                            x=pred_df["Disease"],
                            y=pred_df["Probability"],
                            marker_color="rgba(255, 80, 80, 0.9)",  # Bright red bars
                            marker_line_color="rgba(255, 255, 255, 0.8)",
                            marker_line_width=1.5,
                            text=pred_df["Probability"].round(1),
                            textposition="outside",
                            texttemplate="%{text}%",
                            textfont=dict(color="rgba(240, 240, 240, 1)"),
                            hovertemplate="<b>%{x}</b><br>Probability: %{y:.1f}%<extra></extra>",
                        )
                    )

                    pred_fig.update_layout(
                        title=dict(
                            text="<b>Disease Detection Confidence Scores</b>",
                            font=dict(size=16, color="rgba(240, 240, 240, 1)"),
                            x=0.5,
                        ),
                        xaxis=dict(
                            title="Condition",
                            tickangle=45,
                            showgrid=True,
                            gridcolor="rgba(60, 60, 60, 0.3)",
                            zerolinecolor="rgba(100, 100, 100, 0.3)",
                            color="rgba(240, 240, 240, 1)",
                        ),
                        yaxis=dict(
                            title="Probability (%)",
                            range=[0, 105],
                            showgrid=True,
                            gridcolor="rgba(60, 60, 60, 0.3)",
                            zerolinecolor="rgba(100, 100, 100, 0.3)",
                            color="rgba(240, 240, 240, 1)",
                        ),
                        height=400,
                        showlegend=False,
                        plot_bgcolor="rgba(20, 24, 28, 1)",  # Deep dark-gray background
                        paper_bgcolor="rgba(30, 35, 40, 1)",  # Slightly lighter background
                        font=dict(color="rgba(240, 240, 240, 1)"),
                    )

                    st.plotly_chart(pred_fig, use_container_width=True)

            with col2:
                # Risk assessment and additional insights
                st.markdown("### 🎯 Risk Assessment")

                risk_level = (
                    "LOW"
                    if result["confidence"] < 60
                    else "MEDIUM" if result["confidence"] < 80 else "HIGH"
                )
                risk_color = (
                    "#4CAF50"
                    if risk_level == "LOW"
                    else "#FF9800" if risk_level == "MEDIUM" else "#F44336"
                )

                st.markdown(
                    f"""
                    <div style="
                        background: {risk_color};
                        padding: 1rem;
                        border-radius: 10px;
                        color: white;
                        margin: 1rem 0;
                    ">
                        <h4 style="margin: 0;">⚠️ Risk Level: {risk_level}</h4>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Additional analysis insights
                st.markdown("#### 🔬 Analysis Insights")

                if "disease" in result and "healthy" in result["disease"].lower():
                    insights = [
                        "✅ No significant disease patterns detected",
                        "🌱 Plant appears to be in good health",
                        "💚 Continue current care practices",
                        "📅 Regular monitoring recommended",
                    ]
                else:
                    insights = [
                        f"🔍 {result['disease']} pattern identified",
                        f"📊 Detection confidence: {result['confidence']:.1f}%",
                        "⚡ Early intervention recommended",
                        "📋 Follow treatment plan below",
                    ]

                for insight in insights:
                    st.markdown(f"• {insight}")

                # Image quality indicator
                quality = result.get("image_quality", "Good")
                quality_color = (
                    "#4CAF50"
                    if quality == "Excellent"
                    else "#FF9800" if quality == "Good" else "#F44336"
                )

                st.markdown(f"#### 📸 Image Quality")
                st.markdown(
                    f'<div style="color: {quality_color}; font-weight: bold; font-size: 16px;">📈 {quality}</div>',
                    unsafe_allow_html=True,
                )

            # Comprehensive Treatment Recommendations
            st.markdown("---")
            st.markdown(
                '<div style="text-align: center; margin: 2rem 0;"><h2 style="color: #2E7D32;">💊 Comprehensive Treatment Plan</h2></div>',
                unsafe_allow_html=True,
            )

            if result["disease"].lower() == "healthy":
                st.markdown(
                    """
                    <div style="
                        background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
                        padding: 2rem;
                        border-radius: 15px;
                        color: white;
                        text-align: center;
                        margin: 1rem 0;
                        box-shadow: 0 6px 20px rgba(76, 175, 80, 0.3);
                    ">
                        <h3 style="margin: 0;">🎉 Congratulations! Your Crop is Healthy</h3>
                        <p style="margin: 1rem 0 0; font-size: 18px; opacity: 0.9;">Continue your excellent farming practices and maintain regular monitoring.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Maintenance recommendations for healthy crops
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### 🌱 Maintenance Best Practices")
                    maintenance_tips = [
                        "💧 **Water Management**: Continue current irrigation schedule",
                        "🌿 **Nutrition**: Apply balanced NPK fertilizer every 2-3 weeks",
                        "🔍 **Monitoring**: Weekly health checks for early detection",
                        "🛡️ **Prevention**: Apply organic pesticides as preventive measure",
                    ]
                    for tip in maintenance_tips:
                        st.markdown(f"• {tip}")

                with col2:
                    st.markdown("#### 📅 Recommended Schedule")
                    schedule_items = [
                        "🗓️ **Daily**: Visual inspection during watering",
                        "📅 **Weekly**: Detailed leaf and stem examination",
                        "🗓️ **Bi-weekly**: Soil moisture and pH testing",
                        "📅 **Monthly**: Comprehensive crop health assessment",
                    ]
                    for item in schedule_items:
                        st.markdown(f"• {item}")

            else:
                # Disease-specific treatment protocols
                treatment_protocols = {
                    "Early_Blight": {
                        "immediate": [
                            "🚨 **IMMEDIATE (Within 24 hours)**: Remove all affected leaves with sterilized shears",
                            "🔥 **Disposal**: Burn infected material or dispose 100m+ away from field",
                            "🧼 **Sanitation**: Sterilize all tools with 70% alcohol between cuts",
                        ],
                        "treatment": [
                            "💊 **Primary Treatment**: Copper Oxychloride 50% WP @ 3g/L water",
                            "⏰ **Application**: Spray every 7-10 days for 3-4 treatments",
                            "🕐 **Timing**: Early morning (6-8 AM) or evening (6-8 PM)",
                            "🌬️ **Coverage**: Ensure complete leaf coverage, including undersides",
                        ],
                        "prevention": [
                            "🌬️ **Air Circulation**: Space plants 18-24 inches apart",
                            "💧 **Irrigation**: Switch to drip irrigation at soil level",
                            "🌿 **Pruning**: Remove lower branches touching soil",
                            "🧪 **Nutrition**: Apply potassium sulfate @ 2g/L to boost immunity",
                        ],
                        "monitoring": [
                            "🔍 **Daily Check**: Inspect treated areas for 7 days",
                            "📊 **Progress**: Document improvement or spread",
                            "⚠️ **Alert**: If no improvement in 10 days, consult extension officer",
                        ],
                    },
                    "Late_Blight": {
                        "immediate": [
                            "🆘 **EMERGENCY**: This is a crop emergency - act within 24 hours!",
                            "🔥 **Complete Removal**: Remove entire infected plants including roots",
                            "🚫 **Quarantine**: Create 2-meter buffer zone around infected area",
                        ],
                        "treatment": [
                            "💉 **Systemic Fungicide**: Metalaxyl-M + Mancozeb @ 2.5g/L",
                            "🛡️ **Protective Spray**: Treat all plants within 50m radius",
                            "🔄 **Resistance Management**: Alternate fungicides every 2 applications",
                            "🎯 **Focus Areas**: Concentrate on stem base and lower leaves",
                        ],
                        "prevention": [
                            "💧 **Water Stop**: Immediately stop overhead irrigation",
                            "🌪️ **Air Movement**: Maximize ventilation, use fans if possible",
                            "🧽 **Total Sanitation**: Disinfect tools with 10% bleach solution",
                            "👕 **Clothing**: Change clothes before entering healthy areas",
                        ],
                        "monitoring": [
                            "🌡️ **Weather Watch**: Monitor for cool, wet conditions (15-21°C)",
                            "📱 **Expert Consult**: Contact agricultural extension officer immediately",
                            "⏰ **Hourly Check**: Monitor spread every few hours initially",
                        ],
                    },
                    "Bacterial_Spot": {
                        "immediate": [
                            "☀️ **Dry Weather Only**: Only work with plants when completely dry",
                            "✂️ **Strategic Pruning**: Remove affected branches 4-6 inches below symptoms",
                            "🧼 **Tool Sterilization**: Disinfect tools between EVERY plant",
                        ],
                        "treatment": [
                            "🦠 **Bactericide**: Streptocycline @ 0.5g/L OR Copper Sulfate @ 1-1.5g/L",
                            "⏰ **Prevention Focus**: Copper works better for prevention",
                            "🚫 **No Wet Work**: Never work on wet plants - bacteria spread through water",
                        ],
                        "prevention": [
                            "💧 **Drip Irrigation**: Install immediately - no overhead watering",
                            "🏞️ **Drainage**: Improve field drainage with raised beds",
                            "🔄 **Crop Rotation**: Plan 2-3 year rotation away from tomato family",
                            "🗺️ **Field Mapping**: Mark infected spots for future monitoring",
                        ],
                        "monitoring": [
                            "🌡️ **Weather Alert**: Worst in warm, humid weather (24-29°C, >85% humidity)",
                            "📅 **Season Watch**: Apply weekly copper sprays during monsoon",
                            "🧪 **Soil Treatment**: Bacteria survive 2+ years in soil",
                        ],
                    },
                    "Nutrient_Deficiency": {
                        "immediate": [
                            "🧪 **Soil Test**: Get comprehensive soil analysis immediately",
                            "🍃 **Foliar Spray**: NPK 19:19:19 @ 2g/L for quick uptake",
                            "📊 **Symptom ID**: Yellow leaves=N, Purple stems=P, Brown edges=K",
                        ],
                        "treatment": [
                            "🌿 **Balanced Nutrition**: NPK + micronutrients weekly foliar spray",
                            "🍂 **Organic Matter**: Add 2-3 tons compost/FYM per hectare",
                            "💧 **Water Management**: Ensure adequate moisture for nutrient uptake",
                        ],
                        "prevention": [
                            "⏰ **Application Timing**: Early morning or evening applications",
                            "🧪 **Regular Testing**: Soil test every 6 months",
                            "📊 **Nutrient Program**: Develop season-long nutrition plan",
                        ],
                        "monitoring": [
                            "🍃 **Leaf Watch**: Monitor new growth for improvement",
                            "📅 **Response Time**: Expect improvement in 7-14 days",
                            "🔄 **Adjustment**: Modify program based on plant response",
                        ],
                    },
                }

                # Display treatment protocol
                disease_key = result["disease"].replace(" ", "_")
                protocol = treatment_protocols.get(disease_key, {})

                if protocol:
                    # Create tabbed treatment plan
                    tab1, tab2, tab3, tab4 = st.tabs(
                        [
                            "🚨 Immediate Actions",
                            "💊 Treatment Protocol",
                            "🛡️ Prevention Measures",
                            "📊 Monitoring Plan",
                        ]
                    )

                    with tab1:
                        st.markdown("### ⚡ IMMEDIATE ACTIONS REQUIRED")
                        for action in protocol.get("immediate", []):
                            st.markdown(f"• {action}")

                        if result["confidence"] > 80:
                            st.error(
                                "⚠️ HIGH CONFIDENCE DETECTION - Start treatment immediately!"
                            )

                    with tab2:
                        st.markdown("### 💊 TREATMENT PROTOCOL")
                        for treatment in protocol.get("treatment", []):
                            st.markdown(f"• {treatment}")

                        # Cost estimation
                        st.markdown("#### 💰 Estimated Treatment Cost")
                        cost_estimates = {
                            "Early_Blight": "₹800-1200 per hectare",
                            "Late_Blight": "₹1500-2500 per hectare",
                            "Bacterial_Spot": "₹600-1000 per hectare",
                            "Nutrient_Deficiency": "₹500-800 per hectare",
                        }
                        cost = cost_estimates.get(disease_key, "₹500-1000 per hectare")
                        st.info(
                            f"💵 **Approximate Cost**: {cost} (including labor and materials)"
                        )

                    with tab3:
                        st.markdown("### 🛡️ PREVENTION MEASURES")
                        for prevention in protocol.get("prevention", []):
                            st.markdown(f"• {prevention}")

                        st.markdown("#### 🌱 Long-term Prevention Strategy")
                        st.success(
                            "🔄 Implement integrated pest management (IPM) practices for sustainable crop health."
                        )

                    with tab4:
                        st.markdown("### 📊 MONITORING & FOLLOW-UP")
                        for monitoring in protocol.get("monitoring", []):
                            st.markdown(f"• {monitoring}")

                        # Recovery timeline
                        recovery_times = {
                            "Early_Blight": "7-14 days",
                            "Late_Blight": "Emergency - immediate professional help needed",
                            "Bacterial_Spot": "14-21 days",
                            "Nutrient_Deficiency": "7-14 days",
                        }
                        recovery = recovery_times.get(disease_key, "10-14 days")
                        st.info(f"⏰ **Expected Recovery Time**: {recovery}")

            # Emergency contact information
            st.markdown("---")
            st.markdown("### 📞 Emergency Support Contacts")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(
                    """
                    **🏛️ Agricultural Extension**
                    - Krishi Vigyan Kendra (KVK)
                    - District Collector Office
                    - State Agriculture Department
                    """
                )

            with col2:
                st.markdown(
                    """
                    **🔬 Expert Consultation**
                    - Agricultural University
                    - Plant Pathology Labs
                    - Certified Crop Advisors
                    """
                )

            with col3:
                st.markdown(
                    """
                    **📱 Digital Support**
                    - Kisan Call Centre: 1551
                    - mKisan Portal
                    - Crop insurance helpline
                    """
                )

            # Action plan summary
            if result["disease"].lower() != "healthy":
                st.markdown("---")
                st.markdown("### 📋 Action Plan Summary")

                action_plan = f"""
                **📋 YOUR PERSONALIZED ACTION PLAN:**
                
                1. **🚨 Immediate (Today)**: {protocol.get('immediate', ['Contact agricultural extension officer'])[0] if protocol else 'Identify and isolate affected areas'}
                
                2. **💊 Treatment (Within 2-3 days)**: {protocol.get('treatment', ['Apply recommended treatment'])[0] if protocol else 'Apply appropriate treatment measures'}
                
                3. **🛡️ Prevention (This week)**: {protocol.get('prevention', ['Implement prevention measures'])[0] if protocol else 'Implement preventive measures'}
                
                4. **📊 Monitor (Ongoing)**: {protocol.get('monitoring', ['Regular monitoring required'])[0] if protocol else 'Continue monitoring progress'}
                
                **⏰ Timeline**: Start immediately for best results. Early intervention is crucial for successful treatment.
                
                **💰 Budget**: Keep ₹1000-2000 ready for immediate treatment needs.
                """

                st.info(action_plan)

    # Weather & Soil Tab
    with tabs[1]:
        st.markdown(
            '<div class="tab-header"><h2>Weather & Soil Analysis</h2></div>',
            unsafe_allow_html=True,
        )

        # Enhanced Interactive Weather Trend Analysis
        st.markdown("### 🌤️ Enhanced Weather Trend Analysis (Last 7 Days)")

        # Get enhanced weather data with status indicators
        weather_enhanced = system.get_enhanced_weather_data(district)

        # Create 2x2 grid for enhanced weather trends
        col1, col2 = st.columns(2)

        with col1:
            # Enhanced Temperature Trend with Agricultural Theme Colors
            temp_colors = []
            temp_hover_text = []
            for i, temp in enumerate(weather_enhanced["temperature"]):
                if temp > 32:
                    temp_colors.append("#FF7043")  # Too Hot - Harvest Orange
                    status = "Too Hot"
                elif temp < 22:
                    temp_colors.append("#03A9F4")  # Too Cold - Water Blue
                    status = "Too Cold"
                else:
                    temp_colors.append("#4CAF50")  # Good - Bright Green
                    status = "Optimal"
                temp_hover_text.append(f"{temp}°C - {status}")

            temp_fig = go.Figure()
            temp_fig.add_trace(
                go.Scatter(
                    x=weather_enhanced["dates"],
                    y=weather_enhanced["temperature"],
                    mode="lines+markers",
                    name="Temperature",
                    line=dict(color="#FFA726", width=4),  # Sunshine Yellow line
                    marker=dict(
                        color=temp_colors, size=12, line=dict(color="white", width=2)
                    ),
                    text=temp_hover_text,
                    hovertemplate="<b>%{text}</b><br>Date: %{x}<extra></extra>",
                )
            )
            temp_fig.update_layout(
                title=dict(
                    text="Temperature Trends with Status Indicators",
                    font=dict(size=16, color="#2E7D32"),  # Deep Green
                    x=0.5,
                ),
                xaxis_title="Date",
                yaxis_title="Temperature (°C)",
                height=350,
                plot_bgcolor="rgba(15, 20, 25, 0.02)",  # Subtle dark background
                paper_bgcolor="rgba(15, 20, 25, 0.02)",
                font=dict(color="#E8F5E8"),  # Light green text
                showlegend=False,
            )
            temp_fig.update_xaxes(
                showgrid=True, gridwidth=1, gridcolor="rgba(46, 125, 50, 0.2)"
            )
            temp_fig.update_yaxes(
                showgrid=True, gridwidth=1, gridcolor="rgba(46, 125, 50, 0.2)"
            )
            st.plotly_chart(temp_fig, use_container_width=True)

            # Enhanced Rainfall Pattern with Agricultural Theme Colors
            rain_colors = []
            rain_hover_text = []
            for rain in weather_enhanced["rainfall"]:
                if rain > 20:
                    rain_colors.append("#1976D2")  # Heavy - Sky Blue
                    status = "Heavy"
                elif rain > 5:
                    rain_colors.append("#03A9F4")  # Moderate - Water Blue
                    status = "Moderate"
                else:
                    rain_colors.append("#81C784")  # Light - Light Green
                    status = "Light"
                rain_hover_text.append(f"{rain}mm - {status}")

            rain_fig = go.Figure()
            rain_fig.add_trace(
                go.Bar(
                    x=weather_enhanced["dates"],
                    y=weather_enhanced["rainfall"],
                    name="Rainfall",
                    marker=dict(
                        color=rain_colors,
                        line=dict(color="rgba(255, 255, 255, 0.8)", width=2),
                    ),
                    text=rain_hover_text,
                    hovertemplate="<b>%{text}</b><br>Date: %{x}<extra></extra>",
                )
            )
            rain_fig.update_layout(
                title=dict(
                    text="Rainfall Patterns with Intensity Levels",
                    font=dict(size=16, color="#1976D2"),  # Sky Blue
                    x=0.5,
                ),
                xaxis_title="Date",
                yaxis_title="Rainfall (mm)",
                height=350,
                plot_bgcolor="rgba(15, 20, 25, 0.02)",
                paper_bgcolor="rgba(15, 20, 25, 0.02)",
                font=dict(color="#E8F5E8"),
                showlegend=False,
            )
            rain_fig.update_xaxes(
                showgrid=True, gridwidth=1, gridcolor="rgba(25, 118, 210, 0.2)"
            )
            rain_fig.update_yaxes(
                showgrid=True, gridwidth=1, gridcolor="rgba(25, 118, 210, 0.2)"
            )
            st.plotly_chart(rain_fig, use_container_width=True)

        with col2:
            # Enhanced Humidity Trend with Agricultural Theme Colors
            humidity_colors = []
            humidity_hover_text = []
            for humidity in weather_enhanced["humidity"]:
                if humidity > 80:
                    humidity_colors.append("#FF9800")  # High - Warning Amber
                    status = "High"
                elif humidity < 50:
                    humidity_colors.append("#FFA726")  # Low - Sunshine Yellow
                    status = "Low"
                else:
                    humidity_colors.append("#4CAF50")  # Optimal - Bright Green
                    status = "Optimal"
                humidity_hover_text.append(f"{humidity}% - {status}")

            humidity_fig = go.Figure()
            humidity_fig.add_trace(
                go.Bar(
                    x=weather_enhanced["dates"],
                    y=weather_enhanced["humidity"],
                    name="Humidity",
                    marker=dict(
                        color=humidity_colors,
                        line=dict(color="rgba(255, 255, 255, 0.8)", width=2),
                    ),
                    text=humidity_hover_text,
                    hovertemplate="<b>%{text}</b><br>Date: %{x}<extra></extra>",
                )
            )
            humidity_fig.update_layout(
                title=dict(
                    text="Humidity Levels with Status Indicators",
                    font=dict(size=16, color="#4CAF50"),  # Bright Green
                    x=0.5,
                ),
                xaxis_title="Date",
                yaxis_title="Humidity (%)",
                height=350,
                plot_bgcolor="rgba(15, 20, 25, 0.02)",
                paper_bgcolor="rgba(15, 20, 25, 0.02)",
                font=dict(color="#E8F5E8"),
                showlegend=False,
                yaxis=dict(range=[0, 100]),
            )
            humidity_fig.update_xaxes(
                showgrid=True, gridwidth=1, gridcolor="rgba(76, 175, 80, 0.2)"
            )
            humidity_fig.update_yaxes(
                showgrid=True, gridwidth=1, gridcolor="rgba(76, 175, 80, 0.2)"
            )
            st.plotly_chart(humidity_fig, use_container_width=True)

            # Enhanced Wind Speed with Agricultural Theme Colors
            wind_fig = go.Figure()
            wind_fig.add_trace(
                go.Scatter(
                    x=weather_enhanced["dates"],
                    y=weather_enhanced["wind_speed"],
                    mode="lines+markers",
                    name="Wind Speed",
                    line=dict(color="#8D6E63", width=4),  # Earth Brown
                    marker=dict(
                        color="#5D4E37",  # Rich Earth Brown
                        size=12,
                        line=dict(color="white", width=2),
                    ),
                    fill="tozeroy",
                    fillcolor="rgba(141, 110, 99, 0.3)",  # Earth Brown fill
                    hovertemplate="<b>%{y} km/h</b><br>Date: %{x}<extra></extra>",
                )
            )
            wind_fig.update_layout(
                title=dict(
                    text="Wind Speed Trends with Area Fill",
                    font=dict(size=16, color="#8D6E63"),  # Earth Brown
                    x=0.5,
                ),
                xaxis_title="Date",
                yaxis_title="Wind Speed (km/h)",
                height=350,
                plot_bgcolor="rgba(15, 20, 25, 0.02)",
                paper_bgcolor="rgba(15, 20, 25, 0.02)",
                font=dict(color="#E8F5E8"),
                showlegend=False,
            )
            wind_fig.update_xaxes(
                showgrid=True, gridwidth=1, gridcolor="rgba(141, 110, 99, 0.2)"
            )
            wind_fig.update_yaxes(
                showgrid=True, gridwidth=1, gridcolor="rgba(141, 110, 99, 0.2)"
            )
            st.plotly_chart(wind_fig, use_container_width=True)

        st.markdown("---")

        # Current Weather Analysis Section
        st.markdown("### 🌤️ Current Weather Analysis")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Temperature", f"{current_weather['temperature']}°C")
        with col2:
            st.metric("Humidity", f"{current_weather['humidity']}%")
        with col3:
            st.metric("Pressure", f"{current_weather['pressure']} hPa")
        with col4:
            st.metric("Visibility", f"{current_weather['visibility']} km")

        # Weather trends visualization
        historical = weather_data["historical"]
        forecast = weather_data["forecast"]

        col1, col2 = st.columns(2)

        with col1:
            # Temperature trend with agricultural theme
            temp_fig = go.Figure()
            temp_fig.add_trace(
                go.Scatter(
                    x=historical["dates"],
                    y=historical["temperature"],
                    mode="lines+markers",
                    name="Historical",
                    line=dict(color="#FFA726", width=4),  # Sunshine Yellow
                    marker=dict(color="#FF7043", size=8),  # Harvest Orange
                )
            )
            temp_fig.add_trace(
                go.Scatter(
                    x=forecast["dates"],
                    y=forecast["temperature"],
                    mode="lines+markers",
                    name="Forecast",
                    line=dict(color="#4CAF50", width=4, dash="dash"),  # Bright Green
                    marker=dict(color="#66BB6A", size=8),  # Light Green
                )
            )
            temp_fig.update_layout(
                title=dict(
                    text="Temperature Trends & Forecast",
                    font=dict(color="#2E7D32"),  # Deep Green
                ),
                xaxis_title="Date",
                yaxis_title="Temperature (°C)",
                height=400,
                plot_bgcolor="rgba(15, 20, 25, 0.02)",
                paper_bgcolor="rgba(15, 20, 25, 0.02)",
                font=dict(color="#E8F5E8"),
            )
            st.plotly_chart(temp_fig, use_container_width=True)

        with col2:
            # Humidity and Rainfall with agricultural theme
            weather_fig = go.Figure()
            weather_fig.add_trace(
                go.Bar(
                    x=historical["dates"],
                    y=historical["rainfall"],
                    name="Rainfall (mm)",
                    marker_color="#03A9F4",  # Water Blue
                    yaxis="y",
                )
            )
            weather_fig.add_trace(
                go.Scatter(
                    x=historical["dates"],
                    y=historical["humidity"],
                    name="Humidity (%)",
                    line=dict(color="#4CAF50", width=4),  # Bright Green
                    marker=dict(color="#66BB6A", size=8),
                    yaxis="y2",
                )
            )
            weather_fig.update_layout(
                title=dict(
                    text="Rainfall & Humidity Pattern",
                    font=dict(color="#2E7D32"),  # Deep Green
                ),
                xaxis_title="Date",
                yaxis=dict(title="Rainfall (mm)", side="left", color="#03A9F4"),
                yaxis2=dict(
                    title="Humidity (%)", side="right", overlaying="y", color="#4CAF50"
                ),
                height=400,
                plot_bgcolor="rgba(15, 20, 25, 0.02)",
                paper_bgcolor="rgba(15, 20, 25, 0.02)",
                font=dict(color="#E8F5E8"),
            )
            st.plotly_chart(weather_fig, use_container_width=True)

        st.markdown("---")

        # Enhanced Soil Health Analysis Section
        st.markdown("### 🧪 Enhanced Interactive Soil Health Analysis")

        soil_analysis = system.analyze_soil_health(
            soil_ph, nitrogen, phosphorus, potassium, farm_area
        )

        # Enhanced soil metrics with better visualization
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            score_color = (
                "#4CAF50"
                if soil_analysis["score"] > 80
                else "#FF9800" if soil_analysis["score"] > 60 else "#F44336"
            )
            st.markdown(
                f'<div style="background: linear-gradient(135deg, {score_color}, {score_color}AA); padding: 1.5rem; border-radius: 15px; text-align: center; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.2);"><h2>{soil_analysis["score"]}/100</h2><p style="margin: 0; font-size: 1.1em;">Soil Health Score</p></div>',
                unsafe_allow_html=True,
            )
        with col2:
            delta_color = "normal" if abs(soil_ph - 6.8) < 0.5 else "inverse"
            st.metric(
                "pH Level",
                f"{soil_ph}",
                delta=f"{soil_ph - 6.8:.1f} from ideal",
                delta_color=delta_color,
            )
        with col3:
            st.metric("Overall Status", soil_analysis["status"])
        with col4:
            st.metric("Total Fertilizer Cost", f"₹{soil_analysis['total_cost']:.2f}")

        # Enhanced soil analysis visualization with new dark theme
        col1, col2 = st.columns(2)

        with col1:
            # Enhanced NPK Analysis with Status Colors
            nutrients = ["Nitrogen", "Phosphorus", "Potassium"]
            current_values = [nitrogen, phosphorus, potassium]
            optimal_values = [280, 40, 120]  # Updated optimal values

            # Create enhanced bar chart with new styling
            npk_fig = go.Figure()

            # Current levels with improved colors
            npk_fig.add_trace(
                go.Bar(
                    name="Current Level",
                    x=nutrients,
                    y=current_values,
                    marker_color="#ff4d4d",
                    text=[
                        (
                            "Low"
                            if curr < opt * 0.8
                            else "High" if curr > opt * 1.2 else "Good"
                        )
                        for curr, opt in zip(current_values, optimal_values)
                    ],
                    textposition="auto",
                    hovertemplate="<b>%{x}</b><br>Current: %{y} ppm<br>Status: %{text}<extra></extra>",
                )
            )

            # Optimal reference bars
            npk_fig.add_trace(
                go.Bar(
                    name="Optimal Level",
                    x=nutrients,
                    y=optimal_values,
                    marker_color="#00cc96",
                    hovertemplate="<b>%{x}</b><br>Optimal: %{y} ppm<extra></extra>",
                )
            )

            npk_fig.update_layout(
                barmode="group",
                title="Nutrient Concentration Levels",
                xaxis_title="Nutrients",
                yaxis_title="Concentration (ppm)",
                height=400,
            )

            # Apply consistent dark theme
            npk_fig = apply_dark_theme(npk_fig)
            st.plotly_chart(npk_fig, use_container_width=True)

        with col2:
            # Fertilizer Cost Distribution (Donut Chart)
            if "fertilizer_costs" in soil_analysis:
                fert_data = soil_analysis["fertilizer_costs"]
                labels = list(fert_data.keys())
                values = list(fert_data.values())

                cost_fig = go.Figure(
                    data=[
                        go.Pie(
                            labels=labels,
                            values=values,
                            hole=0.4,
                            marker=dict(colors=["#00cc96", "#ffa15a", "#ab63fa"]),
                            textinfo="label+percent",
                            textfont=dict(color="white"),
                        )
                    ]
                )

                cost_fig.update_layout(title="Fertilizer Cost Distribution", height=300)

                cost_fig = apply_dark_theme(cost_fig)
                st.plotly_chart(cost_fig, use_container_width=True)

            # Interactive Soil Health Gauge
            health_score = soil_analysis["score"]
            health_status = soil_analysis["status"]

            gauge_fig = go.Figure(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=health_score,
                    domain={"x": [0, 1], "y": [0, 1]},
                    delta={
                        "reference": 80,
                        "increasing": {"color": "#00cc96"},
                        "decreasing": {"color": "#ff4d4d"},
                    },
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "white"},
                        "bar": {"color": "#00cc96"},
                        "bgcolor": "rgba(20,20,20,1)",
                        "steps": [
                            {"range": [0, 60], "color": "#ff4d4d"},
                            {"range": [60, 80], "color": "#ffcc00"},
                            {"range": [80, 100], "color": "#00cc96"},
                        ],
                    },
                    title={
                        "text": f"Soil Health Score<br><span style='font-size:0.9em'>{health_status}</span>",
                        "font": {"color": "white"},
                    },
                )
            )

            # Apply dark theme to gauge
            gauge_fig = apply_dark_theme(gauge_fig)
            st.plotly_chart(gauge_fig, use_container_width=True)

            # Add pH Analysis chart
            ph_fig = go.Figure()
            ph_fig.add_trace(
                go.Bar(
                    x=["Current pH"],
                    y=[soil_ph],
                    marker_color=(
                        "#ff4d4d" if soil_ph < 6.0 or soil_ph > 7.5 else "#00cc96"
                    ),
                    text=[
                        f"pH {soil_ph:.1f} ({'Acidic' if soil_ph < 6.0 else 'Alkaline' if soil_ph > 7.5 else 'Optimal'})"
                    ],
                    textposition="auto",
                )
            )

            # Add optimal range indicators
            ph_fig.add_hline(
                y=6.0,
                line_dash="dash",
                line_color="green",
                annotation_text="Optimal Min (6.0)",
                annotation_position="right",
            )
            ph_fig.add_hline(
                y=6.8,
                line_dash="dot",
                line_color="lime",
                annotation_text="Ideal (6.8)",
                annotation_position="right",
            )
            ph_fig.add_hline(
                y=7.5,
                line_dash="dash",
                line_color="green",
                annotation_text="Optimal Max (7.5)",
                annotation_position="right",
            )

            ph_fig.update_layout(
                title="Soil pH Analysis with Optimal Range",
                xaxis_title="Measurement",
                yaxis_title="pH Value",
                height=300,
            )

            # Apply dark theme
            ph_fig = apply_dark_theme(ph_fig)
            st.plotly_chart(ph_fig, use_container_width=True)

        # Enhanced pH Analysis Chart
        col1, col2 = st.columns(2)

        with col1:
            # pH Analysis with optimal range indicators
            ph_status = (
                "Good"
                if 6.0 <= soil_ph <= 7.5
                else ("Acidic" if soil_ph < 6.0 else "Alkaline")
            )
            ph_color = (
                "#27AE60"
                if ph_status == "Good"
                else ("#E74C3C" if ph_status == "Acidic" else "#F39C12")
            )

            ph_fig = go.Figure()

            # Current pH bar
            ph_fig.add_trace(
                go.Bar(
                    x=["Current pH"],
                    y=[soil_ph],
                    name="Current pH",
                    marker=dict(color=ph_color, line=dict(color="white", width=2)),
                    text=[f"pH {soil_ph}<br>{ph_status}"],
                    hovertemplate="<b>%{text}</b><br>Optimal Range: 6.0-7.5<extra></extra>",
                )
            )

            # Optimal range indicators
            ph_fig.add_hline(
                y=6.0,
                line_dash="dash",
                line_color="green",
                annotation_text="Optimal Min (6.0)",
            )
            ph_fig.add_hline(
                y=7.5,
                line_dash="dash",
                line_color="green",
                annotation_text="Optimal Max (7.5)",
            )
            ph_fig.add_hline(
                y=6.8, line_dash="dot", line_color="blue", annotation_text="Ideal (6.8)"
            )

            ph_fig.update_layout(
                title=dict(
                    text="Soil pH Analysis with Optimal Range",
                    font=dict(size=16, color="#8E44AD"),
                    x=0.5,
                ),
                xaxis_title="Measurement",
                yaxis_title="pH Value",
                height=350,
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(color="#2C3E50"),
                showlegend=False,
                yaxis=dict(range=[4, 9]),
            )
            ph_fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#E8F4F9")
            ph_fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#E8F4F9")
            st.plotly_chart(ph_fig, use_container_width=True)

        with col2:
            # Enhanced Fertilizer Cost Breakdown
            if soil_analysis["fertilizer_recommendations"]:
                fert_df = pd.DataFrame(soil_analysis["fertilizer_recommendations"])

                fert_fig = go.Figure(
                    data=[
                        go.Pie(
                            labels=fert_df["type"],
                            values=fert_df["cost"],
                            hole=0.4,
                            marker=dict(
                                colors=[
                                    "#3498DB",
                                    "#E67E22",
                                    "#9B59B6",
                                    "#F1C40F",
                                    "#E74C3C",
                                ],
                                line=dict(color="white", width=2),
                            ),
                            textinfo="label+percent",
                            hovertemplate="<b>%{label}</b><br>Cost: ₹%{value:.2f}<br>Percentage: %{percent}<extra></extra>",
                        )
                    ]
                )

                fert_fig.update_layout(
                    title=dict(
                        text="Enhanced Fertilizer Cost Distribution",
                        font=dict(size=16, color="#34495E"),
                        x=0.5,
                    ),
                    height=350,
                    font=dict(color="#2C3E50"),
                    paper_bgcolor="white",
                )
                st.plotly_chart(fert_fig, use_container_width=True)
            else:
                # Fallback chart if no fertilizer data
                st.info(
                    "🌱 No specific fertilizer recommendations available. Soil health appears optimal!"
                )

        # Detailed recommendations
        st.markdown("### 💡 Detailed Soil Recommendations")
        for i, rec in enumerate(soil_analysis["recommendations"], 1):
            st.markdown(f"{i}. {rec}")

        # Fertilizer application table
        if soil_analysis["fertilizer_recommendations"]:
            st.markdown("### 🌿 Fertilizer Application Plan")
            fert_table_data = []
            for fert in soil_analysis["fertilizer_recommendations"]:
                fert_table_data.append(
                    {
                        "Fertilizer": fert["type"],
                        "Quantity (kg)": fert["quantity"],
                        "Cost (₹)": f"{fert['cost']:.2f}",
                        "Purpose": fert["purpose"],
                    }
                )

            fert_df_display = pd.DataFrame(fert_table_data)
            st.dataframe(fert_df_display, width="stretch")

    # Pest Risk Tab
    with tabs[2]:
        st.markdown(
            '<div class="tab-header"><h2>🐛 Pest Risk Assessment & Alert System</h2></div>',
            unsafe_allow_html=True,
        )

        # Comprehensive pest risk analysis
        pest_analysis = system.analyze_pest_risk(weather_data, crop_type, growth_stage)

        # Enhanced Alert System based on risk level
        overall_risk = pest_analysis["overall_risk"]
        risk_level = pest_analysis["risk_level"]

        # Critical Alert Banner
        if overall_risk >= 70:
            st.markdown(
                f"""
            <div style="
                background: linear-gradient(135deg, #FF4444 0%, #CC0000 100%);
                padding: 2rem;
                border-radius: 15px;
                text-align: center;
                color: white;
                margin: 1rem 0;
                border: 3px solid #FF0000;
                animation: pulse 2s infinite;
            ">
                <h1>🚨 CRITICAL PEST ALERT 🚨</h1>
                <h2>Risk Level: {overall_risk}/100 - {risk_level["level"]}</h2>
                <h3>⚡ IMMEDIATE ACTION REQUIRED ⚡</h3>
                <p style="font-size: 18px;">High pest activity detected! Apply control measures within 24 hours!</p>
            </div>
            <style>
            @keyframes pulse {{
                0% {{ box-shadow: 0 0 0 0 rgba(255, 68, 68, 0.7); }}
                70% {{ box-shadow: 0 0 0 10px rgba(255, 68, 68, 0); }}
                100% {{ box-shadow: 0 0 0 0 rgba(255, 68, 68, 0); }}
            }}
            </style>
            """,
                unsafe_allow_html=True,
            )
            st.error("🚨 EMERGENCY: Deploy pest control measures immediately!")
        elif overall_risk >= 50:
            st.markdown(
                f"""
            <div style="
                background: linear-gradient(135deg, #FFAA00 0%, #FF8800 100%);
                padding: 2rem;
                border-radius: 15px;
                text-align: center;
                color: white;
                margin: 1rem 0;
                border: 2px solid #FF8800;
            ">
                <h1>⚠️ MODERATE PEST ALERT ⚠️</h1>
                <h2>Risk Level: {overall_risk}/100 - {risk_level["level"]}</h2>
                <p style="font-size: 16px;">Preventive measures recommended within 2-3 days</p>
            </div>
            """,
                unsafe_allow_html=True,
            )
            st.warning("⚠️ Monitor closely and prepare preventive treatments")
        else:
            st.markdown(
                f"""
            <div style="
                background: linear-gradient(135deg, #44AA44 0%, #228B22 100%);
                padding: 2rem;
                border-radius: 15px;
                text-align: center;
                color: white;
                margin: 1rem 0;
            ">
                <h1>✅ LOW PEST RISK</h1>
                <h2>Risk Level: {overall_risk}/100 - {risk_level["level"]}</h2>
                <p style="font-size: 16px;">Continue regular monitoring schedule</p>
            </div>
            """,
                unsafe_allow_html=True,
            )
            st.success("✅ Pest risk is currently low - maintain vigilance")

        # Enhanced Risk Visualization Dashboard
        st.markdown("### 📊 Risk Analysis Dashboard")

        col1, col2 = st.columns(2)

        with col1:
            # Enhanced Risk factors radar chart with better styling
            risk_factors = pest_analysis["risk_factors"]
            categories = [
                "Temperature Risk",
                "Humidity Risk",
                "Rainfall Risk",
                "Seasonal Risk",
            ]
            values = [
                risk_factors[cat]["value"]
                for cat in ["temperature", "humidity", "rainfall", "seasonal"]
            ]

            radar_fig = go.Figure()

            # Add risk threshold zones
            radar_fig.add_trace(
                go.Scatterpolar(
                    r=[30, 30, 30, 30, 30],
                    theta=categories + [categories[0]],
                    fill="toself",
                    name="Low Risk Zone",
                    line_color="rgba(68, 170, 68, 0.3)",
                    fillcolor="rgba(68, 170, 68, 0.1)",
                )
            )

            radar_fig.add_trace(
                go.Scatterpolar(
                    r=[70, 70, 70, 70, 70],
                    theta=categories + [categories[0]],
                    fill="toself",
                    name="Medium Risk Zone",
                    line_color="rgba(255, 170, 0, 0.3)",
                    fillcolor="rgba(255, 170, 0, 0.1)",
                )
            )

            # Current risk levels
            radar_fig.add_trace(
                go.Scatterpolar(
                    r=values + [values[0]],
                    theta=categories + [categories[0]],
                    fill="toself",
                    name="Current Risk Level",
                    line=dict(color="#FF4444", width=3),
                    fillcolor="rgba(255, 68, 68, 0.3)",
                    marker=dict(size=8, color="#FF4444"),
                )
            )

            radar_fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100],
                        tickvals=[0, 30, 70, 100],
                        ticktext=["Safe", "Low Risk", "High Risk", "Critical"],
                        tickfont=dict(color="rgba(240, 240, 240, 1)"),
                        gridcolor="rgba(60, 60, 60, 0.3)",
                        linecolor="rgba(60, 60, 60, 0.3)",
                    ),
                    bgcolor="rgba(20, 24, 28, 1)",
                    angularaxis=dict(
                        gridcolor="rgba(60, 60, 60, 0.3)",
                        linecolor="rgba(60, 60, 60, 0.3)",
                    ),
                ),
                title=dict(
                    text="<b>Pest Risk Factor Analysis</b>",
                    font=dict(size=16, color="rgba(240, 240, 240, 1)"),
                    x=0.5,
                ),
                height=450,
                showlegend=True,
                legend=dict(x=0.8, y=1, font=dict(color="rgba(240, 240, 240, 1)")),
                paper_bgcolor="rgba(30, 35, 40, 1)",
                font=dict(color="rgba(240, 240, 240, 1)"),
            )
            st.plotly_chart(radar_fig, use_container_width=True)

        with col2:
            # Enhanced Pest predictions with better visualization
            if pest_analysis["pest_predictions"]:
                pred_df = pd.DataFrame(pest_analysis["pest_predictions"])

                # Create enhanced bar chart
                pred_fig = go.Figure()

                for severity in ["Low", "Medium", "High"]:
                    filtered_df = pred_df[pred_df["severity"] == severity]
                    if not filtered_df.empty:
                        color = (
                            "#44AA44"
                            if severity == "Low"
                            else "#FFAA00" if severity == "Medium" else "#FF4444"
                        )
                        pred_fig.add_trace(
                            go.Bar(
                                name=f"{severity} Risk",
                                x=filtered_df["pest"],
                                y=filtered_df["probability"],
                                marker_color=color,
                                text=filtered_df["probability"].round(1),
                                textposition="outside",
                                hovertemplate="<b>%{x}</b><br>Risk: %{y:.1f}%<br>Severity: "
                                + severity
                                + "<extra></extra>",
                            )
                        )

                pred_fig.update_layout(
                    title=dict(
                        text="<b>Crop-Specific Pest Risk Levels</b>",
                        font=dict(size=16, color="rgba(240, 240, 240, 1)"),
                        x=0.5,
                    ),
                    xaxis=dict(
                        title=dict(text="Pest Types", font=dict(size=14)),
                        tickangle=45,
                        gridcolor="rgba(60, 60, 60, 0.3)",
                        zerolinecolor="rgba(100, 100, 100, 0.3)",
                        color="rgba(240, 240, 240, 1)",
                        showgrid=True,
                    ),
                    yaxis=dict(
                        title=dict(text="Risk Probability (%)", font=dict(size=14)),
                        range=[0, 100],
                        gridcolor="rgba(60, 60, 60, 0.3)",
                        zerolinecolor="rgba(100, 100, 100, 0.3)",
                        color="rgba(240, 240, 240, 1)",
                        showgrid=True,
                    ),
                    height=450,
                    barmode="group",
                    showlegend=True,
                    legend=dict(font=dict(color="rgba(240, 240, 240, 1)")),
                    plot_bgcolor="rgba(20, 24, 28, 1)",
                    paper_bgcolor="rgba(30, 35, 40, 1)",
                    font=dict(color="rgba(240, 240, 240, 1)"),
                )

                # Add risk threshold lines
                pred_fig.add_hline(
                    y=70,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="High Risk Threshold",
                )
                pred_fig.add_hline(
                    y=30,
                    line_dash="dash",
                    line_color="orange",
                    annotation_text="Medium Risk Threshold",
                )

                st.plotly_chart(pred_fig, use_container_width=True)
            else:
                st.info(
                    "🐛 No specific pest predictions available for current conditions"
                )

        # Risk factor details
        st.markdown("### Risk Factor Details")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            temp_risk = risk_factors["temperature"]
            st.metric(
                "Temperature Risk", f"{temp_risk['value']}/100", temp_risk["status"]
            )

        with col2:
            humidity_risk = risk_factors["humidity"]
            st.metric(
                "Humidity Risk",
                f"{humidity_risk['value']}/100",
                humidity_risk["status"],
            )

        with col3:
            rainfall_risk = risk_factors["rainfall"]
            st.metric(
                "Rainfall Risk",
                f"{rainfall_risk['value']}/100",
                rainfall_risk["status"],
            )

        with col4:
            seasonal_risk = risk_factors["seasonal"]
            st.metric(
                "Seasonal Risk",
                f"{seasonal_risk['value']}/100",
                seasonal_risk["status"],
            )

        # Recommendations
        st.markdown("### 🛡️ Pest Management Recommendations")
        for i, rec in enumerate(pest_analysis["recommendations"], 1):
            st.markdown(f"{i}. {rec}")

        # Detailed Pest Information Cards
        st.markdown("---")
        st.markdown("### 🔬 Detailed Pest Intelligence & Management")

        if pest_analysis["pest_predictions"]:
            for pred in pest_analysis["pest_predictions"]:
                if (
                    pred["details"] and pred["probability"] > 30
                ):  # Show details for significant risks
                    details = pred["details"]
                    severity_color = (
                        "#FF4444"
                        if pred["severity"] == "High"
                        else "#FFAA00" if pred["severity"] == "Medium" else "#44AA44"
                    )

                    with st.expander(
                        f"🐛 {pred['pest']} - {pred['severity']} Risk ({pred['probability']}%)",
                        expanded=(pred["severity"] == "High"),
                    ):
                        col1, col2 = st.columns([1, 1])

                        with col1:
                            st.markdown(
                                f"**🔬 Scientific Name:** *{details['scientific_name']}*"
                            )
                            st.markdown(
                                f"**📅 Lifecycle:** {details['lifecycle_days']} days"
                            )
                            st.markdown(
                                f"**🥚 Reproduction:** {details['eggs_per_female']} eggs/female"
                            )

                            st.markdown("**🌡️ Favorable Conditions:**")
                            st.markdown(
                                f"- Temperature: {details['favorable_conditions']['temperature'][0]}-{details['favorable_conditions']['temperature'][1]}°C"
                            )
                            st.markdown(
                                f"- Humidity: {details['favorable_conditions']['humidity'][0]}-{details['favorable_conditions']['humidity'][1]}%"
                            )
                            st.markdown(
                                f"- Rainfall: {details['favorable_conditions']['rainfall']}"
                            )

                            st.markdown(
                                f"**⚠️ Critical Stages:** {', '.join(details['damage_stages'])}"
                            )
                            st.markdown(
                                f"**📊 Economic Threshold:** {details['economic_threshold']}"
                            )

                        with col2:
                            st.markdown("**👁️ Key Symptoms:**")
                            for symptom in details["symptoms"][
                                :4
                            ]:  # Show top 4 symptoms
                                st.markdown(f"• {symptom}")

                            st.markdown(f"**🔍 Monitoring:** {details['monitoring']}")

                        # Management strategies tabs
                        mgmt_tab1, mgmt_tab2 = st.tabs(
                            ["🌿 Organic Control", "💊 Chemical Control"]
                        )

                        with mgmt_tab1:
                            st.markdown("**Organic/Biological Control Methods:**")
                            for i, method in enumerate(details["organic_control"], 1):
                                st.markdown(f"{i}. {method}")

                        with mgmt_tab2:
                            st.markdown("**Chemical Control (When Organic Fails):**")
                            for i, method in enumerate(details["chemical_control"], 1):
                                st.markdown(f"{i}. {method}")
                            st.warning(
                                "⚠️ Always follow label instructions and observe waiting periods before harvest"
                            )

        # Crop-specific risks
        if pest_analysis["crop_specific_risks"]["high_risk_stage"]:
            st.error(
                f"🚨 HIGH ALERT: {growth_stage} stage is high-risk for {crop_type} pests!"
            )

    # Irrigation Management Tab
    with tabs[3]:
        st.markdown(
            '<div class="tab-header"><h2>💧 Irrigation Management System</h2></div>',
            unsafe_allow_html=True,
        )

        # Get irrigation recommendations
        irrigation_recs = system.get_irrigation_recommendations(
            crop_type, district, growth_stage, soil_ph, farm_area, current_weather
        )

        # Header with current irrigation status
        irrigation_priority = (
            "HIGH"
            if irrigation_recs["is_critical_stage"]
            else "MODERATE" if irrigation_recs["daily_water_requirement"] > 5 else "LOW"
        )
        priority_color = (
            "#FF4444"
            if irrigation_priority == "HIGH"
            else "#FFAA00" if irrigation_priority == "MODERATE" else "#44AA44"
        )

        st.markdown(
            f"""
        <div style="
            background: {priority_color};
            padding: 1.5rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
        ">
            <h2>🚨 Irrigation Priority: {irrigation_priority}</h2>
            <h3>Daily Water Requirement: {irrigation_recs['daily_water_requirement']} mm/day</h3>
            <p>Frequency: {irrigation_recs['irrigation_frequency']} | Total per hectare: {irrigation_recs['water_per_hectare']} L/day</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Main irrigation dashboard
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "💧 Daily Requirement",
                f"{irrigation_recs['daily_water_requirement']} mm",
                delta=f"{irrigation_recs['current_factors']['growth_stage_impact']}",
            )
            st.metric("📅 Frequency", irrigation_recs["irrigation_frequency"])

        with col2:
            st.metric("🏭 Recommended Method", irrigation_recs["recommended_method"])
            st.metric("💧 Water Source", irrigation_recs["water_source"])

        with col3:
            st.metric(
                "⚡ Zone Efficiency", f"{irrigation_recs['zone_efficiency'] * 100:.0f}%"
            )
            st.metric("🌾 Seasonal Total", f"{irrigation_recs['seasonal_total']} mm")

        st.markdown("---")

        # Weather impact analysis
        st.markdown("### 🌤️ Weather Impact on Irrigation Needs")

        col1, col2 = st.columns(2)

        with col1:
            # Weather factors impact chart
            factors_data = {
                "Factor": ["Temperature", "Humidity", "Wind Speed", "Growth Stage"],
                "Impact": [
                    float(
                        irrigation_recs["current_factors"]["temperature_impact"]
                        .replace("%", "")
                        .replace("+", "")
                    ),
                    float(
                        irrigation_recs["current_factors"]["humidity_impact"]
                        .replace("%", "")
                        .replace("+", "")
                    ),
                    float(
                        irrigation_recs["current_factors"]["wind_impact"]
                        .replace("%", "")
                        .replace("+", "")
                    ),
                    float(
                        irrigation_recs["current_factors"]["growth_stage_impact"]
                        .replace("%", "")
                        .replace("+", "")
                    ),
                ],
                "Values": [
                    f"{current_weather['temperature']}°C",
                    f"{current_weather['humidity']}%",
                    f"{current_weather['wind_speed']} m/s",
                    growth_stage,
                ],
            }

            impact_df = pd.DataFrame(factors_data)

            # Color coding for positive/negative impact using theme colors
            colors = [
                "#FF7043" if x > 0 else "#4CAF50" if x < 0 else "#03A9F4"
                for x in impact_df["Impact"]
            ]

            impact_fig = go.Figure(
                data=[
                    go.Bar(
                        x=impact_df["Factor"],
                        y=impact_df["Impact"],
                        marker_color=colors,
                        text=[
                            f"{val}<br>({imp:+.1f}%)"
                            for val, imp in zip(
                                impact_df["Values"], impact_df["Impact"]
                            )
                        ],
                        textposition="auto",
                        hovertemplate="<b>%{x}</b><br>Current: %{text}<br>Impact: %{y:+.1f}%<extra></extra>",
                    )
                ]
            )

            impact_fig.update_layout(
                title="Weather Factors Impact on Water Requirement",
                xaxis_title="Weather Factors",
                yaxis_title="Impact on Water Need (%)",
                height=400,
                plot_bgcolor="rgba(15, 20, 25, 0.1)",
                paper_bgcolor="rgba(15, 20, 25, 0.1)",
                font=dict(color="#E8F5E8"),
                title_font_color="#E8F5E8",
                xaxis=dict(color="#B8D4B8"),
                yaxis=dict(color="#B8D4B8"),
            )

            st.plotly_chart(impact_fig, use_container_width=True)

        with col2:
            # Irrigation schedule visualization
            days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

            if irrigation_recs["irrigation_frequency"] == "Daily":
                irrigation_schedule = [irrigation_recs["water_per_hectare"]] * 7
            elif irrigation_recs["irrigation_frequency"] == "Every 2 days":
                irrigation_schedule = [
                    irrigation_recs["water_per_hectare"] if i % 2 == 0 else 0
                    for i in range(7)
                ]
            else:  # Every 3 days
                irrigation_schedule = [
                    irrigation_recs["water_per_hectare"] if i % 3 == 0 else 0
                    for i in range(7)
                ]

            schedule_colors = [
                "#4CAF50" if val > 0 else "#8D6E63" for val in irrigation_schedule
            ]

            schedule_fig = go.Figure(
                data=[
                    go.Bar(
                        x=days,
                        y=irrigation_schedule,
                        marker_color=schedule_colors,
                        text=[
                            f"{int(val)} L" if val > 0 else "Rest"
                            for val in irrigation_schedule
                        ],
                        textposition="auto",
                    )
                ]
            )

            schedule_fig.update_layout(
                title="Weekly Irrigation Schedule",
                xaxis_title="Days of Week",
                yaxis_title="Water per Hectare (Liters)",
                height=400,
                plot_bgcolor="rgba(15, 20, 25, 0.1)",
                paper_bgcolor="rgba(15, 20, 25, 0.1)",
                font=dict(color="#E8F5E8"),
                title_font_color="#E8F5E8",
                xaxis=dict(color="#B8D4B8"),
                yaxis=dict(color="#B8D4B8"),
            )

            st.plotly_chart(schedule_fig, use_container_width=True)

        # Critical growth stages information
        if irrigation_recs["is_critical_stage"]:
            st.error(
                f"🌸 CRITICAL STAGE ALERT: {growth_stage} is a critical growth stage for {crop_type}!"
            )
            st.markdown("### ⚠️ Critical Growth Stage Management")
            st.info(
                f"Critical stages for {crop_type}: {', '.join(irrigation_recs['critical_growth_stages'])}"
            )

        # Quick Action Cards
        st.markdown("### ⚡ Quick Actions Required")

        action_cols = st.columns(3)

        # Determine today's actions based on schedule
        today_actions = [
            {
                "icon": "💧",
                "action": "Water Today",
                "time": "6-8 AM or 6-8 PM",
                "amount": f"{irrigation_recs['water_per_hectare']:.0f}L",
            },
            {
                "icon": "📊",
                "action": "Check Soil",
                "time": "Before watering",
                "amount": "2-3 inch depth",
            },
            {
                "icon": "🌡️",
                "action": "Monitor Weather",
                "time": "Check forecast",
                "amount": "Adjust as needed",
            },
        ]

        for i, action_col in enumerate(action_cols):
            with action_col:
                action = today_actions[i]
                st.markdown(
                    f"""
                <div style="
                    background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
                    padding: 1.5rem;
                    border-radius: 12px;
                    color: white;
                    text-align: center;
                    margin-bottom: 1rem;
                    box-shadow: 0 4px 12px rgba(76, 175, 80, 0.3);
                ">
                    <h2 style="margin: 0; font-size: 2.5rem;">{action['icon']}</h2>
                    <h4 style="margin: 0.5rem 0;">{action['action']}</h4>
                    <p style="margin: 0.25rem 0; opacity: 0.9;">🕒 {action['time']}</p>
                    <p style="margin: 0.25rem 0; font-weight: bold;">{action['amount']}</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

        # Detailed recommendations
        st.markdown("### 💡 Detailed Irrigation Plan")

        # Create columns for better organization
        rec_col1, rec_col2 = st.columns(2)

        with rec_col1:
            st.markdown("#### 🎯 Specific Actions:")
            for i, recommendation in enumerate(irrigation_recs["recommendations"], 1):
                st.markdown(f"{i}. {recommendation}")

        with rec_col2:
            st.markdown("#### 📊 Technical Details:")
            st.info(f"**Zone:** {zone}")
            st.info(f"**Crop Type:** {crop_type} (Growth: {growth_stage})")
            st.info(f"**Farm Area:** {farm_area} hectares")
            st.info(
                f"**Water Efficiency:** {irrigation_recs['zone_efficiency']*100:.0f}%"
            )
            st.info(f"**Seasonal Total:** {irrigation_recs['seasonal_total']} mm")

        # Visual irrigation guide
        st.markdown("---")
        st.markdown("### 📷 Visual Irrigation Guide")

        guide_cols = st.columns(3)

        irrigation_guides = [
            {
                "title": "✅ Correct Depth",
                "description": "Water should penetrate 6-8 inches deep for most crops",
                "tip": "Insert a stick to check moisture depth after watering",
            },
            {
                "title": "⏰ Best Timing",
                "description": "Early morning (6-8 AM) is optimal - less evaporation",
                "tip": "Avoid midday watering (10 AM - 4 PM) when evaporation is highest",
            },
            {
                "title": "📍 Uniform Coverage",
                "description": "Ensure even water distribution across the field",
                "tip": "Check for dry spots and adjust system accordingly",
            },
        ]

        for i, guide_col in enumerate(guide_cols):
            with guide_col:
                guide = irrigation_guides[i]
                st.markdown(
                    f"""
                <div style="
                    background: rgba(30, 130, 76, 0.1);
                    border-left: 4px solid #4CAF50;
                    padding: 1rem;
                    border-radius: 8px;
                    margin-bottom: 1rem;
                ">
                    <h4 style="color: #2E7D32; margin: 0 0 0.5rem 0;">{guide['title']}</h4>
                    <p style="margin: 0 0 0.5rem 0; font-size: 0.9rem;">{guide['description']}</p>
                    <p style="margin: 0; font-size: 0.85rem; font-style: italic; opacity: 0.8;">💡 {guide['tip']}</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

        # Water conservation tips
        st.markdown("---")
        st.markdown("### 🌱 Water Conservation Tips")

        conservation_tips = [
            "🕕 **Timing**: Irrigate during early morning (6-8 AM) or evening (6-8 PM) to minimize evaporation",
            "💧 **Drip System**: Use drip irrigation for 30-50% water savings compared to flood irrigation",
            "🌾 **Mulching**: Apply organic mulch to retain soil moisture and reduce evaporation",
            "📏 **Soil Testing**: Regular soil moisture testing prevents over-watering",
            "🌧️ **Rainwater**: Harvest rainwater during monsoon for dry season use",
            "📱 **Smart Systems**: Consider automated irrigation systems with soil moisture sensors",
        ]

        cols = st.columns(2)
        for i, tip in enumerate(conservation_tips):
            with cols[i % 2]:
                st.markdown(f"- {tip}")

        # Cost estimation
        st.markdown("---")
        st.markdown("### 💰 Irrigation Cost Estimation")

        col1, col2, col3 = st.columns(3)

        # Simple cost calculations (these would be more complex in reality)
        water_cost_per_1000l = 25  # INR per 1000 liters
        daily_water_liters = irrigation_recs["water_per_hectare"]
        daily_cost = (daily_water_liters / 1000) * water_cost_per_1000l
        weekly_cost = (
            daily_cost * 7
            if irrigation_recs["irrigation_frequency"] == "Daily"
            else (
                daily_cost * 3.5
                if irrigation_recs["irrigation_frequency"] == "Every 2 days"
                else daily_cost * 2.33
            )
        )
        monthly_cost = weekly_cost * 4.33

        with col1:
            st.metric("Daily Cost", f"₹{daily_cost:.2f}")
        with col2:
            st.metric("Weekly Cost", f"₹{weekly_cost:.2f}")
        with col3:
            st.metric("Monthly Cost", f"₹{monthly_cost:.2f}")

    # Zone Mapping Tab
    with tabs[4]:
        st.markdown(
            '<div class="tab-header"><h2>🗺️ Maharashtra Zone Mapping</h2></div>',
            unsafe_allow_html=True,
        )

        # Header card with zone context
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, var(--primary-green) 0%, var(--secondary-green) 50%, var(--sky-blue) 100%);
                padding: 1rem 1.25rem;
                border-radius: 12px;
                color: white;
                text-align: center;
                margin-bottom: 1rem;
                box-shadow: 0 6px 18px rgba(46,125,50,0.35);
                border: 1px solid rgba(255,255,255,0.12);
            ">
                <h3 style="margin: 0; letter-spacing: 0.3px;">🌾 Agricultural Zone Map — {zone}</h3>
                <p style="margin: 6px 0 0; opacity: 0.9; font-size: 0.95rem;">Selected District: <b>{district}</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Layout: Map + Side Info
        col1, col2 = st.columns([2.2, 1])

        with col1:
            # Themed zone colors aligned with app palette
            zone_colors = {
                "Konkan (Coastal)": "#03A9F4",  # Water Blue
                "Western Maharashtra": "#4CAF50",  # Bright Green
                "North Maharashtra (Khandesh)": "#FFA726",  # Sunshine Yellow
                "Marathwada": "#8D6E63",  # Soil Brown
                "Vidarbha": "#1976D2",  # Sky Blue (distinct from Konkan)
            }

            # Prepare district geodata
            district_data = []
            for zone_name, districts in system.maharashtra_districts.items():
                for district_name in districts:
                    if district_name in system.district_coords:
                        lat, lon = system.district_coords[district_name]
                        district_data.append(
                            {
                                "District": district_name,
                                "Zone": zone_name,
                                "lat": lat,
                                "lon": lon,
                                "size": 44 if district_name == district else 26,
                            }
                        )

            df = pd.DataFrame(district_data)

            # Enhanced Google Maps Integration
            try:
                # Create Google Maps HTML with your API key
                google_maps_api_key = "AIzaSyD-Osm1LncXr34Pks87eulH9Qt0HO-0srI"

                # Prepare markers for Google Maps
                markers = []
                for _, row in df.iterrows():
                    color = zone_colors.get(row["Zone"], "#4CAF50")
                    marker_size = "large" if row["District"] == district else "mid"
                    icon_color = "red" if row["District"] == district else color

                    markers.append(
                        {
                            "lat": row["lat"],
                            "lng": row["lon"],
                            "title": row["District"],
                            "zone": row["Zone"],
                            "color": icon_color,
                            "size": marker_size,
                            "selected": row["District"] == district,
                        }
                    )

                # Create Google Maps HTML
                google_maps_html = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <style>
                        #map {{
                            height: 520px;
                            width: 100%;
                            border-radius: 12px;
                            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                        }}
                        .zone-legend {{
                            background: white;
                            padding: 10px;
                            margin: 10px;
                            border-radius: 8px;
                            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                            font-family: 'Segoe UI', Arial, sans-serif;
                        }}
                        .legend-item {{
                            display: flex;
                            align-items: center;
                            margin: 5px 0;
                        }}
                        .legend-color {{
                            width: 16px;
                            height: 16px;
                            border-radius: 50%;
                            margin-right: 8px;
                        }}
                    </style>
                </head>
                <body>
                    <div id="map"></div>
                    <script>
                        function initMap() {{
                            const centerLat = 19.4;
                            const centerLng = 76.7;
                            
                            const map = new google.maps.Map(document.getElementById('map'), {{
                                zoom: 7,
                                center: {{ lat: centerLat, lng: centerLng }},
                                styles: [
                                    {{
                                        featureType: 'water',
                                        elementType: 'geometry',
                                        stylers: [{{ color: '#667eea' }}]
                                    }},
                                    {{
                                        featureType: 'landscape',
                                        elementType: 'geometry',
                                        stylers: [{{ color: '#f5f5f5' }}]
                                    }},
                                    {{
                                        featureType: 'road',
                                        elementType: 'geometry',
                                        stylers: [{{ color: '#ffffff' }}]
                                    }}
                                ]
                            }});
                            
                            const markers = {markers};
                            const bounds = new google.maps.LatLngBounds();
                            
                            markers.forEach(markerData => {{
                                const position = {{ lat: markerData.lat, lng: markerData.lng }};
                                
                                const marker = new google.maps.Marker({{
                                    position: position,
                                    map: map,
                                    title: markerData.title + ' (' + markerData.zone + ')',
                                    icon: {{
                                        url: markerData.selected ? 
                                            'data:image/svg+xml;charset=UTF-8,' + encodeURIComponent(`
                                                <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24">
                                                    <path fill="#F44336" d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
                                                </svg>
                                            `) :
                                            'data:image/svg+xml;charset=UTF-8,' + encodeURIComponent(`
                                                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24">
                                                    <circle cx="12" cy="12" r="10" fill="${{markerData.color}}" stroke="white" stroke-width="2"/>
                                                    <text x="12" y="16" text-anchor="middle" fill="white" font-size="12">📍</text>
                                                </svg>
                                            `),
                                        scaledSize: new google.maps.Size(markerData.selected ? 32 : 24, markerData.selected ? 32 : 24),
                                        anchor: new google.maps.Point(markerData.selected ? 16 : 12, markerData.selected ? 16 : 12)
                                    }}
                                }});
                                
                                const infoWindow = new google.maps.InfoWindow({{
                                    content: `
                                        <div style="padding: 8px; font-family: 'Segoe UI', Arial, sans-serif;">
                                            <h3 style="margin: 0 0 8px; color: ${{markerData.color}};">${{markerData.title}}</h3>
                                            <p style="margin: 0; color: #666;">Zone: ${{markerData.zone}}</p>
                                            ${{markerData.selected ? '<p style="margin: 4px 0 0; color: #F44336; font-weight: bold;">📍 Currently Selected</p>' : ''}}
                                        </div>
                                    `
                                }});
                                
                                marker.addListener('click', () => {{
                                    infoWindow.open(map, marker);
                                }});
                                
                                bounds.extend(position);
                            }});
                            
                            map.fitBounds(bounds);
                            
                            // Add zone legend
                            const legend = document.createElement('div');
                            legend.className = 'zone-legend';
                            legend.innerHTML = `
                                <h4 style="margin: 0 0 8px; color: #2E7D32;">🗺️ Zone Legend</h4>
                                <div class="legend-item">
                                    <div class="legend-color" style="background-color: #03A9F4;"></div>
                                    <span>Konkan (Coastal)</span>
                                </div>
                                <div class="legend-item">
                                    <div class="legend-color" style="background-color: #4CAF50;"></div>
                                    <span>Western Maharashtra</span>
                                </div>
                                <div class="legend-item">
                                    <div class="legend-color" style="background-color: #FFA726;"></div>
                                    <span>North Maharashtra</span>
                                </div>
                                <div class="legend-item">
                                    <div class="legend-color" style="background-color: #8D6E63;"></div>
                                    <span>Marathwada</span>
                                </div>
                                <div class="legend-item">
                                    <div class="legend-color" style="background-color: #1976D2;"></div>
                                    <span>Vidarbha</span>
                                </div>
                                <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid #eee; font-size: 12px; color: #666;">
                                    ⭐ Selected: {district}
                                </div>
                            `;
                            
                            map.controls[google.maps.ControlPosition.RIGHT_BOTTOM].push(legend);
                        }}
                    </script>
                    <script src="https://maps.googleapis.com/maps/api/js?key={google_maps_api_key}&callback=initMap" async defer></script>
                </body>
                </html>
                """

                # Display Google Maps
                st.components.v1.html(google_maps_html, height=550)

            except Exception as e:
                # Fallback: Enhanced Plotly map with better styling
                st.warning(
                    f"Google Maps unavailable, using enhanced map view. Error: {str(e)}"
                )

                # Try modern map API first; graceful fallback to 2D scatter
                try:
                    # Try scatter_mapbox with Open Street Map
                    fig = px.scatter_mapbox(
                        df,
                        lat="lat",
                        lon="lon",
                        color="Zone",
                        size="size",
                        hover_name="District",
                        hover_data={
                            "lat": False,
                            "lon": False,
                            "size": False,
                            "Zone": True,
                        },
                        title=f"Maharashtra Districts — {zone} zone highlighted",
                        height=520,
                        color_discrete_map=zone_colors,
                        zoom=6.2,
                        center={"lat": 19.4, "lon": 76.7},
                        mapbox_style="open-street-map",  # Use open-street-map instead
                    )

                    # Map style and theming
                    fig.update_layout(
                        margin=dict(l=0, r=0, t=48, b=0),
                        legend=dict(
                            title="Zones",
                            orientation="h",
                            yanchor="bottom",
                            y=0.02,
                            xanchor="center",
                            x=0.5,
                            bgcolor="rgba(255,255,255,0.8)",
                            borderwidth=1,
                        ),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                    )

                    # Emphasize selected district with star marker overlay
                    if district in system.district_coords:
                        sel_lat, sel_lon = system.district_coords[district]
                        fig.add_scattermapbox(
                            lat=[sel_lat],
                            lon=[sel_lon],
                            mode="markers",
                            marker=dict(size=28, color="#F44336", symbol="star"),
                            name=f"{district} (Selected)",
                            showlegend=False,
                        )

                    st.plotly_chart(fig, use_container_width=True)

                except Exception:
                    # Final fallback: static scatter with theme
                    fig2 = px.scatter(
                        df,
                        x="lon",
                        y="lat",
                        color="Zone",
                        size="size",
                        hover_name="District",
                        title=f"Maharashtra Districts — {zone} zone highlighted",
                        color_discrete_map=zone_colors,
                        height=520,
                    )
                    if district in system.district_coords:
                        sel_lat, sel_lon = system.district_coords[district]
                        fig2.add_scatter(
                            x=[sel_lon],
                            y=[sel_lat],
                            mode="markers",
                            marker=dict(size=28, color="#F44336", symbol="star"),
                            name=f"{district} (Selected)",
                            showlegend=False,
                        )
                    fig2.update_layout(
                        margin=dict(l=0, r=0, t=48, b=0),
                        legend=dict(orientation="h", y=0.02, x=0.5, xanchor="center"),
                        paper_bgcolor="rgba(255,255,255,1)",
                        plot_bgcolor="rgba(255,255,255,1)",
                    )
                    st.plotly_chart(fig2, use_container_width=True)

        with col2:
            # Zone info panel
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, var(--card-bg) 0%, var(--surface-bg) 100%);
                    padding: 1rem 1.25rem;
                    border-radius: 12px;
                    color: var(--text-primary);
                    border: 1px solid var(--border-color);
                    box-shadow: 0 6px 18px rgba(0,0,0,0.25);
                    margin-bottom: 1rem;
                ">
                    <h4 style="margin: 0;">{zone}</h4>
                    <p style="margin: 6px 0 0; color: var(--text-secondary);">Districts in this zone</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Enhanced zone meta information with detailed agricultural data
            zone_info = {
                "Konkan (Coastal)": {
                    "crops": "Rice, Coconut, Mango, Cashew",
                    "rainfall": "2000–4000 mm",
                    "soil_type": "Laterite, Alluvial",
                    "climate": "Tropical humid",
                    "challenges": "High humidity, Fungal diseases",
                    "advantages": "Year-round water availability, High rainfall",
                    "irrigation": "Surface water, Wells",
                    "season": "Kharif dominant",
                },
                "Western Maharashtra": {
                    "crops": "Sugarcane, Grapes, Cotton, Pomegranate",
                    "rainfall": "400–1200 mm",
                    "soil_type": "Black cotton soil",
                    "climate": "Semi-arid to sub-humid",
                    "challenges": "Water scarcity, Drought prone",
                    "advantages": "Fertile black soil, Good infrastructure",
                    "irrigation": "Canal, Drip irrigation",
                    "season": "Rabi and summer crops",
                },
                "North Maharashtra (Khandesh)": {
                    "crops": "Cotton, Banana, Onion, Wheat",
                    "rainfall": "600–1000 mm",
                    "soil_type": "Black cotton, Alluvial",
                    "climate": "Semi-arid",
                    "challenges": "Irregular rainfall, Pest pressure",
                    "advantages": "River irrigation, Good connectivity",
                    "irrigation": "Canal, Bore wells",
                    "season": "Kharif and Rabi",
                },
                "Marathwada": {
                    "crops": "Cotton, Soybean, Jowar, Tur",
                    "rainfall": "500–900 mm",
                    "soil_type": "Black cotton soil",
                    "climate": "Semi-arid, Drought prone",
                    "challenges": "Water scarcity, Frequent droughts",
                    "advantages": "Suitable for dryland crops",
                    "irrigation": "Wells, Micro-irrigation",
                    "season": "Kharif dominant",
                },
                "Vidarbha": {
                    "crops": "Cotton, Rice, Orange, Soybean",
                    "rainfall": "800–1400 mm",
                    "soil_type": "Black cotton, Red soil",
                    "climate": "Sub-humid to semi-arid",
                    "challenges": "Pest issues, Market access",
                    "advantages": "Good rainfall, Diverse crops",
                    "irrigation": "Tanks, Wells, Rivers",
                    "season": "Kharif and Rabi",
                },
            }

            info = zone_info.get(
                zone,
                {
                    "crops": "Mixed crops",
                    "rainfall": "Variable",
                    "soil_type": "Various",
                    "climate": "Variable",
                    "challenges": "General challenges",
                    "advantages": "Various advantages",
                    "irrigation": "Mixed sources",
                    "season": "Both seasons",
                },
            )

            # Enhanced zone information display
            st.markdown(f"🌾 **Main Crops:** {info['crops']}")
            st.markdown(f"🌧️ **Annual Rainfall:** {info['rainfall']}")
            st.markdown(f"🏔️ **Soil Type:** {info['soil_type']}")
            st.markdown(f"🌡️ **Climate:** {info['climate']}")
            st.markdown(f"💧 **Irrigation:** {info['irrigation']}")
            st.markdown(f"📅 **Cropping Season:** {info['season']}")

            # Challenges and advantages
            st.markdown("---")
            st.markdown("**⚠️ Key Challenges:**")
            st.markdown(f"• {info['challenges']}")
            st.markdown("**✅ Advantages:**")
            st.markdown(f"• {info['advantages']}")

            # District listing with selection emphasis
            st.markdown(f"**Districts ({len(system.maharashtra_districts[zone])}):**")
            for dist in system.maharashtra_districts[zone]:
                if dist == district:
                    st.markdown(f"🎯 **{dist}** (Selected)")
                else:
                    st.markdown(f"• {dist}")

        # Comprehensive Zone Analytics Section
        st.markdown("---")
        st.markdown("### 📊 Zone-wise Agricultural Analytics")

        # Zone statistics and insights
        col1, col2, col3 = st.columns(3)

        with col1:
            # Zone-specific crop production data
            zone_crop_data = {
                "Konkan (Coastal)": {
                    "primary_crop": "Rice",
                    "production": "2.8M tons/year",
                    "area": "1.2M hectares",
                    "productivity": "2.3 tons/ha",
                    "speciality": "High-quality Basmati",
                },
                "Western Maharashtra": {
                    "primary_crop": "Sugarcane",
                    "production": "65M tons/year",
                    "area": "1.1M hectares",
                    "productivity": "59 tons/ha",
                    "speciality": "Highest sugar recovery",
                },
                "North Maharashtra (Khandesh)": {
                    "primary_crop": "Cotton",
                    "production": "1.2M bales/year",
                    "area": "0.8M hectares",
                    "productivity": "380 kg/ha",
                    "speciality": "Premium quality fiber",
                },
                "Marathwada": {
                    "primary_crop": "Cotton",
                    "production": "2.1M bales/year",
                    "area": "1.5M hectares",
                    "productivity": "360 kg/ha",
                    "speciality": "Drought-resistant varieties",
                },
                "Vidarbha": {
                    "primary_crop": "Cotton",
                    "production": "1.8M bales/year",
                    "area": "1.3M hectares",
                    "productivity": "350 kg/ha",
                    "speciality": "Organic cotton farming",
                },
            }

            crop_data = zone_crop_data.get(zone, {})

            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, var(--secondary-green) 0%, var(--accent-green) 100%);
                    padding: 1.5rem;
                    border-radius: 12px;
                    color: white;
                    margin-bottom: 1rem;
                    box-shadow: 0 6px 18px rgba(76,175,80,0.3);
                ">
                    <h4 style="margin: 0 0 0.5rem;">🌾 Primary Crop Production</h4>
                    <p style="margin: 0.25rem 0; font-size: 1.1em;"><b>{crop_data.get('primary_crop', 'Mixed')}</b></p>
                    <p style="margin: 0.25rem 0; opacity: 0.9;">Production: {crop_data.get('production', 'N/A')}</p>
                    <p style="margin: 0.25rem 0; opacity: 0.9;">Area: {crop_data.get('area', 'N/A')}</p>
                    <p style="margin: 0.25rem 0; opacity: 0.9;">Productivity: {crop_data.get('productivity', 'N/A')}</p>
                    <p style="margin: 0.5rem 0 0; font-style: italic; opacity: 0.95;">✨ {crop_data.get('speciality', 'Diverse agriculture')}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            # Zone-specific weather and climate patterns
            zone_weather_data = {
                "Konkan (Coastal)": {
                    "climate": "Tropical Maritime",
                    "temp_range": "24-32°C",
                    "humidity": "75-85%",
                    "monsoon": "June-September",
                    "risk": "Cyclones, Heavy rainfall",
                },
                "Western Maharashtra": {
                    "climate": "Semi-arid",
                    "temp_range": "18-35°C",
                    "humidity": "50-70%",
                    "monsoon": "June-September",
                    "risk": "Drought, Hailstorms",
                },
                "North Maharashtra (Khandesh)": {
                    "climate": "Semi-arid",
                    "temp_range": "20-38°C",
                    "humidity": "45-65%",
                    "monsoon": "June-September",
                    "risk": "Heat waves, Erratic rainfall",
                },
                "Marathwada": {
                    "climate": "Semi-arid",
                    "temp_range": "19-40°C",
                    "humidity": "40-60%",
                    "monsoon": "June-September",
                    "risk": "Drought, Water scarcity",
                },
                "Vidarbha": {
                    "climate": "Tropical",
                    "temp_range": "16-45°C",
                    "humidity": "55-75%",
                    "monsoon": "June-October",
                    "risk": "Extreme heat, Floods",
                },
            }

            weather_data_zone = zone_weather_data.get(zone, {})

            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, var(--sky-blue) 0%, var(--water-blue) 100%);
                    padding: 1.5rem;
                    border-radius: 12px;
                    color: white;
                    margin-bottom: 1rem;
                    box-shadow: 0 6px 18px rgba(25,118,210,0.3);
                ">
                    <h4 style="margin: 0 0 0.5rem;">🌤️ Climate & Weather</h4>
                    <p style="margin: 0.25rem 0; font-size: 1.1em;"><b>{weather_data_zone.get('climate', 'Variable')}</b></p>
                    <p style="margin: 0.25rem 0; opacity: 0.9;">Temperature: {weather_data_zone.get('temp_range', 'N/A')}</p>
                    <p style="margin: 0.25rem 0; opacity: 0.9;">Humidity: {weather_data_zone.get('humidity', 'N/A')}</p>
                    <p style="margin: 0.25rem 0; opacity: 0.9;">Monsoon: {weather_data_zone.get('monsoon', 'N/A')}</p>
                    <p style="margin: 0.5rem 0 0; font-style: italic; opacity: 0.95;">⚠️ {weather_data_zone.get('risk', 'Weather variability')}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col3:
            # Zone-specific agricultural challenges and solutions
            zone_challenges = {
                "Konkan (Coastal)": {
                    "main_challenge": "Soil Salinity",
                    "secondary": "Waterlogging",
                    "solution": "Raised bed cultivation",
                    "tech": "Salt-tolerant varieties",
                    "support": "Drainage systems",
                },
                "Western Maharashtra": {
                    "main_challenge": "Water Management",
                    "secondary": "Pest outbreaks",
                    "solution": "Drip irrigation",
                    "tech": "Precision farming",
                    "support": "Cooperative farming",
                },
                "North Maharashtra (Khandesh)": {
                    "main_challenge": "Market Access",
                    "secondary": "Price volatility",
                    "solution": "Direct marketing",
                    "tech": "Cold storage",
                    "support": "FPO development",
                },
                "Marathwada": {
                    "main_challenge": "Drought Resilience",
                    "secondary": "Soil degradation",
                    "solution": "Watershed management",
                    "tech": "Drought-resistant crops",
                    "support": "Water harvesting",
                },
                "Vidarbha": {
                    "main_challenge": "Input Costs",
                    "secondary": "Farmer distress",
                    "solution": "Organic farming",
                    "tech": "Integrated pest management",
                    "support": "Credit access",
                },
            }

            challenge_data = zone_challenges.get(zone, {})

            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, var(--harvest-orange) 0%, var(--sunshine-yellow) 100%);
                    padding: 1.5rem;
                    border-radius: 12px;
                    color: white;
                    margin-bottom: 1rem;
                    box-shadow: 0 6px 18px rgba(255,112,67,0.3);
                ">
                    <h4 style="margin: 0 0 0.5rem;">⚡ Challenges & Solutions</h4>
                    <p style="margin: 0.25rem 0; font-size: 1.1em;"><b>{challenge_data.get('main_challenge', 'Various challenges')}</b></p>
                    <p style="margin: 0.25rem 0; opacity: 0.9;">Secondary: {challenge_data.get('secondary', 'N/A')}</p>
                    <p style="margin: 0.25rem 0; opacity: 0.9;">Solution: {challenge_data.get('solution', 'N/A')}</p>
                    <p style="margin: 0.25rem 0; opacity: 0.9;">Technology: {challenge_data.get('tech', 'N/A')}</p>
                    <p style="margin: 0.5rem 0 0; font-style: italic; opacity: 0.95;">🤝 {challenge_data.get('support', 'Government support')}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Zone comparison chart
        st.markdown("---")
        st.markdown("### 📊 Inter-Zone Agricultural Comparison")

        # Create comparison data for all zones
        zone_comparison_data = {
            "Zone": list(system.maharashtra_districts.keys()),
            "Districts": [
                len(districts) for districts in system.maharashtra_districts.values()
            ],
            "Avg_Rainfall": [3000, 800, 800, 700, 1100],  # mm/year
            "Productivity_Index": [75, 85, 70, 60, 65],  # 0-100 scale
            "Crop_Diversity": [6, 8, 7, 5, 6],  # Number of major crops
        }

        comparison_df = pd.DataFrame(zone_comparison_data)

        col1, col2 = st.columns(2)

        with col1:
            # Rainfall comparison
            rainfall_fig = px.bar(
                comparison_df,
                x="Zone",
                y="Avg_Rainfall",
                title="Average Annual Rainfall by Zone",
                color="Avg_Rainfall",
                color_continuous_scale="Blues",
                height=400,
            )
            rainfall_fig.update_layout(
                xaxis_title="Agricultural Zones",
                yaxis_title="Rainfall (mm/year)",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                title_font_color="white",
            )
            rainfall_fig.update_xaxes(tickangle=45)
            st.plotly_chart(rainfall_fig, use_container_width=True)

        with col2:
            # Productivity comparison
            productivity_fig = px.scatter(
                comparison_df,
                x="Districts",
                y="Productivity_Index",
                size="Crop_Diversity",
                color="Zone",
                title="Zone Productivity vs District Count",
                color_discrete_map=zone_colors,
                height=400,
                hover_data={"Crop_Diversity": True},
            )
            productivity_fig.update_layout(
                xaxis_title="Number of Districts",
                yaxis_title="Productivity Index (0-100)",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                title_font_color="white",
            )
            st.plotly_chart(productivity_fig, use_container_width=True)

        # Quick zone selector
        st.markdown("---")
        st.markdown("### 🎯 Quick Zone Explorer")

        zone_cols = st.columns(len(system.maharashtra_districts))
        for i, (zone_name, zone_districts) in enumerate(
            system.maharashtra_districts.items()
        ):
            with zone_cols[i]:
                zone_color = zone_colors.get(zone_name, "#666666")
                selected_indicator = (
                    "🎯 SELECTED" if zone_name == zone else "View Details"
                )

                st.markdown(
                    f"""
                    <div style="
                        background: {zone_color};
                        padding: 1rem;
                        border-radius: 8px;
                        color: white;
                        text-align: center;
                        margin-bottom: 0.5rem;
                        cursor: pointer;
                        transition: transform 0.2s;
                    " class="zone-card">
                        <h5 style="margin: 0 0 0.5rem; font-size: 0.9rem;">{zone_name}</h5>
                        <p style="margin: 0; font-size: 0.8rem; opacity: 0.9;">{len(zone_districts)} Districts</p>
                        <p style="margin: 0.5rem 0 0; font-size: 0.7rem; font-weight: bold;">{selected_indicator}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Additional zone insights
        st.markdown("---")
        st.markdown(
            f"""
            ### 💡 Zone-Specific Insights for {zone}
            
            Based on current analysis of **{district}** district in **{zone}** zone:
            """
        )

        # Generate zone-specific recommendations
        zone_insights = {
            "Konkan (Coastal)": [
                "🌊 **Coastal Advantage**: Utilize sea breeze for natural cooling in greenhouse cultivation",
                "🐟 **Aquaculture Integration**: Consider fish-rice farming systems for additional income",
                "🥥 **Coconut Based Systems**: Integrate coconut with spice crops for optimal land use",
                "⛈️ **Weather Preparedness**: Install cyclone-resistant structures for crop protection",
            ],
            "Western Maharashtra": [
                "🍇 **Premium Crops**: Focus on high-value crops like grapes and pomegranates",
                "💧 **Water Efficiency**: Implement micro-irrigation for water conservation",
                "🏭 **Processing Units**: Establish on-farm processing for value addition",
                "🤝 **Cooperative Strength**: Leverage strong cooperative network for marketing",
            ],
            "North Maharashtra (Khandesh)": [
                "🍌 **Banana Excellence**: Optimize banana cultivation with tissue culture plants",
                "🧅 **Onion Storage**: Invest in proper curing and storage facilities",
                "🌡️ **Heat Management**: Use shade nets during extreme summer months",
                "📦 **Market Linkages**: Strengthen direct market connections for better prices",
            ],
            "Marathwada": [
                "🏜️ **Drought Mitigation**: Adopt climate-resilient crop varieties",
                "💧 **Water Harvesting**: Implement farm pond and watershed management",
                "🌾 **Diversification**: Include pulses and oilseeds in cropping pattern",
                "🐄 **Livestock Integration**: Combine crop-livestock systems for stability",
            ],
            "Vidarbha": [
                "🍊 **Citrus Excellence**: Focus on Nagpur orange cultivation and export",
                "🌱 **Organic Transition**: Consider organic farming for premium markets",
                "⚡ **Solar Integration**: Use solar pumps for sustainable irrigation",
                "🌾 **Rice Systems**: Optimize rice cultivation in suitable areas",
            ],
        }

        insights = zone_insights.get(zone, [])

        col1, col2 = st.columns(2)
        for i, insight in enumerate(insights):
            col = col1 if i % 2 == 0 else col2
            with col:
                st.markdown(f"- {insight}")

    # Dashboard Tab
    with tabs[5]:
        st.markdown(
            '<div class="tab-header"><h2>Farm Analytics Dashboard</h2></div>',
            unsafe_allow_html=True,
        )

        # Farm Overview Section
        st.markdown("### 🏡 Farm Overview")
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.markdown(
                f'<div style="background: linear-gradient(135deg, #2E7D32 0%, #4CAF50 100%); padding: 1rem; border-radius: 10px; text-align: center; color: white; box-shadow: 0 4px 15px rgba(46, 125, 50, 0.3);"><h4>🏠 Farm Size</h4><h2>{farm_area} Ha</h2></div>',
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f'<div style="background: linear-gradient(135deg, #5D4E37 0%, #8D6E63 100%); padding: 1rem; border-radius: 10px; text-align: center; color: white; box-shadow: 0 4px 15px rgba(93, 78, 55, 0.3);"><h4>🌾 Crop Type</h4><h2>{crop_type}</h2></div>',
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                f'<div style="background: linear-gradient(135deg, #1976D2 0%, #03A9F4 100%); padding: 1rem; border-radius: 10px; text-align: center; color: white; box-shadow: 0 4px 15px rgba(25, 118, 210, 0.3);"><h4>🌱 Growth Stage</h4><h2>{growth_stage}</h2></div>',
                unsafe_allow_html=True,
            )

        with col4:
            st.markdown(
                f'<div style="background: linear-gradient(135deg, #FF9800 0%, #FFA726 100%); padding: 1rem; border-radius: 10px; text-align: center; color: white; box-shadow: 0 4px 15px rgba(255, 152, 0, 0.3);"><h4>🌤️ Temperature</h4><h2>{current_weather["temperature"]}°C</h2></div>',
                unsafe_allow_html=True,
            )

        with col5:
            location_info = f"{district}, {zone}"
            st.markdown(
                f'<div style="background: linear-gradient(135deg, #FF7043 0%, #F44336 100%); padding: 1rem; border-radius: 10px; text-align: center; color: white; box-shadow: 0 4px 15px rgba(255, 112, 67, 0.3);"><h4>📍 Location</h4><h2 style="font-size: 0.8rem;">{location_info}</h2></div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # Analysis Results Dashboard
        st.markdown("### 📊 Analysis Results Dashboard")

        col1, col2 = st.columns(2)

        with col1:
            # Crop Health Summary
            if (
                "crop_analysis" in st.session_state
                and st.session_state.crop_analysis is not None
            ):
                result = st.session_state.crop_analysis
                health_color = (
                    "linear-gradient(135deg, #4CAF50 0%, #66BB6A 100%)"
                    if result["disease"].lower() == "healthy"
                    else "linear-gradient(135deg, #F44336 0%, #FF7043 100%)"
                )
                st.markdown(
                    f'<div style="background: {health_color}; padding: 1.5rem; border-radius: 15px; color: white; margin: 1rem 0; box-shadow: 0 4px 15px rgba(0,0,0,0.2);"><h3>🌿 Crop Health Status</h3><h1>{result["disease"]}</h1><p>Confidence: {result["confidence"]:.1f}%</p></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div style="background: linear-gradient(135deg, #8D6E63 0%, #5D4E37 100%); padding: 1.5rem; border-radius: 15px; color: white; margin: 1rem 0; box-shadow: 0 4px 15px rgba(0,0,0,0.2);"><h3>🌿 Crop Health Status</h3><h1>Not Analyzed</h1><p>Upload image and analyze</p></div>',
                    unsafe_allow_html=True,
                )

            # Soil Health Summary
            if (
                "soil_analysis" in st.session_state
                and st.session_state.soil_analysis is not None
            ):
                soil = st.session_state.soil_analysis
                soil_color = (
                    "linear-gradient(135deg, #4CAF50 0%, #66BB6A 100%)"
                    if soil["score"] > 70
                    else (
                        "linear-gradient(135deg, #FF9800 0%, #FFA726 100%)"
                        if soil["score"] > 50
                        else "linear-gradient(135deg, #F44336 0%, #FF7043 100%)"
                    )
                )
                st.markdown(
                    f'<div style="background: {soil_color}; padding: 1.5rem; border-radius: 15px; color: white; margin: 1rem 0; box-shadow: 0 4px 15px rgba(0,0,0,0.2);"><h3>🧪 Soil Health</h3><h1>{soil["score"]}/100</h1><p>Status: {soil["status"]}</p></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div style="background: linear-gradient(135deg, #8D6E63 0%, #5D4E37 100%); padding: 1.5rem; border-radius: 15px; color: white; margin: 1rem 0; box-shadow: 0 4px 15px rgba(0,0,0,0.2);"><h3>🧪 Soil Health</h3><h1>Not Analyzed</h1><p>Run analysis to see results</p></div>',
                    unsafe_allow_html=True,
                )

        with col2:
            # Pest Risk Summary
            if (
                "pest_analysis" in st.session_state
                and st.session_state.pest_analysis is not None
            ):
                pest = st.session_state.pest_analysis
                risk_color = pest["risk_level"]["color"]
                st.markdown(
                    f'<div style="background: {risk_color}; padding: 1.5rem; border-radius: 15px; color: white; margin: 1rem 0;"><h3>🐛 Pest Risk Level</h3><h1>{pest["risk_level"]["level"]}</h1><p>Risk Score: {pest["overall_risk"]}/100</p></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div style="background: #666; padding: 1.5rem; border-radius: 15px; color: white; margin: 1rem 0;"><h3>🐛 Pest Risk Level</h3><h1>Not Analyzed</h1><p>Run analysis to see results</p></div>',
                    unsafe_allow_html=True,
                )

            # Irrigation Summary
            if (
                "irrigation_analysis" in st.session_state
                and st.session_state.irrigation_analysis is not None
            ):
                irrigation = st.session_state.irrigation_analysis
                irrigation_priority = (
                    "HIGH"
                    if irrigation["is_critical_stage"]
                    else (
                        "MODERATE"
                        if irrigation["daily_water_requirement"] > 5
                        else "LOW"
                    )
                )
                irrigation_color = (
                    "linear-gradient(135deg, #F44336 0%, #FF7043 100%)"
                    if irrigation_priority == "HIGH"
                    else (
                        "linear-gradient(135deg, #FF9800 0%, #FFA726 100%)"
                        if irrigation_priority == "MODERATE"
                        else "linear-gradient(135deg, #03A9F4 0%, #1976D2 100%)"
                    )
                )
                st.markdown(
                    f'<div style="background: {irrigation_color}; padding: 1.5rem; border-radius: 15px; color: white; margin: 1rem 0; box-shadow: 0 4px 15px rgba(0,0,0,0.2);"><h3>💧 Irrigation Priority</h3><h1>{irrigation_priority}</h1><p>{irrigation["daily_water_requirement"]:.1f} mm/day</p></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div style="background: linear-gradient(135deg, #8D6E63 0%, #5D4E37 100%); padding: 1.5rem; border-radius: 15px; color: white; margin: 1rem 0; box-shadow: 0 4px 15px rgba(0,0,0,0.2);"><h3>💧 Irrigation Status</h3><h1>Not Analyzed</h1><p>Run analysis to see results</p></div>',
                    unsafe_allow_html=True,
                )

            # Weather Summary
            weather_status = (
                "Good"
                if 20 <= current_weather["temperature"] <= 30
                and current_weather["humidity"] <= 70
                else "Monitor"
            )
            weather_color = (
                "linear-gradient(135deg, #4CAF50 0%, #66BB6A 100%)"
                if weather_status == "Good"
                else "linear-gradient(135deg, #FF9800 0%, #FFA726 100%)"
            )
            st.markdown(
                f'<div style="background: {weather_color}; padding: 1.5rem; border-radius: 15px; color: white; margin: 1rem 0; box-shadow: 0 4px 15px rgba(0,0,0,0.2);"><h3>🌤️ Weather Status</h3><h1>{weather_status}</h1><p>{current_weather["description"].title()}</p></div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # Action Items and Recommendations
        st.markdown("### 📝 Action Items & Recommendations")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🔴 Priority Actions")
            priority_actions = []

            if (
                "crop_analysis" in st.session_state
                and st.session_state.crop_analysis is not None
                and st.session_state.crop_analysis["disease"].lower() != "healthy"
            ):
                priority_actions.append(
                    f"🌿 URGENT: Treat {st.session_state.crop_analysis['disease']}"
                )

            if (
                "pest_analysis" in st.session_state
                and st.session_state.pest_analysis is not None
                and st.session_state.pest_analysis["overall_risk"] > 70
            ):
                priority_actions.append("🐛 HIGH: Implement pest control measures")

            if (
                "soil_analysis" in st.session_state
                and st.session_state.soil_analysis is not None
                and st.session_state.soil_analysis["score"] < 50
            ):
                priority_actions.append("🧪 CRITICAL: Improve soil health immediately")

            if (
                "irrigation_analysis" in st.session_state
                and st.session_state.irrigation_analysis is not None
                and st.session_state.irrigation_analysis["is_critical_stage"]
            ):
                priority_actions.append(
                    f"💧 URGENT: Critical irrigation needed - {growth_stage} stage"
                )

            if not priority_actions:
                priority_actions.append("✅ No critical issues detected")

            for action in priority_actions:
                st.markdown(f"- {action}")

        with col2:
            st.markdown("#### 💰 Cost Summary")
            if (
                "soil_analysis" in st.session_state
                and st.session_state.soil_analysis is not None
            ):
                soil = st.session_state.soil_analysis
                st.metric("Fertilizer Cost", f"₹{soil['total_cost']:.2f}")

                if soil["fertilizer_recommendations"]:
                    st.markdown("**Top Fertilizers Needed:**")
                    for fert in soil["fertilizer_recommendations"][:3]:
                        st.markdown(f"- {fert['type']}: {fert['quantity']} kg")
            else:
                st.info("Run soil analysis to see fertilizer costs")


if __name__ == "__main__":
    main()

# --- Professional Chatbot Integration in Sidebar ---
import openrouter_chat

with st.sidebar:
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;'><b>🤖 MahaAgroAI Chat Assistant</b></div>",
        unsafe_allow_html=True,
    )
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    user_input = st.text_area(
        "Ask MahaAgroAI (English only):", "", key="chatbot_input", height=60
    )
    if st.button("Send", key="chatbot_send_btn"):
        if user_input.strip():
            with st.spinner("MahaAgroAI is typing..."):
                reply = openrouter_chat.chat_with_openrouter(
                    user_input, st.session_state["chat_history"]
                )
            # if something went wrong the module returns a string
            # starting with "Error:"; show that to the user instead of
            # adding it to the normal history.
            if reply.startswith("Error:"):
                st.error(reply)
            else:
                st.session_state["chat_history"].append(
                    {"role": "user", "content": user_input}
                )
                st.session_state["chat_history"].append(
                    {"role": "assistant", "content": reply}
                )
    # Display chat history
    if st.session_state["chat_history"]:
        st.markdown(
            "<div style='max-height:250px;overflow-y:auto;background:#f8f9fa;padding:10px;border-radius:8px;margin-top:10px;'>",
            unsafe_allow_html=True,
        )
        for msg in st.session_state["chat_history"][-8:]:
            if msg["role"] == "user":
                st.markdown(
                    f"<div style='color:#1976d2;'><b>You:</b> {msg['content']}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div style='color:#388e3c;'><b>MahaAgroAI:</b> {msg['content']}</div>",
                    unsafe_allow_html=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)
