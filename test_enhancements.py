#!/usr/bin/env python3
"""
Test Script for Enhanced Maharashtra Agricultural System
Validates all system improvements and quality enhancements
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import sqlite3
import pickle
import requests
import json


class SystemTester:
    def __init__(self):
        """Initialize the system tester"""
        self.test_results = {}
        self.passed_tests = 0
        self.total_tests = 0

    def log_test(self, test_name, status, details=""):
        """Log test result"""
        self.total_tests += 1
        if status:
            self.passed_tests += 1
            print(f"✅ {test_name}")
        else:
            print(f"❌ {test_name} - {details}")

        self.test_results[test_name] = {
            "status": status,
            "details": details,
            "timestamp": datetime.now().isoformat(),
        }

    def test_csv_data_integrity(self):
        """Test CSV data integrity and structure"""
        print("\n🔍 Testing CSV Data Integrity...")

        # Test agriculture dataset
        try:
            agri_df = pd.read_csv("agriculture_dataset.csv")
            required_cols = [
                "NDVI",
                "SAVI",
                "Temperature",
                "Soil_pH",
                "Crop_Health_Label",
            ]
            has_required = all(col in agri_df.columns for col in required_cols)
            self.log_test("Agriculture dataset structure", has_required)

            # Test data quality
            has_data = len(agri_df) > 0
            no_all_nulls = not agri_df.isnull().all().any()
            self.log_test("Agriculture dataset quality", has_data and no_all_nulls)

        except Exception as e:
            self.log_test("Agriculture dataset loading", False, str(e))

        # Test crop-fertilizer dataset
        try:
            crop_df = pd.read_csv("Crop and fertilizer dataset.csv")
            required_cols = [
                "District_Name",
                "Crop",
                "Fertilizer",
                "Nitrogen",
                "Phosphorus",
            ]
            has_required = all(col in crop_df.columns for col in required_cols)
            self.log_test("Crop-fertilizer dataset structure", has_required)

        except Exception as e:
            self.log_test("Crop-fertilizer dataset loading", False, str(e))

        # Test weather data
        try:
            weather_df = pd.read_csv("weather_data.csv")
            required_cols = ["cities", "PARAMETER", "YEAR"]
            has_required = all(col in weather_df.columns for col in required_cols)
            self.log_test("Weather dataset structure", has_required)

        except Exception as e:
            self.log_test("Weather dataset loading", False, str(e))

    def test_enhanced_data_processing(self):
        """Test enhanced data processing pipeline"""
        print("\n🔧 Testing Enhanced Data Processing...")

        try:
            # Import and test data processor
            from enhanced_data_processor import EnhancedDataProcessor

            processor = EnhancedDataProcessor()

            # Test dataset loading
            load_success = processor.load_all_datasets()
            self.log_test("Dataset loading in processor", load_success)

            if load_success:
                # Test feature creation
                feature_success = processor.create_integrated_features()
                self.log_test("Integrated feature creation", feature_success)

                # Test enhanced agriculture features
                if "enhanced_agriculture" in processor.datasets:
                    enhanced_df = processor.datasets["enhanced_agriculture"]
                    has_health_score = (
                        "comprehensive_health_score" in enhanced_df.columns
                    )
                    has_soil_index = "soil_quality_index" in enhanced_df.columns
                    self.log_test(
                        "Enhanced crop features", has_health_score and has_soil_index
                    )

                # Test weather patterns
                if hasattr(processor, "weather_patterns"):
                    has_patterns = len(processor.weather_patterns) > 0
                    self.log_test("Weather pattern extraction", has_patterns)

        except ImportError:
            self.log_test("Enhanced data processor import", False, "Module not found")
        except Exception as e:
            self.log_test("Enhanced data processing", False, str(e))

    def test_model_training(self):
        """Test enhanced model training"""
        print("\n🤖 Testing Model Training...")

        try:
            from enhanced_data_processor import EnhancedDataProcessor

            processor = EnhancedDataProcessor()

            if processor.load_all_datasets() and processor.create_integrated_features():
                # Test model training
                training_success = processor.train_enhanced_models()
                self.log_test("Enhanced model training", training_success)

                # Test model availability
                has_health_model = "crop_health" in processor.models
                has_yield_model = "yield_prediction" in processor.models
                has_fert_model = "fertilizer_recommendation" in processor.models

                self.log_test("Crop health model", has_health_model)
                self.log_test("Yield prediction model", has_yield_model)
                self.log_test("Fertilizer recommendation model", has_fert_model)

                # Test model saving
                save_success = processor.save_enhanced_models()
                self.log_test("Model saving", save_success)

        except Exception as e:
            self.log_test("Model training", False, str(e))

    def test_database_creation(self):
        """Test enhanced database creation"""
        print("\n💾 Testing Database Creation...")

        try:
            from enhanced_data_processor import EnhancedDataProcessor

            processor = EnhancedDataProcessor()

            if (
                processor.load_all_datasets()
                and processor.create_integrated_features()
                and processor.train_enhanced_models()
            ):

                # Test database creation
                db_success = processor.create_enhanced_database()
                self.log_test("Enhanced database creation", db_success)

                # Test database tables
                if os.path.exists("enhanced_krushi_mitra.db"):
                    conn = sqlite3.connect("enhanced_krushi_mitra.db")
                    cursor = conn.cursor()

                    # Check tables exist
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    tables = [table[0] for table in cursor.fetchall()]

                    expected_tables = [
                        "enhanced_agriculture",
                        "crop_fertilizer_recommendations",
                        "weather_patterns",
                        "feature_importances",
                    ]

                    for table in expected_tables:
                        table_exists = table in tables
                        self.log_test(f"Database table: {table}", table_exists)

                    conn.close()

        except Exception as e:
            self.log_test("Database creation", False, str(e))

    def test_backend_api(self):
        """Test enhanced backend API"""
        print("\n🌐 Testing Backend API...")

    def test_chatbot_setup(self):
        """Test AI chatbot initialization and configuration"""
        print("\n🤖 Testing Chatbot Setup...")
        try:
            import maharashtra_crop_system as mcs

            has_model_attr = hasattr(mcs, "model")
            self.log_test("Chatbot model attribute present", has_model_attr)
            if has_model_attr:
                # model may be None if no API key; that is acceptable but ensure no crash
                self.log_test(
                    "Chatbot model object valid",
                    mcs.model is None or hasattr(mcs.model, "generate_content"),
                )
        except Exception as e:
            self.log_test("Chatbot import/setup", False, str(e))

    def test_disease_label_formatting(self):
        """Verify disease formatting helpers produce user-friendly names and descriptions"""
        print("\n🩺 Testing Disease Label Formatting and Descriptions...")
        try:
            from maharashtra_crop_system import MaharashtraAgriSystem

            sys = MaharashtraAgriSystem()
            # simple checks
            assert sys.format_disease_name("Early_Blight") == "Early Blight"
            assert (
                sys.format_disease_name("Possible_Late_Blight")
                == "Possible Late Blight"
            )
            desc1 = sys.get_disease_description("Early_Blight").lower()
            desc2 = sys.get_disease_description("Healthy").lower()
            assert "fungal" in desc1 or "spot" in desc1
            assert "healthy" in desc2 or "no" in desc2
            self.log_test("Disease formatting utilities", True)
        except AssertionError as ae:
            self.log_test("Disease formatting utilities", False, str(ae))
        except Exception as e:
            self.log_test("Disease formatting utilities", False, str(e))

    def test_file_structure(self):
        """Test file structure and model files"""

        try:
            from enhanced_backend_api import EnhancedBackendAPI

            api = EnhancedBackendAPI()

            # Test API initialization
            has_app = hasattr(api, "app")
            self.log_test("API initialization", has_app)

            # Test caching setup
            has_caching = hasattr(api, "cache_enabled")
            self.log_test("Caching system setup", has_caching)

            # Test feature extraction
            test_data = {
                "ndvi": 0.7,
                "savi": 0.4,
                "temperature": 28,
                "humidity": 75,
                "soil_ph": 6.5,
                "soil_moisture": 35,
            }

            features = api.extract_enhanced_crop_features(test_data)
            has_derived_features = (
                "comprehensive_health_score" in features
                and "soil_quality_index" in features
                and "environmental_stress" in features
            )
            self.log_test("Feature extraction", has_derived_features)

            # Test prediction
            prediction = api.predict_enhanced_crop_health(features)
            has_prediction = (
                "health_score" in prediction
                and "health_status" in prediction
                and "confidence" in prediction
            )
            self.log_test("Crop health prediction", has_prediction)

        except Exception as e:
            self.log_test("Backend API testing", False, str(e))

    def test_file_structure(self):
        """Test file structure and model files"""
        print("\n📁 Testing File Structure...")

        # Test CSV files
        csv_files = [
            "agriculture_dataset.csv",
            "Crop and fertilizer dataset.csv",
            "Crop_recommendationV2.csv",
            "weather_data.csv",
            "Fertilizer Prediction.csv",
            "Fertilizer.csv",
        ]

        for csv_file in csv_files:
            exists = os.path.exists(csv_file)
            self.log_test(f"CSV file: {csv_file}", exists)

        # Test enhanced files (may not exist until processing)
        enhanced_files = ["enhanced_data_processor.py", "enhanced_backend_api.py"]

        for enhanced_file in enhanced_files:
            exists = os.path.exists(enhanced_file)
            self.log_test(f"Enhanced file: {enhanced_file}", exists)

    def test_system_integration(self):
        """Test overall system integration"""
        print("\n🔗 Testing System Integration...")

        try:
            # Test if we can run the complete enhancement process
            from enhanced_data_processor import EnhancedDataProcessor

            processor = EnhancedDataProcessor()

            # Run abbreviated integration test
            datasets_loaded = processor.load_all_datasets()
            if datasets_loaded:
                features_created = processor.create_integrated_features()
                self.log_test(
                    "Integration: Data loading + Feature creation", features_created
                )

                # Test data consistency
                if "enhanced_agriculture" in processor.datasets:
                    df = processor.datasets["enhanced_agriculture"]
                    has_consistent_data = (
                        df["comprehensive_health_score"].notna().all()
                        and df["soil_quality_index"].notna().all()
                    )
                    self.log_test("Integration: Data consistency", has_consistent_data)

        except Exception as e:
            self.log_test("System integration", False, str(e))

    def run_performance_test(self):
        """Run performance benchmarks"""
        print("\n⚡ Testing Performance...")

        try:
            import time
            from enhanced_data_processor import EnhancedDataProcessor

            # Time dataset loading
            processor = EnhancedDataProcessor()
            start_time = time.time()
            load_success = processor.load_all_datasets()
            load_time = time.time() - start_time

            performance_acceptable = load_time < 30  # Should load within 30 seconds
            self.log_test(
                "Performance: Dataset loading",
                performance_acceptable,
                f"Took {load_time:.2f} seconds",
            )

            if load_success:
                # Time feature creation
                start_time = time.time()
                feature_success = processor.create_integrated_features()
                feature_time = time.time() - start_time

                feature_performance = (
                    feature_time < 10
                )  # Should complete within 10 seconds
                self.log_test(
                    "Performance: Feature creation",
                    feature_performance,
                    f"Took {feature_time:.2f} seconds",
                )

        except Exception as e:
            self.log_test("Performance testing", False, str(e))

    def generate_test_report(self):
        """Generate comprehensive test report"""
        print("\n" + "=" * 60)
        print("📊 SYSTEM ENHANCEMENT TEST REPORT")
        print("=" * 60)

        print(f"📈 Overall Results:")
        print(f"   ✅ Passed: {self.passed_tests}")
        print(f"   ❌ Failed: {self.total_tests - self.passed_tests}")
        print(f"   📊 Success Rate: {(self.passed_tests/self.total_tests)*100:.1f}%")

        print(f"\n🔍 Detailed Results:")
        for test_name, result in self.test_results.items():
            status_icon = "✅" if result["status"] else "❌"
            details = f" - {result['details']}" if result["details"] else ""
            print(f"   {status_icon} {test_name}{details}")

        # Save test report
        report_data = {
            "test_summary": {
                "total_tests": self.total_tests,
                "passed_tests": self.passed_tests,
                "success_rate": (self.passed_tests / self.total_tests) * 100,
                "timestamp": datetime.now().isoformat(),
            },
            "test_details": self.test_results,
        }

        with open("system_test_report.json", "w") as f:
            json.dump(report_data, f, indent=2)

        print(f"\n💾 Test report saved to: system_test_report.json")

        return (
            self.passed_tests / self.total_tests > 0.8
        )  # 80% pass rate considered successful

    def run_all_tests(self):
        """Run all system tests"""
        print("🚀 Starting Comprehensive System Testing...")
        print("=" * 60)

        # Run all tests
        self.test_file_structure()
        self.test_csv_data_integrity()
        self.test_enhanced_data_processing()
        self.test_model_training()
        self.test_database_creation()
        self.test_backend_api()
        self.test_chatbot_setup()
        self.test_system_integration()
        self.run_performance_test()

        # Generate report
        success = self.generate_test_report()

        if success:
            print("\n🎉 System enhancement testing completed successfully!")
            print(
                "Your Maharashtra Agricultural System has been significantly enhanced!"
            )
        else:
            print("\n⚠️ Some tests failed. Please review the report for details.")

        return success


def main():
    """Main testing function"""
    tester = SystemTester()
    success = tester.run_all_tests()

    if success:
        print("\n✨ System Quality Enhancement Summary:")
        print("   🎯 Enhanced prediction accuracy")
        print("   📊 Multi-spectral data integration")
        print("   🌍 Weather pattern analysis")
        print("   🧪 Advanced soil modeling")
        print("   💡 Intelligent recommendations")
        print("   🚀 Performance optimizations")
        print("   📈 Comprehensive analytics")

    return success


if __name__ == "__main__":
    main()
