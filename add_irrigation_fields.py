#!/usr/bin/env python3
"""Add soil moisture and rain forecast to irrigation recommendations"""

with open("maharashtra_crop_system.py", "r", encoding="utf-8") as f:
    content = f.read()

# Find and replace the recommendations dictionary ending
old_text = """            "recommendations": self.generate_irrigation_recommendations(
                zone, crop_type, growth_stage, final_daily_need
            ),
        }

        return recommendations"""

new_text = """            "recommendations": self.generate_irrigation_recommendations(
                zone, crop_type, growth_stage, final_daily_need
            ),
            "soil_moisture": self.calculate_soil_moisture(soil_ph, final_daily_need, current_weather.get("rainfall", 0)),
            "rain_forecast_impact": self.get_rain_forecast_impact(current_weather.get("rainfall", 0), final_daily_need),
        }

        return recommendations"""

if old_text in content:
    content = content.replace(old_text, new_text)
    with open("maharashtra_crop_system.py", "w", encoding="utf-8") as f:
        f.write(content)
    print(
        "✅ Added soil_moisture and rain_forecast_impact to irrigation recommendations"
    )
else:
    print("❌ Could not find target text in file")
