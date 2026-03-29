#!/usr/bin/env python3
"""Insert new high-priority methods into maharashtra_crop_system.py"""

with open("maharashtra_crop_system.py", "r", encoding="utf-8") as f:
    content = f.read()

# Find insertion point
marker = "    def generate_zone_risk_summary(self):"
location = content.find(marker)

if location == -1:
    print("ERROR: Could not find insertion point")
    exit(1)

# New methods
new_methods = '''    def get_pest_treatment_plans(self, crop_type, riskScore):
        """Get actionable treatment plans with products and pricing"""
        treatments = {
            "Cotton": {"Bollworm": [{"product": "Chlorpyriphos 20%", "price": "279/L"}, {"product": "Neem Oil", "price": "180/L"}], "Aphids": [{"product": "Imidacloprid", "price": "350/L"}]},
            "Rice": {"Brown Plant Hopper": [{"product": "Buprofezin", "price": "280/L"}]},
            "Tomato": {"Fruit Borer": [{"product": "Spinosad", "price": "280/L"}]}
        }
        severity = "Critical" if riskScore > 70 else "Moderate" if riskScore > 40 else "Low"
        return {"severity": severity, "treatments": treatments.get(crop_type, {}), "note": "Wear PPE"}

    def get_optimal_spraying_windows(self, weather):
        """Get optimal spray timing"""
        t = weather.get("temperature", 25)
        w = weather.get("wind_speed", 2.5)
        r = weather.get("rainfall", 0)
        
        warnings = []
        if r > 3: warnings.append(f"Rain {r}mm - Skip")
        if w > 3.5: warnings.append(f"Wind {w}m/s - Wait")
        
        windows = ["06-09 AM", "17-20 PM"] if (15 <= t <= 32 and w <= 3.5 and r <= 3) else ["Wait for better weather"]
        return {"windows": windows, "warnings": warnings, "reapply": "7-10 days"}

    def calculate_soil_moisture(self, soil_ph, daily_need, rainfall):
        """Calculate soil moisture (0-100%)"""
        rain_c = min(50, rainfall * 5)
        retention = 0.9 if soil_ph > 7.5 else 0.8 if 5.5 <= soil_ph <= 7.5 else 0.6
        demand = min(50, (daily_need / 10) * 10)
        moisture = int((rain_c * retention + (50 - demand)) * 0.7)
        moisture = max(10, min(95, moisture))
        
        if moisture > 70: status, action = "Very High", "Space irrigations"
        elif moisture > 50: status, action = "Adequate", "Continue"
        elif moisture > 30: status, action = "Low", "Water soon"
        else: status, action = "Critical", "Irrigate now"
        
        return {"level": f"{moisture}%", "status": status, "action": action}

    def get_rain_forecast_impact(self, expected_rain, daily_need):
        """Analyze rain impact on irrigation"""
        if expected_rain >= daily_need * 0.8:
            return {"recommendation": "Skip irrigation", "reason": "Rain will meet needs"}
        elif expected_rain >= daily_need * 0.4:
            return {"recommendation": f"Reduce by {(daily_need-expected_rain):.1f}mm", "reason": "Partial rain expected"}
        else:
            return {"recommendation": "Irrigate as planned", "reason": "Insufficient rain"}

'''

# Insert
new_content = content[:location] + new_methods + "\n    " + content[location:]

with open("maharashtra_crop_system.py", "w", encoding="utf-8") as f:
    f.write(new_content)

print("SUCCESS: New methods added!")
