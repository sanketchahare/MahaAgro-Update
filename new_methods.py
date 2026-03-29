    def get_pest_treatment_plans(self, crop_type, riskScore):
        """Get actionable pest treatment plans with products and pricing"""
        treatments = {
            "Cotton": {
                "Bollworm": [
                    {"product": "Chlorpyriphos 20% EC", "dosage": "2ml/L", "price": "279/L", "target": "Larvae"},
                    {"product": "Neem Oil 3%", "dosage": "4ml/L", "price": "180/L", "target": "Early stage"}
                ],
                "Aphids": [
                    {"product": "Imidacloprid 17.8% SL", "dosage": "0.5ml/L", "price": "350/L", "target": "All stages"}
                ]
            },
            "Rice": {
                "Brown Plant Hopper": [
                    {"product": "Buprofezin 25% SC", "dosage": "1ml/L", "price": "280/L", "target": "Nymphs"}
                ]
            },
            "Tomato": {
                "Fruit Borer": [
                    {"product": "Spinosad 45% SC", "dosage": "0.5ml/L", "price": "280/L", "target": "In fruits"}
                ]
            }
        }
        severity = "Critical" if riskScore > 70 else "Moderate" if riskScore > 40 else "Low"
        ct = treatments.get(crop_type, {})
        return {"severity": severity, "treatment_options": ct, "safety": "Wear PPE. Follow label. No rain/high wind spraying."}

    def get_optimal_spraying_windows(self, weather):
        """Get optimal spray timing based on weather conditions"""
        t = weather.get("temperature", 25)
        h = weather.get("humidity", 65)
        w = weather.get("wind_speed", 2.5)
        r = weather.get("rainfall", 0)
        
        warn = []
        if r > 3:
            warn.append(f"Rain {r}mm - Postpone spraying")
        if w > 3.5:
            warn.append(f"Wind {w}m/s too high")
        if t < 15 or t > 35:
            warn.append(f"Temperature {t}C - Ideal 20-30C")
        
        best = ["06:00-09:00 AM", "17:00-20:00 PM"] if (t >= 15 and t <= 32 and w <= 3.5 and r <= 3) else ["Wait for better conditions"]
        
        return {
            "current": {"temperature": f"{t}C", "humidity": f"{h}%", "wind": f"{w}m/s", "rain": f"{r}mm"},
            "optimal_windows": best,
            "warnings": warn,
            "reapply_after": "7-10 days"
        }

    def calculate_soil_moisture(self, soil_ph, daily_need, rainfall):
        """Calculate soil moisture indicator (0-100%)"""
        rainfall_contribution = min(50, rainfall * 5)
        if soil_ph < 5.5:
            retention = 0.6
            soil_type = "Acidic/Sandy"
        elif 5.5 <= soil_ph <= 7.5:
            retention = 0.8
            soil_type = "Neutral/Loamy"
        else:
            retention = 0.9
            soil_type = "Alkaline/Clay"
        
        demand = min(50, (daily_need / 10) * 10)
        moisture = int((rainfall_contribution * retention + (50 - demand)) * 0.7)
        moisture = max(10, min(95, moisture))
        
        if moisture > 70:
            status = "Very High - Risk of waterlogging"
            action = "Space out irrigations"
            risk = "high"
        elif moisture > 50:
            status = "Adequate"
            action = "Maintain current schedule"
            risk = "low"
        elif moisture > 30:
            status = "Low - Irrigate soon"
            action = "Water within 1-2 days"
            risk = "medium"
        else:
            status = "Critical"
            action = "Irrigate immediately"
            risk = "critical"
        
        return {
            "level": f"{moisture}%",
            "soil_type": soil_type,
            "status": status,
            "action": action,
            "risk": risk
        }

    def get_rain_forecast_impact(self, expected_rain, daily_need):
        """Analyze rain impact on irrigation schedule"""
        if expected_rain >= daily_need * 0.8:
            rec = "Skip irrigation - Rain will meet needs"
            action = "Wait for rain. Monitor soil after."
            adj = "Skip today"
            color = "green"
        elif expected_rain >= daily_need * 0.4:
            rec = f"Reduce irrigation by {(daily_need - expected_rain):.1f}mm"
            action = f"Apply only {expected_rain:.1f}mm + top up after rain"
            adj = f"Reduce by {expected_rain:.1f}mm"
            color = "yellow"
        else:
            rec = f"Irrigate as planned - Rain insufficient"
            action = "Continue full schedule"
            adj = "No adjustment"
            color = "orange"
        
        return {
            "expected_rainfall": f"{expected_rain}mm",
            "water_needed": f"{daily_need:.1f}mm",
            "recommendation": rec,
            "action": action,
            "adjustment": adj,
            "status_color": color
        }
