"""
API TESTING SCRIPT

Quick script to test your FastAPI endpoints.

Usage:
1. Start the API server first:
   uvicorn app.main:app --reload

2. In another terminal, run this script:
   python test_api.py
"""

import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def test_health_check():
    """Test the health check endpoint"""
    print_section("TEST 1: Health Check")

    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")

        if response.status_code == 200:
            print("✅ Health check passed!")
        else:
            print("❌ Health check failed!")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("💡 Make sure the API server is running!")


def test_api_info():
    """Test the info endpoint"""
    print_section("TEST 2: API Info")

    try:
        response = requests.get(f"{BASE_URL}/info")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")

        if response.status_code == 200:
            print("✅ Info endpoint working!")
        else:
            print("❌ Info endpoint failed!")

    except Exception as e:
        print(f"❌ Error: {str(e)}")


def test_prediction_high_risk():
    """Test prediction with high-risk patient data"""
    print_section("TEST 3: Prediction - High Risk Patient")

    # High-risk patient: older, high BMI, poor lipid profile, history
    patient_data = {
        "age": 58.0,
        "gender": 1.0,
        "bmi": 32.5,
        "waist_to_hip_ratio": 0.98,
        "systolic_bp": 145.0,
        "diastolic_bp": 92.0,
        "heart_rate": 80.0,
        "cholesterol": 240.0,
        "ldl": 160.0,
        "hdl": 35.0,
        "triglycerides": 220.0,
        "physical_activity": 1.0,
        "screen_time": 7.0,
        "sleep_duration": 5.5,
        "hypertension_history": 1.0,
        "cardiovascular_history": 1.0,
        "family_history": 1.0
    }

    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=patient_data,
            headers={"Content-Type": "application/json"}
        )

        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")

        if response.status_code == 200:
            result = response.json()
            print(f"\n📊 Prediction Result:")
            print(f"   - Diabetes: {'YES' if result['prediction'] == 1 else 'NO'}")
            print(f"   - Probability: {result['probability']:.2%}")
            print(f"   - Risk Level: {result['risk_level']}")
            print("✅ Prediction successful!")
        else:
            print("❌ Prediction failed!")

    except Exception as e:
        print(f"❌ Error: {str(e)}")


def test_prediction_low_risk():
    """Test prediction with low-risk patient data"""
    print_section("TEST 4: Prediction - Low Risk Patient")

    # Low-risk patient: young, healthy BMI, good lipid profile, active
    patient_data = {
        "age": 28.0,
        "gender": 0.0,
        "bmi": 22.0,
        "waist_to_hip_ratio": 0.78,
        "systolic_bp": 115.0,
        "diastolic_bp": 75.0,
        "heart_rate": 65.0,
        "cholesterol": 170.0,
        "ldl": 100.0,
        "hdl": 60.0,
        "triglycerides": 90.0,
        "physical_activity": 5.0,
        "screen_time": 2.0,
        "sleep_duration": 8.0,
        "hypertension_history": 0.0,
        "cardiovascular_history": 0.0,
        "family_history": 0.0
    }

    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=patient_data,
            headers={"Content-Type": "application/json"}
        )

        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")

        if response.status_code == 200:
            result = response.json()
            print(f"\n📊 Prediction Result:")
            print(f"   - Diabetes: {'YES' if result['prediction'] == 1 else 'NO'}")
            print(f"   - Probability: {result['probability']:.2%}")
            print(f"   - Risk Level: {result['risk_level']}")
            print("✅ Prediction successful!")
        else:
            print("❌ Prediction failed!")

    except Exception as e:
        print(f"❌ Error: {str(e)}")


def test_invalid_data():
    """Test API validation with invalid data"""
    print_section("TEST 5: Data Validation (Should Fail)")

    # Missing required fields
    invalid_data = {
        "age": 45.0,
        "bmi": 28.0
        # Missing other required fields
    }

    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=invalid_data,
            headers={"Content-Type": "application/json"}
        )

        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")

        if response.status_code == 422:
            print("✅ Validation working correctly (rejected invalid data)!")
        else:
            print("❌ Validation should have failed!")

    except Exception as e:
        print(f"❌ Error: {str(e)}")


def run_all_tests():
    """Run all tests"""
    print("\n" + "🧪" * 30)
    print("  DIABETES PREDICTION API - TEST SUITE")
    print("🧪" * 30)

    test_health_check()
    test_api_info()
    test_prediction_high_risk()
    test_prediction_low_risk()
    test_invalid_data()

    print_section("ALL TESTS COMPLETE")
    print("\n💡 Next steps:")
    print("1. Check the interactive docs: http://localhost:8000/docs")
    print("2. Try the API manually with different patient data")
    print("3. Build the HTML frontend next!\n")


if __name__ == "__main__":
    run_all_tests()
