import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import haversine, build_prompt, gym_finder_pipeline

# Optional: use mock if you're testing API-related functions
from unittest.mock import patch, Mock

def test_haversine():
    assert haversine(0, 0, 0, 0) == 0
    dist = haversine(51.5, -0.1, 40.7, -74.0)
    assert isinstance(dist, float)
    assert dist > 5000

def test_build_prompt():
    gyms = ["Gym A", "Gym B"]
    goals = "lose weight"
    constraints = "no machines"
    prompt = build_prompt(gyms, goals, constraints)
    assert isinstance(prompt, str)
    assert "lose weight" in prompt
    assert "no machines" in prompt

@patch("app.requests.get")
def test_get_nearby_gyms(mock_get):
    from app import get_nearby_gyms
    mock_response = {
        "results": [
            {"name": "Mock Gym", "geometry": {"location": {"lat": 0, "lng": 0}}},
        ]
    }
    mock_get.return_value.json.return_value = mock_response
    gyms = get_nearby_gyms("London", radius=1000, max_results=1)
    assert isinstance(gyms, list)
    assert "Mock Gym" in gyms[0]["name"]

@patch("app.openai.ChatCompletion.create")
def test_get_gpt_recommendations(mock_openai):
    from app import get_gpt_recommendations
    mock_openai.return_value = {
        "choices": [{"message": {"content": "Mock response"}}]
    }
    response = get_gpt_recommendations("some prompt")
    assert isinstance(response, str)
    assert "Mock response" in response

def test_create_map():
    from app import create_map
    locations = [{"name": "Test Gym", "lat": 0, "lng": 0}]
    map_html = create_map(locations)
    assert isinstance(map_html, str)
    assert "Test Gym" in map_html

@patch("app.get_nearby_gyms")
@patch("app.get_gpt_recommendations")
@patch("app.create_map")
def test_gym_finder_pipeline(mock_map, mock_gpt, mock_gyms):
    mock_gyms.return_value = [{"name": "Mock Gym", "lt": 0, "lng": 0}]
    mock_gpt.return_value = "GPT says: do squats"
    mock_map.return_value = "<iframe>Map</iframe>"

    result = gym_finder_pipeline("London", "gain muscle", "no cardio")
    assert isinstance(result, tuple)
    assert "GPT says" in result[0]
    assert "iframe" in result[1]
