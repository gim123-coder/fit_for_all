pip install gradio
pip install langchain
pip install openai
pip install folium
pip install snowflake-connector-python bcrypt
{sys.executable} -m pip install snowflake-connector-python

import gradio as gr
import openai
import requests
from math import radians, sin, cos, sqrt, atan2
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
import config

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# -------------------------
# Haversine Distance Helper
# -------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return round(R * c, 2)


# -------------------------
# Snowflake
# -------------------------
def get_snowflake_connection():
    return snowflake.connector.connect(
        user=config.SNOWFLAKE_USER,
        password=config.SNOWFLAKE_PASSWORD,
        account=config.SNOWFLAKE_ACCOUNT,
        warehouse=config.SNOWFLAKE_WAREHOUSE,
        database=config.SNOWFLAKE_DATABASE,
        schema=config.SNOWFLAKE_SCHEMA
    )
  
def create_account(username, password):
    conn = get_snowflake_connection()
    cs = conn.cursor()
    hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    try:
        cs.execute(
            "INSERT INTO USER_ACCOUNTS (username, password_hash) VALUES (%s, %s)",
            (username, hashed_pw)
        )
        conn.commit()
        return "Account created successfully."
    except snowflake.connector.errors.ProgrammingError as e:
        if "unique constraint" in str(e).lower():
            return "Username already exists."
        else:
            return f"Error: {e}"
    finally:
        cs.close()
        conn.close()

# Login
def login(username, password):
    conn = get_snowflake_connection()
    cs = conn.cursor()
    try:
        cs.execute(
            "SELECT password_hash FROM USER_ACCOUNTS WHERE username = %s",
            (username,)
        )
        row = cs.fetchone()
        if row:
            stored_hash = row[0].encode()
            if bcrypt.checkpw(password.encode(), stored_hash):
                return True, "Login successful."
            else:
                return False, "Invalid password."
        else:
            return False, "User not found."
    except Exception as e:
        return False, f"Error: {e}"
    finally:
        cs.close()
        conn.close()

# -------------------------
# Gradio Login and Sign Up Interface
# -------------------------
def login_fn(username, password):
    success, msg = login(username, password)
    if success:
        return (
            gr.update(visible=False),  # Hide login
            gr.update(visible=False),  # Hide signup
            gr.update(visible=True),   # Show app
            msg
        )
    else:
        return (
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=False),
            msg
        )

def signup_fn(new_username, new_password):
    try:
        msg = create_account(new_username, new_password)
        return msg
    except Exception as e:
        return f"Error: {e}"


# -------------------------
# Google Places API Search
# -------------------------
def get_nearby_gyms(address, radius=5000, max_results=20):
    # Step 1: Geocode address
    geo_url = f"https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={GOOGLE_API_KEY}"
    geo_resp = requests.get(geo_url).json()
    if geo_resp["status"] != "OK":
        return [], f"Geocoding error: {geo_resp['status']}"

    lat = geo_resp["results"][0]["geometry"]["location"]["lat"]
    lng = geo_resp["results"][0]["geometry"]["location"]["lng"]

    # Step 2: Places Nearby Search
    places_url = (
        f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?"
        f"location={lat},{lng}&radius={radius}&keyword=gym&type=establishment&key={GOOGLE_API_KEY}"
    )
    places_resp = requests.get(places_url).json()
    if places_resp["status"] != "OK":
        return [], f"Places API error: {places_resp['status']}"

    gyms = []
    for place in places_resp.get("results", [])[:max_results]:
        gym_lat = place["geometry"]["location"]["lat"]
        gym_lng = place["geometry"]["location"]["lng"]
        distance_km = haversine(lat, lng, gym_lat, gym_lng)
        place_id = place.get("place_id")
        photo_url = None
        if "photos" in place:
            photo_ref = place["photos"][0]["photo_reference"]
            photo_url = (
                f"https://maps.googleapis.com/maps/api/place/photo"
                f"?maxwidth=400&photoreference={photo_ref}&key={GOOGLE_API_KEY}"
            )
        map_url = f"https://www.google.com/maps/place/?q=place_id:{place_id}"

        gyms.append({
            "name": place.get("name"),
            "vicinity": place.get("vicinity", "N/A"),
            "rating": place.get("rating", "N/A"),
            "distance_km": distance_km,
            "lat": gym_lat,
            "lng": gym_lng,
            "map_url": map_url,
            "photo_url": photo_url
        })

    return gyms, None

# --- LangChain Prompt ---
def build_prompt(region, disease, budget, gym_data):
    gyms_formatted = "\n".join([
        f"- {g['name']} ({g['vicinity']}, rating {g['rating']}, {g['distance_km']} km away)\n  Link: {g['map_url']}"
        for g in gym_data
    ])

    prompt_template = PromptTemplate(
        input_variables=["region", "disease", "budget", "gyms"],
        template=(
            "A user with the following profile is looking for a gym:\n"
            "- Region/postcode: {region}\n"
            "- Health condition: {disease}\n"
            "- Budget: {budget}\n\n"
            "Here are gyms found nearby:\n{gyms}\n\n"
            "Please pick 3 gyms that are most suitable. Consider:\n"
            "- If their condition suggests they need disability-friendly facilities\n"
            "- Whether the gym brand/location implies affordability (within budget)\n"
            "- Proximity to the user's region\n\n"
            "Format like this:\n"
            "1. [Gym Name] - [estimated price], [distance] – [accessibility note or reason for recommendation]"
        )
    )

    return prompt_template.format(region=region, disease=disease, budget=budget, gyms=gyms_formatted)

# --- OpenAI GPT Call ---
def get_gpt_recommendations(prompt):
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content




# -------------------------
# Call OpenAI API
# -------------------------
from openai import OpenAI

def get_gpt_recommendations(prompt):
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    
    return response.choices[0].message.content

# --- Folium Map ---
def create_map(gym_data, center_lat, center_lng):
    gym_map = folium.Map(location=[center_lat, center_lng], zoom_start=13)
    for gym in gym_data:
        popup = f"<b>{gym['name']}</b><br>{gym['vicinity']}<br>⭐ {gym['rating']}<br>{gym['distance_km']} km"
        if gym["photo_url"]:
            popup += f"<br><img src='{gym['photo_url']}' width='200'>"
        popup += f"<br><a href='{gym['map_url']}' target='_blank'>Open in Maps</a>"
        folium.Marker(
            location=[gym["lat"], gym["lng"]],
            popup=folium.Popup(popup, max_width=300),
            tooltip=gym["name"]
        ).add_to(gym_map)
    return gym_map._repr_html_()


# -------------------------
# Main Pipeline
# -------------------------
def gym_finder_pipeline(region, disease, budget):
    gym_data, error = get_nearby_gyms(region)
    if error:
        return error, None

    prompt = build_prompt(region, disease, budget, gym_data)
    try:
        recommendations = get_gpt_recommendations(prompt)
    except Exception as e:
        return f"OpenAI API Error: {str(e)}", None

    center_lat = gym_data[0]["lat"]
    center_lng = gym_data[0]["lng"]
    map_html = create_map(gym_data, center_lat, center_lng)

    return recommendations, map_html

# -------------------------
# Gradio Interface
# -------------------------
with gr.Blocks() as app:
    # Track login state
    logged_in = gr.State(False)

    # Login interface
    with gr.Column(visible=True) as login_section:
        gr.Markdown("### Login")
        login_username = gr.Textbox(label="Username")
        login_password = gr.Textbox(label="Password", type="password")
        login_btn = gr.Button("Login")
        login_msg = gr.Text(label="Login Status")

    # Sign-up interface
    with gr.Column(visible=True) as signup_section:
        gr.Markdown("### Or Sign Up")
        signup_username = gr.Textbox(label="New Username")
        signup_password = gr.Textbox(label="New Password", type="password")
        signup_btn = gr.Button("Sign Up")
        signup_msg = gr.Text(label="Sign Up Status")

    # Main app interface (initially hidden)
    with gr.Column(visible=False) as app_section:
        gr.Markdown("## Welcome to FitForAll")
        region_input = gr.Textbox(label="Your Region or Postcode")
        disease_input = gr.Textbox(label="Health Condition (optional)", placeholder="e.g. asthma")
        budget_input = gr.Textbox(label="Monthly Budget", placeholder="e.g. £30/month")
        submit_btn = gr.Button("Get Gym Recommendations")
        gpt_output = gr.Text(label="Recommended Gyms")
        map_output = gr.HTML(label="Map of Nearby Gyms")
        logout_btn = gr.Button("Logout")

    # --- Button functionality ---
    login_btn.click(
        fn=login_fn,
        inputs=[login_username, login_password],
        outputs=[login_section, signup_section, app_section, login_msg],
    )

    signup_btn.click(
        fn=signup_fn,
        inputs=[signup_username, signup_password],
        outputs=[signup_msg],
    )

    submit_btn.click(
        fn=gym_finder_pipeline,
        inputs=[region_input, disease_input, budget_input],
        outputs=[gpt_output, map_output]
    )

    logout_btn.click(
        fn=lambda: (gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), ""),
        inputs=[],
        outputs=[login_section, signup_section, app_section, login_msg]
    )

# Launch full app
if __name__ == "__main__":
    app.launch(share=True)