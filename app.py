import os
import gradio as gr
from typing import Annotated, TypedDict, List, Union
from langgraph.graph import StateGraph, END
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
import json
import requests
from math import radians, sin, cos, sqrt, atan2
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
import folium
from openai import OpenAI
import snowflake.connector
import bcrypt
import config as snowflake_config # Import the config file

# Load environment variables (keeping this for consistency, but will use direct keys)
load_dotenv()

# Directly using the API keys from api_keys.ipynb



# Snowflake Connection Configuration
# Now loading directly from the imported config.py
class SnowflakeConfig:
    SNOWFLAKE_USER = snowflake_config.SNOWFLAKE_USER
    SNOWFLAKE_PASSWORD = snowflake_config.SNOWFLAKE_PASSWORD
    SNOWFLAKE_ACCOUNT = snowflake_config.SNOWFLAKE_ACCOUNT
    SNOWFLAKE_WAREHOUSE = snowflake_config.SNOWFLAKE_WAREHOUSE
    SNOWFLAKE_DATABASE = snowflake_config.SNOWFLAKE_DATABASE
    SNOWFLAKE_SCHEMA = snowflake_config.SNOWFLAKE_SCHEMA

# -------------------------
# Snowflake and Authentication Functions
# -------------------------
def get_snowflake_connection():
    try:
        conn = snowflake.connector.connect(
            user=SnowflakeConfig.SNOWFLAKE_USER,
            password=SnowflakeConfig.SNOWFLAKE_PASSWORD,
            account=SnowflakeConfig.SNOWFLAKE_ACCOUNT,
            warehouse=SnowflakeConfig.SNOWFLAKE_WAREHOUSE,
            database=SnowflakeConfig.SNOWFLAKE_DATABASE,
            schema=SnowflakeConfig.SNOWFLAKE_SCHEMA
        )
        return conn
    except Exception as e:
        print(f"Snowflake connection error: {e}")
        return None

def create_account(username, password):
    conn = get_snowflake_connection()
    if not conn:
        return "Failed to connect to database."
    cs = conn.cursor()
    hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    try:
        cs.execute(
            "CREATE TABLE IF NOT EXISTS USER_ACCOUNTS (username VARCHAR UNIQUE, password_hash VARCHAR)"
        )
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
            return f"Error creating account: {e}"
    except Exception as e:
        return f"An unexpected error occurred during account creation: {e}"
    finally:
        cs.close()
        conn.close()

def login(username, password):
    conn = get_snowflake_connection()
    if not conn:
        return False, "Failed to connect to database."
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
        return False, f"Error during login: {e}"
    finally:
        cs.close()
        conn.close()

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
# AI Fitness Coach Functions
# -------------------------

def get_openai_llm(model_name="gpt-3.5-turbo"):
    return ChatOpenAI(model=model_name, temperature=0.7, openai_api_key=OPENAI_API_KEY)

# State definition
class State(TypedDict):
    user_data: dict
    fitness_plan: str
    feedback: str
    progress: List[str]
    messages: Annotated[List[Union[HumanMessage, AIMessage]], add_messages]

# Agents
def user_input_agent(state: State, llm):
    prompt = ChatPromptTemplate.from_template(
        "\"\"\"You are an AI fitness assistant. Process the user input and return a cleaned JSON profile.\n\n        Input:\n        {user_input}\n\n        Output only valid JSON.\"\"\""
    )
    chain = prompt | llm | StrOutputParser()
    user_profile = chain.invoke({"user_input": json.dumps(state["user_data"])})
    try:
        state["user_data"] = json.loads(user_profile)
    except json.JSONDecodeError:
        pass
    state["messages"].append(AIMessage(content=f"Processed user profile: {json.dumps(state['user_data'], indent=2)}"))
    return state

def routine_generation_agent(state: State, llm):
    prompt = ChatPromptTemplate.from_template(
        "\"\"\"You are an AI fitness coach. Based on the user's profile below, create a short weekly workout plan.\n\n        Profile:\n        {user_data}\n\n        Use only the days listed in \"workout_days\".\n        For each workout day, return:\n        - Day\n        - Activity (based on workout_preferences)\n        - Duration (based on workout_duration)\n        - Intensity (based on activity_level and health_conditions)\n\n        Avoid long descriptions. Output in bullet points. End with 1 simple dietary tip.\"\"\""
    )
    chain = prompt | llm | StrOutputParser()
    plan = chain.invoke({"user_data": json.dumps(state["user_data"])})
    state["fitness_plan"] = plan
    state["messages"].append(AIMessage(content=f"Generated fitness plan: {plan}"))
    return state

def feedback_collection_agent(state: State, llm):
    prompt = ChatPromptTemplate.from_template(
        "\"\"\"Summarize this user feedback and suggest 1 minor adjustment:\n\n        Plan:\n        {current_plan}\n\n        Feedback:\n        {user_feedback}\"\"\""
    )
    chain = prompt | llm | StrOutputParser()
    feedback_summary = chain.invoke({
        "current_plan": state["fitness_plan"],
        "user_feedback": state["feedback"]
    })
    state["messages"].append(AIMessage(content=f"Feedback analysis: {feedback_summary}"))
    return state

def routine_adjustment_agent(state: State, llm):
    prompt = ChatPromptTemplate.from_template(
        "\"\"\"Update the following fitness plan based on this feedback:\n\n        Plan:\n        {current_plan}\n\n        Feedback:\n        {feedback}\n\n        Return a concise bullet list plan (same format) with adjustments.\"\"\""
    )
    chain = prompt | llm | StrOutputParser()
    updated_plan = chain.invoke({
        "current_plan": state["fitness_plan"],
        "feedback": state["feedback"]
    })
    state["fitness_plan"] = updated_plan
    state["messages"].append(AIMessage(content=f"Updated fitness plan: {updated_plan}"))
    return state

def progress_monitoring_agent(state: State, llm):
    prompt = ChatPromptTemplate.from_template(
        "\"\"\"Summarize progress and suggest next steps.\n\n        Data: {user_data}\n        Plan: {current_plan}\n        History: {progress_history}\"\"\""
    )
    chain = prompt | llm | StrOutputParser()
    progress_update = chain.invoke({
        "user_data": str(state["user_data"]),
        "current_plan": state["fitness_plan"],
        "progress_history": str(state["progress"])
    })
    state["progress"].append(progress_update)
    state["messages"].append(AIMessage(content=f"Progress update: {progress_update}"))
    return state

def motivational_agent(state: State, llm):
    prompt = ChatPromptTemplate.from_template(
        "\"\"\"Give a 2-line motivational message and 1 fitness tip.\n\n        User:\n        {user_data}\n\n        Plan:\n        {current_plan}\n\n        Recent Progress:\n        {recent_progress}\"\"\""
    )
    chain = prompt | llm | StrOutputParser()
    motivation = chain.invoke({
        "user_data": str(state["user_data"]),
        "current_plan": state["fitness_plan"],
        "recent_progress": state["progress"][-1] if state["progress"] else ""
    })
    state["messages"].append(AIMessage(content=f"Motivation: {motivation}"))
    return state

# Main Coach Class
class AIFitnessCoach:
    def __init__(self):
        self.llm = get_openai_llm()
        self.graph = self.create_graph()

    def create_graph(self):
        workflow = StateGraph(State)
        workflow.add_node("user_input", lambda state: user_input_agent(state, self.llm))
        workflow.add_node("routine_generation", lambda state: routine_generation_agent(state, self.llm))
        workflow.add_node("feedback_collection", lambda state: feedback_collection_agent(state, self.llm))
        workflow.add_node("routine_adjustment", lambda state: routine_adjustment_agent(state, self.llm))
        workflow.add_node("progress_monitoring", lambda state: progress_monitoring_agent(state, self.llm))
        workflow.add_node("motivation", lambda state: motivational_agent(state, self.llm))

        workflow.add_edge("user_input", "routine_generation")
        workflow.add_edge("routine_generation", "feedback_collection")
        workflow.add_edge("feedback_collection", "routine_adjustment")
        workflow.add_edge("routine_adjustment", "progress_monitoring")
        workflow.add_edge("progress_monitoring", "motivation")
        workflow.add_edge("motivation", END)
        workflow.set_entry_point("user_input")
        return workflow.compile()

    def run(self, user_input):
        initial_state = State(
            user_data=user_input,
            fitness_plan="",
            feedback=user_input.get("feedback", ""),
            progress=[],
            messages=[HumanMessage(content=json.dumps(user_input))]
        )
        final_state = self.graph.invoke(initial_state)
        return final_state["messages"]

def process_user_input(age, weight, height, gender, primary_goal, target_timeframe, workout_preferences,
                       workout_duration, workout_days, activity_level, health_conditions):
    user_data = {
        "age": age,
        "weight": weight,
        "height": height,
        "gender": gender,
        "primary_goal": primary_goal,
        "target_timeframe": target_timeframe,
        "workout_preferences": workout_preferences,
        "workout_duration": workout_duration,
        "workout_days": workout_days,
        "activity_level": activity_level,
        "health_conditions": health_conditions,
        "feedback": ""
    }

    coach = AIFitnessCoach()
    messages = coach.run(user_data)

    fitness_plan = ""
    progress = ""
    motivation = ""

    for message in messages:
        if message.type == "ai":
            content = message.content.strip()

            if "Generated fitness plan:" in content:
                fitness_plan = content.split("Generated fitness plan:")[-1].strip()
            elif "Updated fitness plan:" in content:
                fitness_plan = content.split("Updated fitness plan:")[-1].strip()
            elif "Progress update:" in content:
                progress = content.split("Progress update:")[-1].strip()
            elif "Motivation:" in content:
                motivation = content.split("Motivation:")[-1].strip()

    final_output = []

    if fitness_plan:
        final_output.append("🗓️ Weekly Plan:\n" + fitness_plan)
    if progress:
        final_output.append("📈 Progress Summary:\n" + progress)
    if motivation:
        final_output.append("💬 Motivation:\n" + motivation)

    return "\n\n".join(final_output).strip() or "No fitness plan could be generated. Please try again."

# -------------------------
# Gym Finder Functions
# -------------------------

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return round(R * c, 2)

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

def build_prompt(region, budget, gym_data): # Removed 'disease' parameter
    gyms_formatted = "\n".join([
        f"- {g['name']} ({g['vicinity']}, rating {g['rating']}, {g['distance_km']} km away)\n  Link: {g['map_url']}"
        for g in gym_data
    ])

    prompt_template = PromptTemplate(
        input_variables=["region", "budget", "gyms"], # Removed "disease"
        template=(
            "A user with the following profile is looking for a gym:\n"
            "- Region/postcode: {region}\n"
            "- Budget: {budget}\n\n" # Removed health condition line
            "Here are gyms found nearby:\n{gyms}\n\n"
            "Please pick 3 gyms that are most suitable. Consider:\n"
            "- Whether the gym brand/location implies affordability (within budget)\n"
            "- Proximity to the user's region\n\n"
            "Format like this:\n"
            "1. [Gym Name] - [estimated price], [distance] – [reason for recommendation]" # Adjusted accessibility note
        )
    )

    return prompt_template.format(region=region, budget=budget, gyms=gyms_formatted) # Removed disease

def get_gpt_recommendations(prompt):
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content

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

def gym_finder_pipeline(region, budget): # Removed 'disease' parameter
    gym_data, error = get_nearby_gyms(region)
    if error:
        return error, None
    if not gym_data:
        return "No gyms found for the given region. Please try a different location.", None

    prompt = build_prompt(region, budget, gym_data) # Removed disease
    try:
        recommendations = get_gpt_recommendations(prompt)
    except Exception as e:
        return f"OpenAI API Error: {str(e)}", None

    # Use the coordinates of the first found gym as the center for the map
    center_lat = gym_data[0]["lat"]
    center_lng = gym_data[0]["lng"]
    map_html = create_map(gym_data, center_lat, center_lng)

    return recommendations, map_html

# -------------------------
# Unified Gradio Interface
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
        logout_btn = gr.Button("Logout")

        with gr.Tab("AI Fitness Coach"):
            gr.Markdown("### 🏋️‍♀️ Create Your Personalized Fitness Plan")
            with gr.Row():
                age = gr.Number(label="Age")
                weight = gr.Number(label="Weight (kg)")
                height = gr.Number(label="Height (cm)")
                gender = gr.Radio(["Male", "Female", "Other"], label="Gender")

            primary_goal = gr.Dropdown(["Weight loss", "Muscle gain", "Endurance improvement", "General fitness"], label="Primary Goal")
            target_timeframe = gr.Dropdown(["3 months", "6 months", "1 year"], label="Target Timeframe")

            workout_preferences = gr.CheckboxGroup(
                ["Cardio", "Strength training", "Yoga", "Pilates", "Flexibility exercises", "HIIT"],
                label="Workout Type Preferences"
            )
            workout_duration = gr.Slider(15, 120, step=15, label="Preferred Workout Duration (minutes)")
            workout_days = gr.CheckboxGroup(
                ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                label="Preferred Workout Days"
            )

            activity_level = gr.Radio(
                ["Sedentary", "Lightly active", "Moderately active", "Highly active"],
                label="Current Activity Level"
            )
            health_conditions = gr.Textbox(label="Health Conditions or Injuries")

            create_button = gr.Button("Create Fitness Plan")
            plan_output = gr.Textbox(label="Your Personalized Fitness Plan", lines=12)

            create_button.click(
                process_user_input,
                inputs=[
                    age, weight, height, gender, primary_goal, target_timeframe,
                    workout_preferences, workout_duration, workout_days,
                    activity_level, health_conditions
                ],
                outputs=plan_output
            )

        with gr.Tab("Gym Finder"):
            gr.Markdown("### 📍 Find Nearby Gyms")
            region_input = gr.Textbox(label="Your Region or Postcode")
            # Removed the 'disease_input' Gradio component
            budget_input = gr.Textbox(label="Monthly Budget", placeholder="e.g. £30/month")
            submit_btn = gr.Button("Get Gym Recommendations")
            gpt_output = gr.Text(label="Recommended Gyms")
            map_output = gr.HTML(label="Map of Nearby Gyms")

            submit_btn.click(
                fn=gym_finder_pipeline,
                inputs=[region_input, budget_input], # Removed disease_input
                outputs=[gpt_output, map_output]
            )

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

    logout_btn.click(
        fn=lambda: (gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), ""),
        inputs=[],
        outputs=[login_section, signup_section, app_section, login_msg]
    )

# Launch full app
if __name__ == "__main__":
    app.launch(share=True)