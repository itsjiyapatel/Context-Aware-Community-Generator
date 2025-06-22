import os
import pandas as pd
import streamlit as st
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datetime import datetime, timedelta
import random

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
# Load fine-tuned model & tokenizer for community recommendation (Model 1)
model_path1 = "./fitness_t5_model_v2"
tokenizer1 = T5Tokenizer.from_pretrained(model_path1)
model1 = T5ForConditionalGeneration.from_pretrained(model_path1)
model1 = model1.to(device)

# Load fine-tuned model & tokenizer for engagement post generation (Model 2)
model_path2 = "./fine_tuned_t5_fitness"
tokenizer2 = T5Tokenizer.from_pretrained(model_path2)
model2 = T5ForConditionalGeneration.from_pretrained(model_path2)
model2 = model2.to(device)
# Load community dataset
df = pd.read_csv("fitness_communities.csv")

# App Title
st.set_page_config(page_title="Fitness Community Hub", layout="wide")
st.title("🏋️‍♀️ Fitness Community Engagement Platform")

# --- Enhanced User Profile ---
with st.sidebar:
    st.header("👤 Your Profile")
    name = st.text_input("Name", "")
    age = st.number_input("Age", 10, 100, 25)
    fitness_goal = st.selectbox("Primary Goal", df["FitnessGoal"].unique())
    fitness_type = st.selectbox("Preferred Activity", df["FitnessType"].unique())
    experience = st.select_slider("Experience Level", ["Beginner", "Intermediate", "Advanced"])
    risk_score = st.sidebar.slider("Fitness Risk Score", 0, 100, 1)
    tone = st.selectbox("Content Tone", ["Motivational", "Supportive", "Educational", "Challenging"])
    
    # Engagement preferences
    st.header("🔔 Notifications")
    notify_posts = st.checkbox("New community posts", True)
    notify_challenges = st.checkbox("Weekly challenges", True)

# --- Community Engagement Features ---
tab1, tab2, tab3 = st.tabs(["🏠 Home", "💬 Community", "📊 Progress"])
# # Function to generate engagement post
# def generate_engagement_post(community, goal, ftype, risk, sentiment="motivational"):
#     input_text = f"Community: {community} | User following {ftype} program aiming for {goal} with a fitness risk score of {risk}. Generate a {sentiment} post."
#     inputs = tokenizer2(input_text, return_tensors="pt", truncation=True, padding=True).to(device)
#     outputs = model2.generate(inputs["input_ids"], max_length=150, num_beams=4, early_stopping=True)
#     summary = tokenizer2.decode(outputs[0], skip_special_tokens=True)
#     return summary
# --- Helper Functions ---
def generate_engagement_post(community, goal, ftype, risk, sentiment="motivational"):
    input_text = f"Community: {community} | User following {ftype} program aiming for {goal}. Generate a {sentiment} post."
    inputs = tokenizer2(input_text, return_tensors="pt", truncation=True).to(device)
    outputs = model2.generate(inputs["input_ids"], max_length=150)
    return tokenizer2.decode(outputs[0], skip_special_tokens=True)

def log_engagement(data):
    """Save engagement data to CSV"""
    data["timestamp"] = datetime.now().isoformat()
    df = pd.DataFrame([data])
    df.to_csv("engagement_log.csv", mode="a", 
             header=not os.path.exists("engagement_log.csv"), 
             index=False)

def display_community_feed(community):
    """Show recent community activity"""
    if os.path.exists("engagement_log.csv"):
        df = pd.read_csv("engagement_log.csv")
        community_posts = df[df["community"] == community].sort_values("timestamp", ascending=False).head(5)
        
        for _, post in community_posts.iterrows():
            with st.container(border=True):
                st.caption(f"{post['user']} • {post['timestamp']}")
                st.write(f"**{post['type']}**: {post['content']}")
                st.button("👍", key=f"like_{post['timestamp']}")
    else:
        st.info("Be the first to post in this community!")

def display_user_activity(user):
    """Show user's recent activity"""
    if os.path.exists("engagement_log.csv"):
        df = pd.read_csv("engagement_log.csv")
        user_activity = df[df["user"] == user].sort_values("timestamp", ascending=False).head(5)
        
        for _, activity in user_activity.iterrows():
            st.caption(f"{activity['timestamp']} - {activity['type']}")
            st.write(activity['content'])
    else:
        st.info("You haven't been active yet")

def display_badges(user):
    """Show user's earned badges"""
    # This would normally come from a database
    badges = {
        "First Post": True,
        "Week Active": True,
        "Challenge Completed": False,
        "Community Leader": False
    }
    
    cols = st.columns(4)
    for i, (badge, earned) in enumerate(badges.items()):
        cols[i%4].metric(
            badge, 
            "✅" if earned else "❌", 
            help="Earned" if earned else "Not yet earned"
        )

with tab1:
    # Community Discovery
    st.header("🔍 Find Your Communities")
    # Generate top 3 community recommendations
    if st.button("🔍 Generate Top 3 Communities"):
        filtered_df = df[(df["FitnessGoal"] == fitness_goal) & (df["FitnessType"] == fitness_type)]
        if len(filtered_df) < 3:
            st.warning("Not enough communities match your criteria. Showing random picks.")
            selected_comms = random.sample(df["CommunityName"].tolist(), 3)
        else:
            selected_comms = random.sample(filtered_df["CommunityName"].tolist(), 3)

        st.session_state["recommended"] = selected_comms

        st.subheader("🏆 Recommended Communities:")
        for comm in selected_comms:
            post = generate_engagement_post(comm, fitness_goal, fitness_type, risk_score, tone)
            st.write(f"**{comm}**")
            st.write(post)

# User joins a community
    if "recommended" in st.session_state:
        st.subheader("🎯 Join a Community")
        joined_community = st.selectbox("Choose a Community to Join", st.session_state["recommended"], key="join_select")
        if st.button("✅ Join Selected Community"):
            st.success(f"You have successfully joined **{joined_community}**!")
            st.session_state["joined"] = joined_community

# Tab 2: Community Engagement (Enhanced)
with tab2:
    if "joined" in st.session_state:
        community = st.session_state["joined"]
        st.header(f"👥 {community}")
        
        # Engagement metrics quick view
        days_active = 7  # Would normally come from database
        st.metric("Days Active", f"{days_active} days", help="Your participation streak")
        
        # Content creation
        with st.expander("✨ Create New Content", expanded=True):
            content_type = st.selectbox(
                "What would you like to create?",
                ["Discussion Post", "Progress Update", "Question", "Challenge Submission"],
                key="content_type"
            )
            
            if st.button("Generate Content Idea"):
                prompt = (
                    f"Suggest a {content_type.lower()} topic for {community} community "
                    f"about {fitness_goal} through {fitness_type} "
                    f"with {tone.lower()} tone"
                )
                inputs = tokenizer2(prompt, return_tensors="pt").to(device)
                outputs = model2.generate(inputs["input_ids"], max_length=100)
                st.session_state.content_idea = tokenizer2.decode(outputs[0], skip_special_tokens=True)
            
            if "content_idea" in st.session_state:
                st.text_area("Your Content", st.session_state.content_idea, height=150)
                
                if st.button("Post to Community"):
                    # Log engagement
                    log_engagement({
                        "type": "post",
                        "content": st.session_state.content_idea,
                        "community": community,
                        "user": name
                    })
                    st.success("Posted successfully!")
                    del st.session_state.content_idea
                    st.rerun()
        
        # Community discussion feed
        st.subheader("🔥 Trending Discussions")
        display_community_feed(community)
        
        # Weekly Challenge
        with st.expander("🏅 Current Challenge", expanded=True):
            if st.button("Generate Challenge"):
                prompt = f"Create a weekly challenge for {community} about {fitness_goal}"
                inputs = tokenizer2(prompt, return_tensors="pt").to(device)
                outputs = model2.generate(inputs["input_ids"], max_length=150)
                st.session_state.current_challenge = tokenizer2.decode(outputs[0], skip_special_tokens=True)
            
            if "current_challenge" in st.session_state:
                st.write(st.session_state.current_challenge)
                if st.button("Accept Challenge"):
                    log_engagement({
                        "type": "challenge",
                        "content": st.session_state.current_challenge,
                        "community": community,
                        "user": name
                    })
                    st.success("Challenge accepted!")
    else:
        st.info("Join a community to participate")

# Tab 3: Progress Tracking
with tab3:
    if "joined" in st.session_state:
        st.header("📈 Your Progress")
        
        # Engagement stats
        col1, col2, col3 = st.columns(3)
        col1.metric("Posts", "15", "3 this week")
        col2.metric("Reactions", "42", "8 this week")
        col3.metric("Streak", "7 days", "Keep going!")
        
        # Activity history
        st.subheader("Recent Activity")
        display_user_activity(name)
        
        # Achievement badges
        st.subheader("🏆 Your Badges")
        display_badges(name)
    else:
        st.info("Join a community to track your progress")

