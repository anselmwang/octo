import streamlit as st
import tensorflow_datasets as tfds
import numpy as np
from PIL import Image

# Load the dataset
ds = tfds.load("droid_100", data_dir="gs://gresearch/robotics", split="train")
episode = next(iter(ds.shuffle(10, seed=0).take(1)))

# Streamlit App
st.title("DROID Episode Data Explorer")

# Episode Metadata
st.header("Episode Metadata")
st.write(f"Recording Folder Path: {episode['episode_metadata']['recording_folderpath'].numpy().decode('utf-8')}")
st.write(f"File Path: {episode['episode_metadata']['file_path'].numpy().decode('utf-8')}")

# Steps Navigation
st.header("Steps Navigation")
steps = list(episode['steps'])
num_steps = len(steps)
selected_step = st.slider("Select Step", 0, num_steps - 1, 0)

# Step Details
st.subheader(f"Step {selected_step} Details")
st.write(f"Is First: {steps[selected_step]['is_first'].numpy()}")
st.write(f"Is Last: {steps[selected_step]['is_last'].numpy()}")
st.write(f"Is Terminal: {steps[selected_step]['is_terminal'].numpy()}")
st.write(f"Discount: {steps[selected_step]['discount'].numpy()}")
st.write(f"Reward: {steps[selected_step]['reward'].numpy()}")

# Language Instructions
st.subheader("Language Instructions")
st.write(f"Primary: {steps[selected_step]['language_instruction'].numpy().decode('utf-8')}")
st.write(f"Alternative 1: {steps[selected_step]['language_instruction_2'].numpy().decode('utf-8')}")
st.write(f"Alternative 2: {steps[selected_step]['language_instruction_3'].numpy().decode('utf-8')}")

# Observations
st.subheader("Observations")
# Gripper, Cartesian, and Joint Positions
st.write("Gripper Position:", steps[selected_step]['observation']['gripper_position'].numpy())
st.write("Cartesian Position:", steps[selected_step]['observation']['cartesian_position'].numpy())
st.write("Joint Position:", steps[selected_step]['observation']['joint_position'].numpy())

# Images
def display_image(image_data, title):
    image = Image.fromarray(image_data.numpy())
    st.image(image, caption=title)

display_image(steps[selected_step]['observation']['wrist_image_left'], "Wrist Image Left")
display_image(steps[selected_step]['observation']['exterior_image_1_left'], "Exterior Image 1 Left")
display_image(steps[selected_step]['observation']['exterior_image_2_left'], "Exterior Image 2 Left")

# Actions
st.subheader("Actions")
st.write("Gripper Position:", steps[selected_step]['action_dict']['gripper_position'].numpy())
st.write("Gripper Velocity:", steps[selected_step]['action_dict']['gripper_velocity'].numpy())
st.write("Cartesian Position:", steps[selected_step]['action_dict']['cartesian_position'].numpy())
st.write("Cartesian Velocity:", steps[selected_step]['action_dict']['cartesian_velocity'].numpy())
st.write("Joint Position:", steps[selected_step]['action_dict']['joint_position'].numpy())
st.write("Joint Velocity:", steps[selected_step]['action_dict']['joint_velocity'].numpy())
st.write("Action:", steps[selected_step]['action'].numpy())

# Run the app with `streamlit run <script_name>.py`
