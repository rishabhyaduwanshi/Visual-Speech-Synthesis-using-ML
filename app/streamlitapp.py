# Import all of the dependencies
import streamlit as st
import os 
import imageio 

import tensorflow as tf 
import numpy as np
from utils import load_data, num_to_char,load_video
from modelutil import load_model
import cv2

# Set the layout to the streamlit app as wide 
st.set_page_config(layout='wide')

# Setup the sidebar
# with st.sidebar: 
#     st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
#     st.title('LipBuddy')
#     st.info('This application is originally developed from the LipNet deep learning model.')

st.title('Visual Speech Synthesis using ML') 
# Generating a list of options or videos 
options = os.listdir(os.path.join('.', 'data', 's1'))
selected_video = st.selectbox('Select a video', options)

# Generate two columns 
col1, col2 = st.columns(2)

if options: 

    # Rendering the video 
    with col1: 
        st.info('Input Video')
        file_path = os.path.join('.','data','s1', selected_video)
        if selected_video.lower().split(".")[-1] in ["mp4", "avi", "mkv", "mov", "wmv", "flv"]:
            video=open(file_path,'rb')
        else:
            os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')
            # Rendering inside of the app
            video = open('test_video.mp4', 'rb')
        video_bytes = video.read() 
        st.video(video_bytes)


    with col2: 
        st.info('This is all the machine learning model sees when making a prediction')
        file_path = os.path.join('.','data','s1', selected_video)
        if selected_video.lower().split(".")[-1] in ["mp4", "avi", "mkv", "mov", "wmv", "flv"]:

            video=load_video(file_path)
            video=video[:75,...]
            if video.shape[0] < 75:
                # Pad the frames with empty frames (assumed to be black frames)
                empty_frames = np.zeros_like(video[0])  # Assuming frames are grayscale
                num_empty_frames = 75 - video.shape[0]
                video = np.pad(video, [(0, num_empty_frames), (0, 0),  (0, 0), (0, 0)], mode='constant', constant_values=0)

        else:
            video, annotations = load_data(tf.convert_to_tensor(file_path))
        video=np.array(video).squeeze()
        imageio.mimsave('animation.gif', video, fps=10)
        st.image('animation.gif', width=400) 

        st.info('Tokenized output from the Model')
        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)

        # Convert prediction to text
        st.info('Decoded into words')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)
        