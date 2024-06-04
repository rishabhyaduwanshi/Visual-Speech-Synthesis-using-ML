# Import all of the dependencies
import streamlit as st
import os 
import imageio 

import tensorflow as tf 
import numpy as np
from utils import load_data, nums_to_chars,load_video
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
options = os.listdir(os.path.join('data', 's1'))
selected_video = st.selectbox('Select a video', options)

import streamlit as st
# Generate two columns 
col1, col2,col3 = st.columns(3)

if options: 

    # Rendering the video 
    with col1: 
        st.info('Input Video')
        file_path = os.path.join('.','data','s1', selected_video)
        video=None
        print(selected_video.lower().split(".")[-1])
        if selected_video.lower().split(".")[-1] in ["mp4", "avi", "mkv", "mov", "wmv", "flv"]:
            video=open(file_path,'rb')
        elif selected_video.lower().split(".")[-1] == "mpg":
            print("Hiiiiiiiiiiiiiiiiiiii::\n debug \n debug")
            os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')
            # Rendering inside of the app
            video = open('test_video.mp4', 'rb')
        video_bytes = video.read() 
        st.video(video_bytes) 
    st.info('This is all the machine learning model sees when making a prediction')
    # st.info(selected_video.lower().split(".")[-1])
    file_path = os.path.join('.','data','s1', selected_video)
    # if selected_video.lower().split(".")[-1] in ["mp4", "avi", "mkv", "mov", "wmv", "flv"]:
        # video=open(file_path,'rb')
        # cap = cv2.VideoCapture(file_path)
        # video = []
        # while cap.isOpened():
        #     ret, frame = cap.read()
        #     if not ret:
        #         break
        #     video.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
        # video = [cv2.resize(frame, (140, 46)) for frame in video]
        # if video.shape[1] > 75:
        #     video = video[:, :75, ...]

    video=load_video(file_path)
    #adjusting video to 75 frames
    video=video[:75,...]
    if video.shape[0] < 75:
        # Pad the frames with empty frames (assumed to be black frames)
        empty_frames = np.zeros_like(video[0])  # Assuming frames are grayscale
        num_empty_frames = 75 - video.shape[0]
        video = np.pad(video, [(0, num_empty_frames), (0, 0),  (0, 0), (0, 0)], mode='constant', constant_values=0)

        # st.info(tf.shape(video))
        #cap.release()
    # elif selected_video.lower().split(".")[-1] is "mpg":
    #     video, annotations = load_data(tf.convert_to_tensor(file_path))
        # st.info(tf.shape(video))
    video=np.array(video).squeeze()
    # cropped_buffer_uint8 = [np.uint8(image) for image in video]

    imageio.mimsave('animation.gif', video, fps=10)
    st.image('animation.gif', width=400) 

    st.info('Tokenized output from the Model')
    model = load_model()
    # print("Model summary before loading weights:")
    # print(model.summary())
    # checkpoint = tf.train.Checkpoint(model=model)
    # checkpoint.restore(tf.train.latest_checkpoint('models - checkpoint 96.zip'))

    # print("\nModel summary after loading weights from checkpoint:")
    # print(model.summary())
    yhat = model.predict(tf.expand_dims(video, axis=0))
    decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
    st.text(decoder)

    # Convert prediction to text
    st.info('Decoded into words')
    converted_prediction = nums_to_chars(decoder,selected_video)
    st.text(converted_prediction)