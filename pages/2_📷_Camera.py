import streamlit as st
import mediapipe as mp

import cv2

# For debugging
from icecream import ic


st.set_page_config(layout="wide", page_icon="📷")

st.title('Webcam Detection')

# SIDEBAR
st.sidebar.header("Model")

max_faces = st.sidebar.number_input(
    label='Maximum Number of Faces',
    value=2,
    min_value=1 )

detection_confidence = st.sidebar.slider(
    label='Minimum Detection Confidence',
    min_value=0.0,
    max_value=1.0,
    value=0.5 )

tracking_confidence = st.sidebar.slider(
    label='Minimum Tracking Confidence',
    min_value=0.0,
    max_value=1.0,
    value=0.5 )

# image/video options
col_image, col_info_1, col_info_2 = st.columns([5, 1, 1])
with col_image:
    st_frame = st.empty()
with col_info_1:
    st.markdown('**Width**')
    st.markdown('**Height**')
    st.markdown('**Frame Rate**')
    st.markdown('**Frame**')
    st.markdown('**Face Count**')
with col_info_2:
    width_text = st.markdown('0 px')
    height_text = st.markdown('0 px')
    fps_text = st.markdown('0 FPS')
    frame_text = st.markdown('0')
    face_count_text = st.markdown('0')

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces = max_faces, 
    min_detection_confidence = detection_confidence,
    min_tracking_confidence = tracking_confidence )

col_play, col_stop, col3 = st.columns([1, 1, 5])
with col_play:
    play_button = st.button(label="Play Webcam")
with col_stop:
    stop_button = st.button(label="Stop Webcam")

play_flag = False
if play_button: play_flag = True
if stop_button: play_flag = False
with st.spinner("Running..."):
    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    width_text.write(f"{width} px")
    height_text.write(f"{height} px")
    fps_text.write(f"{fps:.2f} FPS")

    with face_mesh:
        frame_number = 0
        while play_flag:
            success, image = cap.read()
            if not success:break
            frame_text.write(str(frame_number))

            # Resize the image to a standard size
            image = cv2.resize(image, (720, int(720 * (9 / 16))))

            # Process image face detection
            results = face_mesh.process(image)

            # Drawing face mesh
            annotated_image = image.copy()
            face_count = 0
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    face_count += 1                    
                    mp_drawing.draw_landmarks(
                    image = annotated_image,
                    landmark_list = face_landmarks,
                    connections = mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec = drawing_spec,
                    connection_drawing_spec = drawing_spec )

            face_count_text.write(str(face_count))

            # Show image results
            st_frame.image(
                annotated_image,
                caption='Detected Video',
                channels="BGR",
                use_column_width=True
            )
            frame_number += 1
    cap.release()
    