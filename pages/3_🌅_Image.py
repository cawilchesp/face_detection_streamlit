import streamlit as st
import mediapipe as mp

import cv2
import numpy as np

# For debugging
from icecream import ic


st.set_page_config(layout="wide", page_icon="🌅")

st.title('Image Detection')

# SIDEBAR
st.sidebar.header("Model")

st.cache_resource.clear()
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
source_image = st.file_uploader(
    label="Choose an image...",
    type=("jpg", "jpeg", "png", 'bmp')
)
col_image, col_info_1, col_info_2 = st.columns([5, 1, 1])
with col_image:
    st_frame = st.empty()
with col_info_1:
    st.markdown('**Width**')
    st.markdown('**Height**')
    st.markdown('**Face Count**')
with col_info_2:
    width_text = st.markdown('0 px')
    height_text = st.markdown('0 px')
    face_count_text = st.markdown('0')

if source_image:
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh
    drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=max_faces,
        min_detection_confidence=detection_confidence )
    if st.button("Process Image"):
        with st.spinner("Running..."):
            try:
                tfile = np.asarray(bytearray(source_image.read()), dtype=np.uint8)
                image = cv2.imdecode(tfile, 1)
                height, width, channels = image.shape

                width_text.write(str(width))
                height_text.write(str(height))

                # Process image face detection
                results = face_mesh.process(image)

                # Drawing face mesh
                annotated_image = image.copy()
                face_count = 0
                if results.multi_face_landmarks:
                    # Face Landmark Drawing
                    for face_landmarks in results.multi_face_landmarks:
                        face_count += 1                    
                        mp_drawing.draw_landmarks(
                            image = annotated_image,
                            landmark_list = face_landmarks,
                            connections = mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec = drawing_spec,
                            connection_drawing_spec = drawing_spec )
                
                face_count_text.write(str(face_count))

                st_frame.image(
                    annotated_image,
                    caption='Detected Image',
                    channels="BGR",
                    use_column_width=True
                )
            except Exception as e:
                st.error(f"Error loading image: {e}")
