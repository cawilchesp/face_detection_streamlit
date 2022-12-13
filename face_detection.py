import streamlit as st
import mediapipe as mp

import cv2
import numpy as np
import tempfile
import time

from PIL import Image

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

@st.cache()
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h,w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = width/float(w)
        dim = (int(w*r),height)
    else:
        r = width/float(w)
        dim = (width,int(h*r))

    resized = cv2.resize(image, dim, interpolation=inter)

    return resized

# Main Page ------------------
st.title('Face Mesh')
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width: 350px
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width: 350px
        margin-left: -350px
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar ------------------
st.sidebar.title('Sidebar')
st.sidebar.subheader('Parameters')
app_mode = st.sidebar.selectbox('Choose the app mode', 
                                    ['About App', 'Run on Image', 'Run on video']
                                )

if app_mode == 'About App':
    # Main Page ------------------
    st.markdown('In this application we are using **mediapipe** for creating a Facemesh app.')

elif app_mode == 'Run on Image':
    # Sidebar ------------------
    st.sidebar.markdown('---')
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
            width: 350px
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
            width: 350px
            margin-left: -350px
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    max_faces = st.sidebar.number_input('Maximum Number of Faces', value=2, min_value=1)
    detection_confidence = st.sidebar.slider('Minimum Detection Confidence', min_value=0.0, max_value=1.0, value=0.5)
    st.sidebar.markdown('---')
    img_file_buffer = st.sidebar.file_uploader('Upload Image', type=['jpg','jpeg','png'])
    
    # Main Page ------------------
    st.markdown('**Detected Faces**')
    kpi1_text = st.markdown('0')

    # Processing ------------------
    if img_file_buffer is not None:
        image = np.array(Image.open(img_file_buffer))
        st.sidebar.text('Original image')
        st.sidebar.image(image)
        face_count = 0
        drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)
        with mp_face_mesh.FaceMesh(static_image_mode = True, max_num_faces = max_faces, 
        min_detection_confidence = detection_confidence) as face_mesh:
            results = face_mesh.process(image)
            out_image = image.copy()
            
            # Face Landmark Drawing
            if results.multi_face_landmarks is not None:
                for face_landmarks in results.multi_face_landmarks:
                    face_count += 1
                    mp_drawing.draw_landmarks(
                    image = out_image,
                    landmark_list = face_landmarks,
                    connections = mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec = drawing_spec)

                    kpi1_text.write(f'{face_count}')

                # Main Page ------------------
                st.subheader('Output Image')
                st.image(out_image, use_column_width = True)


elif app_mode == 'Run on video':
    # Sidebar ------------------
    st.sidebar.markdown('---')
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
            width: 350px
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
            width: 350px
            margin-left: -350px
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    max_faces = st.sidebar.number_input('Maximum Number of Faces', value=2, min_value=1)
    detection_confidence = st.sidebar.slider('Minimum Detection Confidence', min_value=0.0, max_value=1.0, value=0.5)
    tracking_confidence = st.sidebar.slider('Minimum Tracking Confidence', min_value=0.0, max_value=1.0, value=0.5)
    st.sidebar.markdown('---')
    use_webcam = st.sidebar.button('Use Webcam')
    stop_video = st.sidebar.button('Stop Video')
    record = st.sidebar.checkbox('Record Video')
    video_file_buffer = st.sidebar.file_uploader('Upload a Video', type = ['mp4', 'mov', 'avi', 'asf', 'm4v'])

    # Main Page ------------------
    st.markdown('**Output**')
    kpi1, kpi2, kpi3 = st.columns(3)

    with kpi1:
        st.markdown('**Frame Rate**')
        kpi1_text = st.markdown('0')

    with kpi2:
        st.markdown('**Detected Faces**')
        kpi2_text = st.markdown('0')

    with kpi3:
        st.markdown('**Image Width**')
        kpi3_text = st.markdown('0')

    # Processing ------------------
    # if record:
    #     st.checkbox('Recording', value = True)
    tffile = tempfile.NamedTemporaryFile(delete=False)
    vid = cv2.VideoCapture()
    if use_webcam:
        vid.open(0)
    else:
        if video_file_buffer is not None:
            tffile.write(video_file_buffer.read())
            vid.open(tffile.name)

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc('M','J','P','G')
    out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))

    if video_file_buffer is not None:
        st.sidebar.text('Input video')
        st.sidebar.video(tffile.name)

    if stop_video:
            vid.release()

    fps = 0
    i = 0
    drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)
    stframe = st.empty()
    with mp_face_mesh.FaceMesh(max_num_faces = max_faces, 
    min_detection_confidence = detection_confidence,
    min_tracking_confidence = tracking_confidence) as face_mesh:
        prevTime = 0

        while(vid.isOpened()):
            i += 1
            ret, frame = vid.read()
            if not ret:
                continue

            results = face_mesh.process(frame)
            frame.flags.writeable = True

            face_count = 0
            if results.multi_face_landmarks:
                ##Face Landmark Drawing
                for face_landmarks in results.multi_face_landmarks:
                    face_count += 1                    
                    mp_drawing.draw_landmarks(
                    image = frame,
                    landmark_list = face_landmarks,
                    connections = mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec = drawing_spec,
                    connection_drawing_spec = drawing_spec)

            currTime = time.time()
            fps = 1 / (currTime - prevTime)
            prevTime = currTime

            if record:
                out.write(frame)
            
            ##Dashboard
            kpi1_text.write(f'{int(fps)}')
            kpi2_text.write(f'{face_count}')
            kpi3_text.write(f'{width}')

            frame = cv2.resize(frame, (0,0), fx = 0.8, fy = 0.8)
            frame = image_resize(image = frame, width = 200)
            stframe.image(frame, channels = 'BGR', use_column_width = True)
                