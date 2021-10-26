import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import time
from PIL import Image

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

DEMO_IMAGE = 'demo.jpg'
DEMO_VIDEO = 'demo.mp4'

st.title('Face Mesh App using MediaPipe')

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

st.sidebar.title("FaceMesh Sidebar")
st.sidebar.subheader("parameters")


@st.cache()
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = width/float(w)
        dim = (int(w*r), height)

    else:
        r = width/float(w)
        dim = (width, int(h*r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    return resized


app_mode = st.sidebar.selectbox(
    'Choose the App mode',
    ['About App', 'Run on Image', 'Run on Video']
)

if app_mode == 'About App':
    st.markdown(
        '''In this Application, we are using **MediaPipe**
        for creating a FaceMesh App.'''
        )
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

    st.video('https://youtu.be/vOdk1nRm_Es')

elif app_mode == 'Run on Image':
    drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)
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

    st.markdown('***Detected Faces***')
    kpi1_text = st.markdown('0')

    max_faces = st.sidebar.number_input(
        'Maximum Number of Face', value=2, min_value=1
        )
    st.sidebar.markdown('---')
    detection_confidence = st.sidebar.slider(
        'Min Detection Confidence', min_value=0.0, max_value=1.0, value=0.5
        )
    st.sidebar.markdown('---')

    img_file_buffer = st.sidebar.file_uploader(
        "Upload an Image", type=["jpg", "jpeg", "png"]
        )
    if img_file_buffer is not None:
        image = np.array(Image.open(img_file_buffer))

    else:
        demo_image = DEMO_IMAGE
        image = np.array(Image.open(demo_image))

    st.sidebar.text('Original Image')
    st.sidebar.image(image)
    face_count = 0

    # Dashboard
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=max_faces,
        min_detection_confidence=detection_confidence
    ) as face_mesh:

        results = face_mesh.process(image)
        out_image = image.copy()

        # Face Landmark Drawing
        for face_landmarks in results.multi_face_landmarks:
            face_count += 1

            mp_drawing.draw_landmarks(
                image=out_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec
            )

            kpi1_text.write(
                f"<h1 style='text-align:center; color:red;'>{face_count}</h1>",
                unsafe_allow_html=True
                )
            st.subheader('Output Image')
            st.image(out_image, use_column_width=True)

elif app_mode == 'Run on Video':

    st.set_option('deprication.showfileUploaderEncoding', False)

    use_webcam = st.sidebar.button('Use Webcam')
    record = st.sidebar.checkbox('Record Video')

    if record:
        st.checkbox('Recording', value=True)

    drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)
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

    max_faces = st.sidebar.number_input(
        'Maximum Number of Face', value=10, min_value=1
        )
    st.sidebar.markdown('---')
    detection_confidence = st.sidebar.slider(
        'Min Detection Confidence', min_value=0.0, max_value=1.0, value=0.5
        )
    tracking_confidence = st.sidebar.slider(
        'Min Tracking Confidence', min_value=0.0, max_value=1.0, value=0.5
        )
    st.sidebar.markdown('---')

    st.markdown('## Output')

    stframe = st.empty
    video_file_buffer = st.sidebar.file_uploader("Upload a Video", type=['mp4', 'mov', 'avi', 'asf', 'm4v'])
    tffile = tempfile.NamedTemporaryFile(delete=False)

    if not video_file_buffer:
        if use_webcam:
            vid = cv2.VideoCapture(0)
        else:
            vid = cv2.VideoCapture(DEMO_VIDEO)
            tffile.name = DEMO_VIDEO

    else:
        tffile.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tffile.name)

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(vid.get(cv2.CAP_PROP_FPS))

    # Recording Part
    codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.Video('output1.mp4', codec, fps_input, (width, height))

    # Dashboard
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=max_faces,
        min_detection_confidence=detection_confidence
    ) as face_mesh:

        results = face_mesh.process(image)
        out_image = image.copy()

        # Face Landmark Drawing
        for face_landmarks in results.multi_face_landmarks:
            face_count += 1

            mp_drawing.draw_landmarks(
                image=out_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec
            )

            kpi1_text.write(
                f"<h1 style='text-align:center; color:red;'>{face_count}</h1>",
                unsafe_allow_html=True
                )
            st.subheader('Output Image')
            st.image(out_image, use_column_width=True)
