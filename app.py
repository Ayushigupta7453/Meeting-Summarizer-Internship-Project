import os
import whisper
import subprocess
import streamlit as st
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from dotenv import load_dotenv # type: ignore
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from langchain_openai import ChatOpenAI # type: ignore
from langchain.schema import SystemMessage, HumanMessage
from concurrent.futures import ThreadPoolExecutor
import hashlib
from docx import Document # type: ignore
import webvtt # type: ignore
from pydub import AudioSegment
# Load environment variables
load_dotenv()
open_api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = open_api_key

# Initialize LangChain OpenAI model
llm = ChatOpenAI(model_name='gpt-3.5-turbo')
AudioSegment.ffmpeg = "C:/Ffmpeg/ffmpeg.exe"

# Streamlit App
st.title("Meeting Summarizer and Plan of Action Generator")
st.write("Upload a video file, or a transcription file directly in .docx or .vtt format, and this app will summarize the meeting and send the summary via email.")

# Add custom CSS for better visual presentation
st.markdown("""
    <style>
        .summary-box {
            padding: 20px;
            border: 2px solid #007acc;
            border-radius: 10px;
            background-color: #f0f8ff;
            margin-bottom: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .summary-box h2 {
            color: #007acc;
        }
        .summary-box p {
            margin-bottom: 10px;
            font-size: 1.1em;
        }
        .section-heading {
            font-size: 1.5em;
            margin-top: 20px;
            margin-bottom: 10px;
            color: #007acc;
        }
        .email-section {
            margin-top: 40px;
            color: #007acc;
        }
        .stButton>button {
            background-color: #007acc;
            color: blue;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
        }
        .stButton>button:hover {
            background-color: #005b99;
        }
    </style>
""", unsafe_allow_html=True)

# Function to hash the content for caching purposes
def get_hash(content):
    return hashlib.md5(content.encode()).hexdigest()

def extract_audio_from_video(video_file):
    video_file_path = 'video.mp4'
    audio_file_path = 'output_audio.mp3'
    
    # Save the uploaded video file to disk
    with open(video_file_path, 'wb') as f:
        f.write(video_file.read())
    
    # Extract audio from the video file using pydub
    video = AudioSegment.from_file(video_file_path)
    audio = video.set_frame_rate(16000).set_channels(1)
    
    # Export the audio to a file
    audio.export(audio_file_path, format='mp3', codec='libmp3lame')
    
    return audio_file_path
# Transcribe and preprocess audio using whisper
def transcribe_and_preprocess(audio_file):
    model = whisper.load_model("base")
    transcribed_text = model.transcribe(audio_file)["text"]
    print(f"Transcribed text: {transcribed_text[:500]}...")  # Print first 500 characters
    tokens = word_tokenize(transcribed_text.lower().translate(str.maketrans('', '', string.punctuation)))
    stop_words = set(stopwords.words('english'))
    preprocessed_text = ' '.join([token for token in tokens if token not in stop_words])
    print(f"Preprocessed text: {preprocessed_text[:500]}...")  # Print first 500 characters
    return transcribed_text, preprocessed_text

# Process .docx file
def process_docx(file):
    doc = Document(file)
    full_text = [para.text for para in doc.paragraphs]
    return '\n'.join(full_text)

# Process .vtt file
def process_vtt(file):
    vtt = webvtt.read_buffer(file)
    full_text = [caption.text for caption in vtt]
    return '\n'.join(full_text)

# Generate summary using the language model
def generate_summary(transcript):
    response = llm.invoke([
        SystemMessage(content='You are an expert assistant that can summarize meetings.'),
        HumanMessage(content=f'Please provide a detailed summary of the following Online Meet recording in a paragraph:\n TEXT: {transcript}\nInclude key decisions, action items, and any notable insights discussed.')
    ])
    return response.content

# Generate additional details for the email
def generate_additional_details(summary):
    responses = {}
    with ThreadPoolExecutor() as executor:
        futures = {
            'category': executor.submit(lambda: llm.invoke([
                SystemMessage(content=f'Here is the detailed summary of the meeting: {summary}'),
                HumanMessage(content='Type of meeting (e.g., Sales pitch, Team meeting, Project update, Client meeting, etc.). Choose one from the options provided in a word.')
            ]).content),

            'emotion': executor.submit(lambda: llm.invoke([
                SystemMessage(content=f'Here is the detailed summary of the meeting: {summary}'),
                HumanMessage(content='Emotion or tone conveyed in this meeting? (e.g., Professional, enthusiastic, urgent, persuasive, etc.). Choose one from the options provided in a word.')
            ]).content),

            'industry': executor.submit(lambda: llm.invoke([
                SystemMessage(content=f'Here is the detailed summary of the meeting: {summary}'),
                HumanMessage(content='Industry related to this meeting? (e.g., Technology, healthcare, finance, etc.). Choose one from the options provided in a word.')
            ]).content),

            'focus': executor.submit(lambda: llm.invoke([
                SystemMessage(content=f'Here is the detailed summary of the meeting: {summary}'),
                HumanMessage(content='Focus of this meeting? (e.g., Introducing a new product, discussing performance, setting goals, etc.). Choose one from the options provided in a word.')
            ]).content),

            'plan': executor.submit(lambda: llm.invoke([
                SystemMessage(content=f'Here is the detailed summary of the meeting: {summary}\nBased on this summary, what is the proposed plan of action?'),
                HumanMessage(content='Please outline a brief plan of action based on the topics discussed. Limit the plan to no more than five points, with each point consisting of no more than 1 or 2 lines.')
            ]).content)
        }
        for key, future in futures.items():
            responses[key] = future.result()
    return responses

# Email sending function
def send_email(subject, body, recipient_list):
    msg = MIMEMultipart()
    msg['From'] = "ayushi7453gupta@gmail.com"
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'html'))

    password = os.getenv("EMAIL_PASSWORD")

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(msg['From'], password)
            msg['Bcc'] = ', '.join(recipient_list)
            server.sendmail(msg['From'], recipient_list, msg.as_string())
            st.success("Email sent to all recipients successfully.")
    except Exception as e:
        st.error(f"SMTP Error: {e}")

# Streamlit File Upload and Processing
file_option = st.radio(
    "Choose the type of file to upload",
    ('Video File', 'Transcription File (.docx)', 'Transcription File (.vtt)')
)

if "show_summary" not in st.session_state:
    st.session_state["show_summary"] = False

uploaded_file = st.file_uploader("Upload File")

if uploaded_file is not None:
    st.success(f"{file_option} uploaded successfully!")

    if st.button("Summarize and Show Summary"):
        with st.spinner('Processing...'):
            if file_option == 'Video File':
                audio_file_path = extract_audio_from_video(uploaded_file)
                transcribed_text, preprocessed_text = transcribe_and_preprocess(audio_file_path)
            elif file_option == 'Transcription File (.docx)':
                transcribed_text = process_docx(uploaded_file)
                preprocessed_text = transcribed_text
            elif file_option == 'Transcription File (.vtt)':
                transcribed_text = process_vtt(uploaded_file)
                preprocessed_text = transcribed_text

            generated_summary = generate_summary(preprocessed_text)
            st.session_state["generated_summary"] = generated_summary
            st.session_state["generated_transcribed_text"] = transcribed_text
            st.session_state["generated_preprocessed_text"] = preprocessed_text
            st.session_state["show_summary"] = True

            additional_details = generate_additional_details(generated_summary)
            st.session_state.update(additional_details)
            st.session_state["generated_plan"] = additional_details['plan']

# Display the stored summary and plan if they exist
if st.session_state.get("show_summary"):
    st.markdown(f"<div class='section-heading'>Meeting Summary:</div><div class='summary-box'>{st.session_state['generated_summary']}</div>", unsafe_allow_html=True)
    formatted_plan = st.session_state["generated_plan"].replace('\n', '<br>')
    st.markdown(f"<div class='section-heading'>Plan of Action:</div><div class='summary-box'>{formatted_plan}</div>", unsafe_allow_html=True)

# Step 2: Send Email
if "generated_summary" in st.session_state:
    st.markdown("<div class='section-heading email-section'>Send Email with Meeting Summary and Plan of Action</div>", unsafe_allow_html=True)

    recipient_emails = st.text_input("Enter recipient email addresses (comma-separated):")

    send_mail_button = st.button("Send Email", key="send_email_button")

    if send_mail_button:
        email_body = (
            f"<b>Type of meeting:</b> {st.session_state.get('category', 'N/A')}<br><br>"
            f"<b>Emotion or tone:</b> {st.session_state.get('emotion', 'N/A')}<br><br>"
            f"<b>Specific industry or topic:</b> {st.session_state.get('industry', 'N/A')}<br><br>"
            f"<b>Any particular focus:</b> {st.session_state.get('focus', 'N/A')}<br><br>"
            "<b>Meeting Summary:</b><br>" + st.session_state["generated_summary"].replace('\n', '<br>') + "<br><br>"
            "<b>Plan of Action:</b><br>" + formatted_plan + "<br><br>"
        )

        recipient_list = [email.strip() for email in recipient_emails.split(",") if email.strip()]
        if recipient_list:
            send_email("Meeting Summary and Plan of Action", email_body, recipient_list)
        else:
            st.error("Please enter at least one recipient email address.")