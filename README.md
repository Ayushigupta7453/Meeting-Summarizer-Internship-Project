The "Meeting Summarizer and Plan of Action Generator" project aims to automate the summarization of meeting recordings or transcriptions, providing actionable insights and sending summaries via email.

•Technical Stack:
•	Programming Languages: Python
•	Frameworks/Libraries: Streamlit, NLTK, pydub, webvtt, whisper,OpenAI
•	Tools/Platforms: Gmail SMTP (for email sending)

Flow of Data:
1.	Users start by interacting with the Streamlit interface to upload meeting recordings or transcription files.
2.	If the meeting recordings are in OneDrive, they are fetched using the Microsoft Graph API.
3.	The system then converts any video files into audio files using FFmpeg.
4.	The audio files are processed using pydub to ensure they are in the correct format.
5.	The processed audio files are transcribed into text using Whisper.
6.	The transcribed text undergoes preprocessing using NLTK.
7.	The preprocessed text is then summarized, and a plan of action is generated using GPT-3.5 Turbo.
8.	The summarized content and action plan are sent via email to the designated recipients using smtplib.
9.	Configuration settings for email automation and other parameters are managed through a JSON file.

Setup and Configuration
Prerequisites
Before using the application, ensure you have the following:
•	Python 3.7 or higher installed
•	Access to a terminal or command prompt
•	Git installed (if you plan to clone the repository from GitHub)
•	Install the required dependencies: using command pip install requirements.txt
•	Run the application


