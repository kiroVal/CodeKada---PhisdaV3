import os, io, uuid, requests
from fastapi import FastAPI, Form, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from twilio.twiml.voice_response import VoiceResponse, Record
import firebase_admin
from firebase_admin import credentials, storage, firestore
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()




# ---- Firebase init ----
if not firebase_admin._apps:
    cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred, {
        "storageBucket": os.getenv("FIREBASE_BUCKET")
    })
db = firestore.client()
bucket = storage.bucket()

# ---- Azure Speech helpers ----
import azure.cognitiveservices.speech as speechsdk

SPEECH_KEY = os.getenv("SPEECH_KEY")
SPEECH_REGION = os.getenv("SPEECH_REGION")
AZURE_TTS_VOICE = os.getenv("AZURE_TTS_VOICE", "en-US-JennyNeural")

def azure_transcribe_from_url(audio_url: str) -> str:
    # Download audio to memory
    wav = requests.get(audio_url, timeout=30).content
    # Save temp
    tmp = f"/tmp/{uuid.uuid4()}.wav"
    with open(tmp, "wb") as f:
        f.write(wav)

    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    audio_config = speechsdk.AudioConfig(filename=tmp)
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    result = recognizer.recognize_once_async().get()
    os.remove(tmp)

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return result.text
    else:
        return ""

def azure_tts_to_mp3(text: str) -> bytes:
    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    speech_config.speech_synthesis_voice_name = AZURE_TTS_VOICE
    audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=False, filename=None)  # we'll capture stream

    # Use a stream to get bytes
    stream = speechsdk.audio.PushAudioOutputStream()
    audio_config = speechsdk.audio.AudioOutputConfig(stream=stream)
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

    # Start synthesis
    done = []

    def on_audio_chunk(evt):
        pass  # not strictly needed

    def on_synthesis_completed(evt):
        done.append(True)

    stream_writer = speechsdk.audio.PullAudioOutputStream()  # not used; alt approach below

    # Easier approach: synthesize to result and read audio data
    # (Azure SDK returns WAV by default; to force MP3:)
    speech_config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3
    )
    result = synthesizer.speak_text_async(text).get()
    return result.audio_data

def upload_bytes_to_firebase(data: bytes, path: str, content_type: str) -> str:
    blob = bucket.blob(path)
    blob.upload_from_string(data, content_type=content_type)
    blob.make_public()
    return blob.public_url

# ---- LangChain (very simple chain) ----
# You can swap to Azure OpenAI if you prefer.
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

SYSTEM_PROMPT = """You are a helpful legal information assistant for Philippine and international general legal topics.
You are NOT a lawyer and do not provide legal advice—only general information.
Answer concisely, cite general legal principles, and suggest speaking to a licensed attorney for specific cases."""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}")
])

def lawyer_bot_answer(question: str) -> str:
    chain = prompt | llm
    resp = chain.invoke({"question": question})
    return resp.content.strip()

# ---- FastAPI ----
app = FastAPI()

@app.post("/twilio/voice", response_class=PlainTextResponse)
async def twilio_voice():
    """Initial webhook when call starts: greet and record a question."""
    vr = VoiceResponse()
    vr.say("Hello. You're connected to the legal information assistant. Please ask your question after the beep. Then pause.")
    vr.record(
        action="/twilio/process_recording",
        method="POST",
        max_length=30,
        play_beep=True,
        trim="trim-silence",
        timeout=3
    )
    vr.say("I didn't receive a recording. Goodbye.")
    return str(vr)

@app.post("/twilio/process_recording", response_class=PlainTextResponse)
async def process_recording(
    RecordingUrl: str = Form(...),
    CallSid: str = Form(...),
    From: str = Form(None),
    To: str = Form(None),
):
    # 1) STT
    transcript = azure_transcribe_from_url(RecordingUrl + ".wav")  # Twilio provides .wav extension

    # 2) LLM
    if transcript.strip() == "":
        answer_text = "Sorry, I could not understand the audio. Please try again or consult a licensed attorney."
    else:
        answer_text = lawyer_bot_answer(transcript)

    # 3) TTS
    audio_bytes = azure_tts_to_mp3(answer_text)

    # 4) Store in Firebase
    now = datetime.now(timezone.utc).isoformat()
    call_doc = {
        "call_sid": CallSid,
        "from": From,
        "to": To,
        "question_transcript": transcript,
        "answer_text": answer_text,
        "created_at": now,
        "type": "qa"
    }
    ref = db.collection("calls").document(CallSid).collection("turns").document(str(uuid.uuid4()))
    ref.set(call_doc)

    # Upload audio
    audio_path = f"calls/{CallSid}/{uuid.uuid4()}.mp3"
    public_url = upload_bytes_to_firebase(audio_bytes, audio_path, "audio/mpeg")

    # 5) Respond with TwiML to play the answer
    vr = VoiceResponse()
    vr.play(public_url)
    vr.say("You may ask one more short question after the beep.")
    vr.record(
        action="/twilio/process_recording",
        method="POST",
        max_length=30,
        play_beep=True,
        trim="trim-silence",
        timeout=3
    )
    vr.say("Thank you for calling. Goodbye.")
    return str(vr)
