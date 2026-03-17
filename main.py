from dotenv import load_dotenv
from pydub import AudioSegment
from openai import OpenAI
import requests
import os
import json
import schedule
import time
import datetime
import numbers
from pyannote.audio import Pipeline
# Suppress non-critical warnings from Pytorch
import warnings
warnings.filterwarnings("ignore")

#Load variables from .env
load_dotenv()

workFolder = 'work'
inFolder = 'in'

# Class for pyannote hook
class UpdateProgress:
    def __init__(self, jobId):
        self.jobId = jobId
    def __call__(self, step_name, step_artifact, file = None, total = None, completed = None):
        if(step_name=="embeddings" and isinstance(total, int)):
            subtotal = total+1
            percentage = int(round(completed / subtotal * 100,0))
            updateJob(self.jobId, 1, '',percentage)
            print(f"Progress for {file}: {int(round(completed / subtotal * 100,0))}%")
    
# Our class for diarizations
class Diarization:
    def __init__(self, speaker, text, start, end):
        self.speaker = speaker
        self.text = text
        self.start = start
        self.end = end

# Make our directories if they don't already exist
if not os.path.exists(workFolder):
    os.makedirs(workFolder)

if not os.path.exists(inFolder):
    os.makedirs(inFolder)

print(f"[{datetime.datetime.now()}] Starting Python speech")
client = OpenAI()

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=os.getenv('TOKEN'))

# Function to start diarization once file is downloaded from s3
def startDiarization(file, userId, jobId):
    diarizationTranscriptions = []
    hook = UpdateProgress(jobId)
    diarization = pipeline(file, hook=hook)
    for turn, _,speaker in diarization.itertracks(yield_label=True):
        # Use Pydub to split the audio from speaker start and end
        clipStart = int(turn.start * 1000)
        clipEnd = int(turn.end * 1000)
        audioFile = AudioSegment.from_file(file)

        # Create a temporary name for our spliced audio
        clipFileName = f"{workFolder}\\{clipStart}_{clipEnd}.mp3"
        audioPartition = audioFile[ clipStart : clipEnd]
        try:
            clipToTranscribe = audioPartition.export(clipFileName, format="mp3")

            # Transcribe the sliced audio with Open AI whisper
            transcription = client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=clipToTranscribe
            )

            # Add the transcription and diarizations to our array ready to send to database table
            diarizationTranscriptions.append(Diarization(speaker=speaker, text=transcription.text, start=clipStart, end=clipEnd))
            clipToTranscribe.close()
        except Exception:
            print(f"[{datetime.datetime.now()}] {Exception}") 
    return json.dumps({"userId":userId,"jobId":jobId,"diarizations":[Diarization.__dict__ for Diarization in diarizationTranscriptions]})

# Function to check for new messages from AWS SQS
def checkMessages():
    print(f"[{datetime.datetime.now()}] Checking for new messages..")
    session = requests.Session()
    res = session.get(os.getenv('SQS_API')+'/receive', headers={'Content-Type': 'application/json'})
    session.close()

    # If no new messages then do not continue
    if len(json.loads(res.text))==0:
        print(f"[{datetime.datetime.now()}] No new messages available")
        return
    try:
        jsonRes = json.loads(res.text)
        jsonBody = json.loads(jsonRes[0]['Body'])
        receiptHandle = jsonRes[0]['ReceiptHandle']
        messageId = jsonRes[0]['MessageId']
        if not jsonBody['key']:
            print(f"[{datetime.datetime.now()}] No S3 key found")
            updateJob(jobId, -1, None)
            return
        key = jsonBody['key']
        userId = jsonBody['userId']
        jobId = jsonBody['jobId']
        session = requests.Session()

        # Download file from S3
        print(key)
        s3Res = session.get(os.getenv('S3_API')+'/retrieve', headers={'Content-Type': 'application/json','Key': key})
        print(f"[{datetime.datetime.now()}] {s3Res.text}")
        if not s3Res.text:
            print(f"[{datetime.datetime.now()}] Unable to retrieve file URL from S3 using key {key}")
            updateJob(jobId, -1, None)
            deleteMessage(receiptHandle, messageId)
            return
        print(f"[{datetime.datetime.now()}] Downloading {key} from {s3Res.text}")
        inFile = downloadFile(s3Res.text, key)
        if not inFile:
            updateJob(jobId, -1, None)
            deleteMessage(receiptHandle, messageId)
            print(f"[{datetime.datetime.now()}] {key} failed to download from s3 api")
            return
        print(f"[{datetime.datetime.now()}] {key} downloaded to {inFile}")
        
        # Update the job record to in progress
        if(updateJob(jobId, 1, '')):
            print(f"[{datetime.datetime.now()}] Job record set to in progress")
        else:
            print(f"[{datetime.datetime.now()}] Unable to update job record")

        # Call our diarizations functions to start splicing and transcribing speakers
        diarizations = startDiarization(inFile, userId, jobId)

        # Finally save the transcriptions to the db
        print(f"[{datetime.datetime.now()}] Saving transcripts to db...")
        transcript = saveTranscript(diarizations)
        print(f"Transcript ID: {transcript['insertedId']}")
        if(transcript['insertedId']):
            if(updateJob(jobId, 2, transcript['insertedId'])):
                print(f"[{datetime.datetime.now()}] Transcript saved and job updated")
            else:
                print(f"[{datetime.datetime.now()}] Transcript saved but unable to update job status")
        else:
            if(updateJob(jobId, -1)):
                print(f"[{datetime.datetime.now()}] Job record {jobId} updated")
            else:
                print(f"[{datetime.datetime.now()}] Unable to update job record {jobId}")
            print(f"[{datetime.datetime.now()}] Transcript was not saved to database")
    except Exception as error:
        print(f"[{datetime.datetime.now()}] {error}")
        updateJob(jobId, -1, None)

    # Delete message from SQS
    try:
        deleteMessage(receiptHandle, messageId)
    except Exception as error:
        print(f"[{datetime.datetime.now()}] {error}")

    # Cleanup all local files
    for file in os.listdir('in'):
        try:
            print(f"[{datetime.datetime.now()}] Deleting {file}")
            os.remove('in/'+file)
        except Exception as error:
            print(f"[{datetime.datetime.now()}] Unable to delete {file}")
    for file in os.listdir('work'):
        try:
            print(f"[{datetime.datetime.now()}] Deleting {file}")
            os.remove('work/'+file)
        except Exception as error:
            print(f"[{datetime.datetime.now()}] Unable to delete {file}")

# Function to save transcript to database
def saveTranscript(jsonBody):
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post(f"{os.getenv('DB_HOST')}/newtranscript", data=jsonBody, headers=headers)
        if response.ok:
            return response.json()
        else:
            return False
    except Exception as err:
        print(f"[{datetime.datetime.now()}] {err}")
        return False

# Function to update job record status and populate transcript table id
def updateJob(jobId, status, transcriptId, percentage=None):
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post(f"{os.getenv('DB_HOST')}/updatestatus", json={"jobId": jobId, "status": status, "transcriptId": transcriptId, "percentage":percentage}, headers=headers)
        if response.ok:
            return True
        else:
            return False
    except Exception as err:
        print(f"[{datetime.datetime.now()}] {err}")
        return False

# Delete SQS message
def deleteMessage(receiptHandle, messageId):
    session = requests.Session()
    print(f"[{datetime.datetime.now()}] Deleting message {messageId}")
    headers = {'Content-Type': 'application/json'}
    try:
        session.post(os.getenv('S3_API')+'/delete', data={'ReceiptHandle':receiptHandle}, headers=headers)
    except Exception as err:
        print(f"[{datetime.datetime.now()}] {err}")  

# Function to download file from S3 API service
def downloadFile(url, key):
    response = requests.get(url)
    print(f"{response.status_code}")
    if not response.status_code == 200:
        print(f"[{datetime.datetime.now()}] {response.status_code} : {response.text}")
        return
    path = os.path.join(inFolder,key)
    try:
        with open(path, mode="wb") as file:
            file.write(response.content)
        return path
    except Exception as err:
        print(f"[{datetime.datetime.now()}] {err}")
        return
    
def checkProgress():
    with ProgressHook as hook:
        print(f"[{datetime.datetime.now()}] {hook}")
        print(f"Update test")
schedule.every(10).seconds.do(checkMessages)

while True:
    schedule.run_pending()
    time.sleep(1)