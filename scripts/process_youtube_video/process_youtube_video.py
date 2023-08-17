import uuid
import json
import argparse
import asyncio
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from loguru import logger
from models.models import Document, DocumentMetadata
from datastore.datastore import DataStore
from datastore.factory import get_datastore
from services.extract_metadata import extract_metadata_from_document
from youtube_transcript_api import YouTubeTranscriptApi

# API information
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'
DEVELOPER_KEY = 'YOUR_DEVELOPER_KEY' # Replace this with your API key

youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=DEVELOPER_KEY)

async def process_youtube_video(
        video_id: str,
        datastore: DataStore
):
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    logger.info('Downloaded transcript for video {}'.format(video_id))
    concat_transcript = ' '.join([v['text'] for v in transcript])
    print("""
###STARTVIDEO###
{data}
    """).format(json.dumps(dict(
        url='https:www.youtube.com/watch?v={}'.format(video_id),
        transcript=concat_transcript,
        title='test'
    )))
    # create a document object with the id or a random id, text and metadata
    document = Document(
        id=video_id,
        text=concat_transcript,
        metadata={
            'source': 'youtube',
            'source_id': video_id,
            'url': 'https://www.youtube.com/watch?v={}'.format(video_id),
        }
    )

    await datastore.upsert(documents=[document])
    logger.info('upserted document {} to datastore'.format(video_id))


async def main():
    # parse the command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--videoid", required=True, help="The ID of the Youtube video")
    parser.add_argument(
        "--extract_metadata",
        default=False,
        type=bool,
        help="A boolean flag to indicate whether to try to extract metadata from the document (using a language model)",
    )
    args = parser.parse_args()

    # get the arguments
    video_id = args.videoid

    # initialize the db instance once as a global variable
    datastore = await get_datastore()
    # process the json dump
    await process_youtube_video(
        video_id, datastore
    )


if __name__ == "__main__":
    asyncio.run(main())
