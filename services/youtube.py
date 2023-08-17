from youtube_transcript_api import YouTubeTranscriptApi
from loguru import logger

from models.models import Document


async def get_document_from_youtube_video(
    video_id: str
) -> Document:
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    logger.info('Downloaded transcript for video {}'.format(video_id))
    concat_transcript = ' '.join([v['text'] for v in transcript])
    document = Document(
        id=video_id,
        text=concat_transcript,
        metadata={
            'source': 'youtube',
            'source_id': video_id,
            'url': 'https://www.youtube.com/watch?v={}'.format(video_id),
        }
    )

    return document

