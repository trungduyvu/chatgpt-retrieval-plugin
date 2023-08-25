import asyncio
from datetime import datetime

import time

from scipy.spatial.distance import cosine
from youtube_transcript_api import YouTubeTranscriptApi
from loguru import logger

import pandas as pd
import numpy as np
from models.models import Document
from langchain.text_splitter import NLTKTextSplitter, CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
import networkx as nx
from networkx.algorithms import community
from langchain import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain

from services.chunks import get_text_chunks


def create_sentences(segments, MIN_WORDS, MAX_WORDS):
    # Combine the non-sentences together
    sentences = []

    is_new_sentence = True
    sentence_length = 0
    sentence_num = 0
    sentence_segments = []

    for i in range(len(segments)):
        if is_new_sentence == True:
            is_new_sentence = False
        # Append the segment
        sentence_segments.append(segments[i])
        segment_words = segments[i].split(' ')
        sentence_length += len(segment_words)

        # If exceed MAX_WORDS, then stop at the end of the segment
        # Only consider it a sentence if the length is at least MIN_WORDS
        if (sentence_length >= MIN_WORDS and segments[i][-1] == '.') or sentence_length >= MAX_WORDS:
            sentence = ' '.join(sentence_segments)
            sentences.append({
                'sentence_num': sentence_num,
                'text': sentence,
                'sentence_length': sentence_length
            })
            # Reset
            is_new_sentence = True
            sentence_length = 0
            sentence_segments = []
            sentence_num += 1

    return sentences


# def create_chunks(sentences, CHUNK_LENGTH, STRIDE):
#     sentences_df = pd.DataFrame(sentences)
#
#     chunks = []
#     for i in range(0, len(sentences_df), (CHUNK_LENGTH - STRIDE)):
#         chunk = sentences_df.iloc[i:i + CHUNK_LENGTH]
#         chunk_text = ' '.join(chunk['text'].tolist())
#
#         chunks.append({
#             'start_sentence_num': chunk['sentence_num'].iloc[0],
#             'end_sentence_num': chunk['sentence_num'].iloc[-1],
#             'text': chunk_text,
#             'num_words': len(chunk_text.split(' '))
#         })
#
#     chunks_df = pd.DataFrame(chunks)
#     return chunks_df.to_dict('records')
#

def parse_title_summary_results(results):
    out = []
    for e in results:
        e = e.replace('\n', '')
        if '|' in e:
            processed = {'title': e.split('|')[0],
                         'summary': e.split('|')[1][1:]
                         }
        elif ':' in e:
            processed = {'title': e.split(':')[0],
                         'summary': e.split(':')[1][1:]
                         }
        elif '-' in e:
            processed = {'title': e.split('-')[0],
                         'summary': e.split('-')[1][1:]
                         }
        else:
            processed = {'title': '',
                         'summary': e
                         }
        out.append(processed)
    return out


async def summarize_stage_1(chunks_text, title, channel):
    # Prompt to get title and summary for each chunk
    map_prompt_template = """You are an expert copy writer. Your task is to give the following text an informative title. 
    Then, on a new line, write a 75-100 word summary of the following snippet of transcript from a youtube video titled 
    """ + title + """ by channel """ + channel + """.:
  {text}

  Return your answer in the following format:
  Title | Summary...
  e.g. 
  Why Artificial Intelligence is Good | AI can make humans more productive by automating many repetitive processes.
"""

    map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])

    # Define the LLMs
    map_llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')
    map_llm_chain = LLMChain(llm=map_llm, prompt=map_prompt)
    s = time.perf_counter()
    # Run the input through the LLM chain (works in parallel)
    map_llm_chain_results = await asyncio.gather(*[map_llm_chain.arun(text=t) for t in chunks_text])
    elapsed = time.perf_counter() - s
    print(f'Stage 1 Elapsed time: {elapsed:0.2f} seconds')
    stage_1_outputs = parse_title_summary_results([e for e in map_llm_chain_results])


    return stage_1_outputs


def get_topics(title_similarity, num_topics=8, bonus_constant=0.25, min_size=3):
    s = time.perf_counter()
    proximity_bonus_arr = np.zeros_like(title_similarity)
    for row in range(proximity_bonus_arr.shape[0]):
        for col in range(proximity_bonus_arr.shape[1]):
            if row == col:
                proximity_bonus_arr[row, col] = 0
            else:
                proximity_bonus_arr[row, col] = 1 / (abs(row - col)) * bonus_constant

    title_similarity += proximity_bonus_arr

    title_nx_graph = nx.from_numpy_array(title_similarity)

    desired_num_topics = num_topics
    # Store the accepted partitionings
    topics_title_accepted = []

    resolution = 0.85
    resolution_step = 0.01
    iterations = 40

    # Find the resolution that gives the desired number of topics
    topics_title = []
    while len(topics_title) not in [desired_num_topics, desired_num_topics + 1, desired_num_topics + 2]:
        topics_title = community.louvain_communities(title_nx_graph, weight='weight', resolution=resolution)
        resolution += resolution_step
    topic_sizes = [len(c) for c in topics_title]
    sizes_sd = np.std(topic_sizes)
    modularity = community.modularity(title_nx_graph, topics_title, weight='weight', resolution=resolution)

    lowest_sd_iteration = 0
    # Set lowest sd to inf
    lowest_sd = float('inf')

    for i in range(iterations):
        topics_title = community.louvain_communities(title_nx_graph, weight='weight', resolution=resolution)
        modularity = community.modularity(title_nx_graph, topics_title, weight='weight', resolution=resolution)

        # Check SD
        topic_sizes = [len(c) for c in topics_title]
        sizes_sd = np.std(topic_sizes)

        topics_title_accepted.append(topics_title)

        if sizes_sd < lowest_sd and min(topic_sizes) >= min_size:
            lowest_sd_iteration = i
            lowest_sd = sizes_sd

    # Set the chosen partitioning to be the one with highest modularity
    topics_title = topics_title_accepted[lowest_sd_iteration]
    print(f'Best SD: {lowest_sd}, Best iteration: {lowest_sd_iteration}')

    topic_id_means = [sum(e) / len(e) for e in topics_title]
    # Arrange title_topics in order of topic_id_means
    topics_title = [list(c) for _, c in sorted(zip(topic_id_means, topics_title), key=lambda pair: pair[0])]
    # Create an array denoting which topic each chunk belongs to
    chunk_topics = [None] * title_similarity.shape[0]
    for i, c in enumerate(topics_title):
        for j in c:
            chunk_topics[j] = i

    elapsed = time.perf_counter() - s
    print(f'Get Topic Elapsed time: {elapsed:0.2f} seconds')
    return {
        'chunk_topics': chunk_topics,
        'topics': topics_title
    }


def summarize_stage_2(stage_1_outputs, topics, title, channel, summary_num_words=400):
    s = time.perf_counter()

    # Prompt that passes in all the titles of a topic, and asks for an overall title of the topic
    title_prompt_template = """You are an expert copy writer. Write an informative title that summarizes each of the following groups of titles. Make sure that the titles capture as much information as possible, 
  and are different from each other:
  {text}

  Return your answer in a numbered list, with new line separating each title:
  1. Title 1
  2. Title 2
  3. Title 3

  Return only the list of title, nothing else
  """

    map_prompt_template = """You are an expert copy writer. Write a concise 75-100 word summary of the 
    following partial summary of a youtube video. 
    ```
    {text}
    ```
"""

    combine_prompt_template = """You are an expert copy writer. Your task is to write a 
    """ + str(summary_num_words) + """-word summary of a youtube video from summaries of its main topics, delimited by ```. 
    The title of the video is """ + title + """ by channel """ + channel + """.:
    Rules:
    - Remove irrelevant information:
    - Include a main takeaways section using bullet points to list information
    - Use straight forward language
    - Return only the summary, nothing else
    - start with "In this video of <channel>, ..."
  ```
  {text}
  ```
  
  """

    title_prompt = PromptTemplate(template=title_prompt_template, input_variables=["text"])
    map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])
    combine_prompt = PromptTemplate(template=combine_prompt_template, input_variables=["text"])

    topics_data = []
    for c in topics:
        topic_data = {
            'summaries': [stage_1_outputs[chunk_id]['summary'] for chunk_id in c],
            'titles': [stage_1_outputs[chunk_id]['title'] for chunk_id in c]
        }
        topic_data['summaries_concat'] = ' '.join(topic_data['summaries'])
        topic_data['titles_concat'] = ', '.join(topic_data['titles'])
        topics_data.append(topic_data)

    # Get a list of each community's summaries (concatenated)
    topics_summary_concat = [c['summaries_concat'] for c in topics_data]
    topics_titles_concat = [c['titles_concat'] for c in topics_data]

    # Concat into one long string to do the topic title creation
    topics_titles_concat_all = ''''''
    for i, c in enumerate(topics_titles_concat):
        topics_titles_concat_all += f'''{i + 1}. {c}
    '''

    # print('topics_titles_concat_all', topics_titles_concat_all)

    llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')
    title_llm_chain = LLMChain(llm=llm, prompt=title_prompt)
    title_llm_chain_input = [{'text': topics_titles_concat_all}]
    title_llm_chain_results = title_llm_chain.apply(title_llm_chain_input)

    # Split by new line
    titles = title_llm_chain_results[0]['text'].split('\n')
    # Remove any empty titles
    titles = [t for t in titles if t != '']
    # Remove spaces at start or end of each title
    titles = [t.strip() for t in titles]

    reduce_llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo-16k')

    # Run the map-reduce chain
    docs = [Document(page_content=t) for t in topics_summary_concat]
    chain = load_summarize_chain(chain_type="map_reduce", map_prompt=map_prompt, combine_prompt=combine_prompt,
                                 return_intermediate_steps=True,
                                 llm=llm, reduce_llm=reduce_llm)

    output = chain({"input_documents": docs}, return_only_outputs=True)
    summaries = output['intermediate_steps']
    stage_2_outputs = [{'title': t, 'summary': s} for t, s in zip(titles, summaries)]
    final_summary = output['output_text']

    # Return: stage_1_outputs (title and summary), stage_2_outputs (title and summary), final_summary, chunk_allocations
    out = {
        'stage_2_outputs': stage_2_outputs,
        'final_summary': final_summary
    }
    elapsed = time.perf_counter() - s
    print(f'Stage 2 Elapsed time: {elapsed:0.2f} seconds')

    return out


async def summarize_youtube_video_transcript(transcripts, title, channel):
    # merge transcript into one string
    transcript = ' '.join([v['text'] for v in transcripts])
    # # split by words
    # words = transcript.split(' ')
    # # split words into chunks of 80 words
    # chunks = [words[i:i + 80] for i in range(0, len(words), 80)]

    text_chunks = get_text_chunks(transcript, 200)

    # stage 1
    stage_1_outputs = await summarize_stage_1(text_chunks, title, channel)
    stage_1_summaries = [e['summary'] for e in stage_1_outputs]
    stage_1_titles = [e['title'] for e in stage_1_outputs]
    num_1_chunks = len(stage_1_summaries)

    openai_embed = OpenAIEmbeddings()

    summary_embeds = np.array(openai_embed.embed_documents(stage_1_summaries))
    title_embeds = np.array(openai_embed.embed_documents(stage_1_titles))

    # Get similarity matrix between the embeddings of the chunk summaries
    summary_similarity_matrix = np.zeros((num_1_chunks, num_1_chunks))
    summary_similarity_matrix[:] = np.nan

    for row in range(num_1_chunks):
        for col in range(row, num_1_chunks):
            # Calculate cosine similarity between the two vectors
            similarity = 1 - cosine(summary_embeds[row], summary_embeds[col])
            summary_similarity_matrix[row, col] = similarity
            summary_similarity_matrix[col, row] = similarity

    num_topics = min(int(num_1_chunks / 4), 8)
    topics_out = get_topics(summary_similarity_matrix, num_topics=num_topics, bonus_constant=0.2)
    chunk_topics = topics_out['chunk_topics']
    topics = topics_out['topics']

    stage_2_outputs = summarize_stage_2(stage_1_outputs, topics,  title, channel, summary_num_words=250)
    stage_2_titles = [e['title'] for e in stage_2_outputs['stage_2_outputs']]
    stage_2_summaries = [e['summary'] for e in stage_2_outputs['stage_2_outputs']]
    final_summary = stage_2_outputs['final_summary']

    return dict(final_summary=final_summary, titles=stage_2_titles, summaries=stage_2_summaries)


async def get_document_from_youtube_video(
    video_id= str
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

