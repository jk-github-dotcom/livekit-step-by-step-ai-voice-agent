#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Livekit AI Voice Agent with MCP Server M-1

# https://docs.livekit.io/agents/start/voice-ai/

# https://github.com/jk-github-dotcom/livekit-step-by-step-ai-voice-agent
# https://dashboard.render.com/web/srv-d1udm5c9c44c73csmkdg/settings


# In[2]:


# Choose kernel "Python (.venv_livekit)"

# python -m venv .venv_livekit
# .venv_livekit\scripts\activate
# pip install ipykernel
# python -m ipykernel install --user --name=.venv_livekit --display-name "Python (.venv_livekit)"

# pip freeze > .req_venv_livekit


# In[3]:


# Description


# In[5]:


# Documentation
# https://learn.deeplearning.ai/courses/building-ai-voice-agents-for-production/lesson/idsit/voice-agent-overview
# https://docs.livekit.io/agents/start/voice-ai/

# Create a project AI_Voice_Assistant on livekit cloud
# https://cloud.livekit.io/projects/p_34qw70e3usd/overview

# Your agent strings together three specialized providers into a high-performance voice pipeline.
# You need accounts and API keys for each.

# Components
# STT
# Provider: OpenAI 
# https://docs.livekit.io/agents/integrations/stt/openai/
# https://pypi.org/project/livekit-plugins-openai/
# Parameter: model, language (model: gpt-4o-transcribe (default) or whisper-1)
# OPENAI_API_KEY in .env
# pip install livekit-plugins-openai

# LLM:
# Provider: OpenAI 
# https://docs.livekit.io/agents/integrations/llm/openai/
# https://pypi.org/project/livekit-plugins-openai/
# Parameter: model, temperature, tool_choice (model: gpt-4o-mini (default) or gpt-4o or o1)
# OPENAI_API_KEY in .env
# pip install livekit-plugins-openai

# TTS
# Provider: Hume
# https://docs.livekit.io/agents/integrations/tts/hume/
# https://pypi.org/project/livekit-plugins-hume/
# Parameter: voice, description, speed, context, instant_mode (voice and/or description: see documentation)
# HUME_API_KEY
# pip install livekit-plugins-hume

# Usage:
# See documentation (voice by name, id or even generated)

#from livekit.plugins import hume
#from hume.tts import PostedUtteranceVoiceWithName (only livekit-plugins-hume==1.0.23 not newest livekit-plugins-hume==1.2.1)

# livekit-plugins-hume==1.0.23

#session = AgentSession(
#   tts=hume.TTS(
#      voice=PostedUtteranceVoiceWithName(name="Colton Rivers", provider="HUME_AI"),
#      description="The voice exudes calm, serene, and peaceful qualities, like a gentle stream flowing through a quiet forest.",
#   )
# ... llm, stt, etc.
#)

# livekit-plugins-hume==1.2.1

#        tts=hume.TTS(
#            voice=hume.VoiceByName(name="Colton Rivers", provider=hume.VoiceProvider.hume),
#            description="The voice exudes calm, serene, and peaceful qualities, like a gentle stream flowing through a quiet forest.",
#            voice=hume.VoiceById(id="0bae3af7-1f3a-426e-9285-13015427577c"),
#        ),


# In[ ]:


# Additional documentation (slightly different)

# Tech with Tim
# Python AI Voice Assistant & Agent - Full Tutorial
# https://www.youtube.com/watch?v=DNWLIAK4BUY


# In[ ]:


# MCP tool

# https://docs.livekit.io/agents/build/tools/#model-context-protocol-mcp-


# In[ ]:


# .env

#LIVEKIT_URL=wss://aivoiceassistant-yf8o74l4.livekit.cloud
#LIVEKIT_API_KEY=<your API Key>
#LIVEKIT_API_SECRET=<your API Secret>

#OPENAI_API_KEY=<Your OpenAI API Key>
#HUME_API_KEY=<Your Hume API Key>
...


# In[ ]:


# requirements.txt

#python-dotenv

#livekit-agents[deepgram,openai,cartesia,hume,elevenlabs,silero,turn-detector]~=1.0
#livekit-agents[mcp]~=1.0

#livekit-plugins-noise-cancellation~=0.2

# or alternatively for deepgram,openai,cartesia,hume,elevenlabs
#livekit-plugins-deepgram
#livekit-plugins-openai
#livekit-plugins-cartesia
#livekit-plugins-hume
#livekit-plugins-elevenlabs


# In[1]:


#from dotenv import load_dotenv
#import asyncio

#from livekit import agents
#from livekit.agents import AgentSession, Agent, RoomInputOptions
#from livekit.plugins import (
#    openai,
#    hume,
#    noise_cancellation,
#    silero,
#)
#from livekit.plugins.turn_detector.multilingual import MultilingualModel

#load_dotenv()


# In[2]:


# from hume.tts import PostedUtteranceVoiceWithName (only livekit-plugins-hume==1.0.23 not newest livekit-plugins-hume==1.2.1)
# from hume.tts import PostedUtteranceVoiceWithId (only livekit-plugins-hume==1.0.23 not newest livekit-plugins-hume==1.2.1)


# In[7]:


from dotenv import load_dotenv
import asyncio

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.agents import mcp
from livekit.plugins import (
    openai,
    hume,
    noise_cancellation,
    silero, # Silero VAD plugin
)
# deployemnt on render fails because of memory restriction (512 MB) (python livekit_ai_voice_agent_mcp.py download-files)
# following documentation https://docs.livekit.io/agents/build/turns/turn-detector/
# try EnglishModel
# Model	Base    Model	        Size on Disk    Per Turn Latency
# English-only	SmolLM2-135M	66 MB	        ~15-45 ms
# Multilingual	Qwen2.5-0.5B	281 MB	        ~50-160 ms

# from livekit.plugins.turn_detector.multilingual import MultilingualModel # LiveKit turn detector plugin
from livekit.plugins.turn_detector.english import EnglishModel
# session = AgentSession(
#    turn_detection=EnglishModel(),
#    stt=openai.STT(model="gpt-4o-transcribe", language = "en"),
#    # ... vad, stt, tts, llm, etc.
#) 

# from hume.tts import PostedUtteranceVoiceWithName (only livekit-plugins-hume==1.0.23 not newest livekit-plugins-hume==1.2.1)
# from hume.tts import PostedUtteranceVoiceWithId (only livekit-plugins-hume==1.0.23 not newest livekit-plugins-hume==1.2.1)

load_dotenv()

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful voice AI assistant.")

async def entrypoint(ctx: agents.JobContext):
    session = AgentSession(
#        stt=openai.STT(model="gpt-4o-transcribe"),
        stt=openai.STT(model="gpt-4o-transcribe", language = "en"), # because of turn detection
        llm=openai.LLM(model="gpt-4o-mini"),
#        tts=hume.TTS(
#            voice=hume.VoiceByName(name="Colton Rivers", provider=hume.VoiceProvider.hume),
#            description="The voice exudes calm, serene, and peaceful qualities, like a gentle stream flowing through a quiet forest.",
#            voice=hume.VoiceById(id="0bae3af7-1f3a-426e-9285-13015427577c"),
#        ),
        tts=openai.TTS(
            model="gpt-4o-mini-tts",
            voice="shimmer",
            instructions="Speak in a friendly and conversational tone.",
        ),
        vad=silero.VAD.load(),
        turn_detection=EnglishModel(),
#        turn_detection=MultilingualModel(),
        mcp_servers=[
            mcp.MCPServerHTTP(
#                "https://jn2atbn3.rpcld.cc/mcp/9dd7b985-3694-40e5-a1d6-f9fe3c941dfe/sse" # See "N8N 2025-05-22 MCP server"
                url="https://jn2atbn3.rpcld.cc/mcp/9bdf3e79-c503-4a5c-b055-17bca1b40242/sse", # See "N8N 2025-06-12 MCP Server M-1"
                timeout=5,
                client_session_timeout_seconds=5,
            )       
        ]         
    )
    
    agent=Assistant() # or agent=Assistant() in session.start()
    
    print("🎤 Starting session...")
    await session.start(
        room=ctx.room,
        agent=agent, # see comment agent=Assistant()
        room_input_options=RoomInputOptions(
            # LiveKit Cloud enhanced noise cancellation
            # - If self-hosting, omit this parameter
            # - For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(), 
        ),
    )
    print("✅ Session started and agent joined room.")
    
    await ctx.connect()

    await session.generate_reply(
        instructions="""
# Role
You are a friendly and helpful assistant called Anna.
Keep your answer short and to the point.
Greet the user and offer your assistance.

# Tools
You have access to the following tools via MCP:
## Gmail - Read eMails
Use this tool to read unread eMails
## Gmail - Send eMails
Use this tool to send eMails. This tool requires the recipient email address, the email subject and email body text.
If you have to retrieve an eMail address use the Pinecone Vector Store.
Please underwrite the eMail with 'Anna (Personal Assistant of Jochen)'
## Pinecone Vector Store
Use this tool to retrieve eMail addresses. If you do not find the eMail address, stop and inform the user by 'Sorry, I cannot find the requested email address'.
## Google Calendar - Get calendar events
Use this tool to retrieve events, their time and location from the calendar
## Google Calendar - Make calendar events
Use this tool to schedule events for the calendar
"""
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))


# In[ ]:


# Step 00: Save and export notebook as executable script livekit_ai_voice_agent.py 


# In[ ]:


# Step 01: Download model files

# Run in project directory and in your virtual environment .venv_livekit
# python livekit_ai_voice_agent_mcp.py download-files (in this case already done by python livekit_ai_voice_agent.py download-files)

# To use the turn-detector, silero, or noise-cancellation plugins, you first need to download the model files:
# C:\Users\Gebruiker\.cache\huggingface\hub\models--livekit--turn-detector


# In[ ]:


# Step 02 Option 01: Speak to your agent via console

# Run in project directory and in your virtual environment .venv_livekit
# Start your agent in console mode to run inside your terminal:

# python livekit_ai_voice_agent_mcp.py console


# In[ ]:


# Step 02 Option 02: Speak to your agent via livekit playground

# Run in project directory and in your virtual environment .venv_livekit
# Start your agent in dev mode to connect it to LiveKit and make it available from anywhere on the internet:

# python livekit_ai_voice_agent_mcp.py dev


# In[ ]:


# More information:

# Use the Agents playground to speak with your agent and explore its full range of multimodal capabilities.

# https://docs.livekit.io/agents/start/playground/
# https://agents-playground.livekit.io/

# Congratulations, your agent is up and running. Continue to use the playground or the console mode as you build and test your agent.

# Agent CLI modes
# In the console mode, the agent runs locally and is only available within your terminal.

# Run your agent in dev (development / debug) or start (production) mode to connect to LiveKit and join rooms.


# In[ ]:


# Step 02 Option 03: Speak to your agent via livekit sandbox

# I can access the agent on the internet

#    via the Livekit playground: https://agents-playground.livekit.io/

#    via the Livekit Sandbox directly: https://synchronized-server-2h6j7i.sandbox.livekit.io/ 
#    via the Livekit Sandbox overview: https://cloud.livekit.io/projects/p_34qw70e3usd/sandbox


# In[ ]:


# Step 02 Option 04: Speak to your agent via your own frontend

# See documentations README.md in project livekit-step-by-step for the frontend, the token server and this voice agent


# In[ ]:


# NEXT STEPS
# https://docs.livekit.io/agents/start/voice-ai/#next-steps

# https://docs.livekit.io/agents/start/frontend/
# https://docs.livekit.io/agents/build/
...

