#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pyaudio
import wave
import time


# In[3]:


wf = wave.open("./sc_v01_test/seven/0ea0e2f4_nohash_0.wav", 'rb')


# In[4]:


# instantiate PyAudio (1)
p = pyaudio.PyAudio()


# In[7]:


# define callback (2)
def callback(in_data, frame_count, time_info, status):
    data = wf.readframes(frame_count)
    print(len(data))
    return (data, pyaudio.paContinue)


# In[8]:


# open stream using callback (3)
stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                frames_per_buffer=640,
                output=True,
                stream_callback=callback)


# In[9]:


stream.start_stream()


# In[10]:


while stream.is_active():
    print("STARTING SLEEP")
    time.sleep(10)
    print("DONE SLEEP")


# In[11]:


# stop stream (6)
stream.stop_stream()
stream.close()
wf.close()

# close PyAudio (7)
p.terminate()


# In[ ]:




