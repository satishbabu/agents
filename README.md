# Learning AI Agents
Following 'Master AI Agents...' course on Udemy from Ed Donner.


# Setup instructions

Requires python 3.12

From the terminal `uv self update` to get latest version of uv.  Then un `uv sync`.

### Add keys for AI models

Open AI:
1. Create an OpenAI account at https://platform.openai.com/.  *disable automatic rechage*
2. Create your API key from https://platform.openai.com/api-keys
3. Add it to .env file as `OPENAI_API_KEY=`.  *Ensure .env is not checked-in as it will expose senstive keys


Anthropic and Google
Repeae above to fill in 
```
GOOGLE_API_KEY=xxxx
ANTHROPIC_API_KEY=xxxx
```

