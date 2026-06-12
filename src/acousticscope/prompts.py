"""Prompt templates used by the AcousticScope prototype."""

RESPONSE_GENERATION_PROMPT = """\
You are simulating a concise human response to an Alexa skill.

Alexa said:
{alexa_response}

Reply with a direct answer in five words or fewer. If the answer is yes, reply only
with "yes". Avoid repeating previous responses:
{previous_responses}
"""

UNDERSTANDING_CLASSIFIER_PROMPT = """\
Imagine a conversation between a user and Alexa.

User command:
{user_command}

Alexa response:
{alexa_response}

Decide whether Alexa understood the user's intent. Return exactly one label:
understood or not_understood.
"""
