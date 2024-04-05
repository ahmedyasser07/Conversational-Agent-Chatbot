# Conversational-Agent-Chatbot
An AI chatbot and an agent that is built with LangChain and openAI API.

This project implements a sophisticated conversational agent using LangChain, designed to leverage large language models (LLMs) with enhanced capabilities like long-term memory and dynamic information retrieval. It showcases the integration of conversational context, memory management, and output parsing to create more coherent and contextually aware dialogues.

## Overview

The conversational agent is built around several key components:

- **Agent Management (`agent.py`)**: Orchestrates the dialogue flow, managing interactions between the user, the LLM, and other system components.
- **Conversation Retrieval (`conversation-retrieval.py`)**: Retrieves historical conversation context to inform current dialogue decisions and LLM responses.
- **Large Language Model Interface (`llm.py`)**: Facilitates direct interaction with large language models, handling requests and processing responses.
- **Long-Term Memory for LLM (`long_term_memory_llm.py`)**: Enhances the LLM's capabilities by incorporating a mechanism for recalling past interactions or learned information over longer conversations.
- **Output Parsing (`parse-output.py`)**: Parses and processes the LLM's output for further use within the system, ensuring responses are formatted correctly and relevant information is extracted.
- **Retrieval Chain Logic (`retrieval-chain.py`)**: Implements a retrieval-chain mechanism, allowing the agent to dynamically pull in external information or conversation history as part of its response generation process.

## Setup

1. **Clone the Repository**

   Begin by cloning this repository to your local machine:

   ```
   git clone https://github.com/yourusername/LangChain-Conversational-Agent.git
   ```

2. **Install Dependencies**

   Navigate to the project directory and install the necessary Python dependencies:

   ```
   cd LangChain-Conversational-Agent
   pip install -r requirements.txt
   ```

   > Note: This project assumes you have Python 3.8+ installed. Adjust commands accordingly if using a virtual environment.

## Running the Project

To run the conversational agent, execute the main script (assuming `agent.py` serves as the entry point):

```
python agent.py
```

Follow the on-screen instructions to interact with the agent. The system is designed to be modular, so you can easily adapt or extend its functionalities to fit your needs.

## Contributing

Contributions to the LangChain Conversational Agent are welcome! Whether it's bug fixes, feature enhancements, or documentation improvements, feel free to fork the repository and submit a pull request.

