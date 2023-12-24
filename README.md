# LLM-File-Explorer

Create a file exploration tool that allows users to interact with and query files on their local machine using a combination of Ollama, LlamaIndex, Qdrant vector database, and a locally installed language model. This tool will make use of natural language processing (NLP) techniques to understand user queries and retrieve relevant files.

Adapted from [LlamaIndex Blog](https://blog.llamaindex.ai/running-mixtral-8x7-locally-with-llamaindex-e6cebeabe0ab)

**Tech Stack:**

- Ollama for language model interaction.
- LlamaIndex for file indexing.
- Qdrant vector database for efficient vector search.

## Usage

1. [Create a vitual environement and] install dependencies

    ```shell
    pip install -r requirements.txt
    ```

2. Launch Ollama `llama2` model in another terminal

    ```shell
    ollama run llama2
    ```

3. Run smoke test

    ```shell
    python3 test.py
    ```

4. Add your CV as input data (`data/cv.pdf`) and run query on document

    ```shell
    python3 main.py "Give me 100-word summary of this CV"
    ```
