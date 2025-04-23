"""
Modal application definition for a RAG (Retrieval-Augmented Generation) chatbot.

This script sets up a Modal app that:
1. Downloads a pre-trained language model (Llama 3 8B Instruct).
2. Quantizes the model to FP8 using TensorRT-LLM tools.
3. Builds a TensorRT-LLM inference engine for the quantized model.
4. Processes local text/markdown files (`./content` directory) into a knowledge base:
    - Chunks large files based on token count.
    - Embeds content (full files or chunks) using Sentence Transformers.
    - Creates a FAISS index for efficient similarity search.
5. Defines a RAGRetriever class to fetch relevant context from the knowledge base.
6. Defines a ChatModel class (Modal Class) that:
    - Loads the TensorRT-LLM engine and the RAG retriever on startup.
    - Exposes a FastAPI endpoint (`/chat`) to handle user questions.
    - Retrieves relevant context using the RAGRetriever.
    - Constructs a prompt incorporating the retrieved context and system instructions.
    - Generates responses using the accelerated TensorRT-LLM engine.
7. Uses specific pinned versions for dependencies to ensure reproducibility.
"""

from typing import List, Dict, Optional
import os
import json
from pathlib import Path
import modal
import pydantic
import logging
import numpy as np
from transformers import AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# === Configuration Constants ===

# --- Model & Download ---
MODEL_ID = "NousResearch/Meta-Llama-3-8B-Instruct"
MODEL_REVISION = "b1532e4dee724d9ba63fe17496f298254d87ca64" # Pinned revision
MODEL_DIR = "/root/model/model_input" # Download location within container

# --- Quantization & Engine Build ---
GIT_HASH = "b0880169d0fb8cd0363049d91aa548e58a41be07" # Pinned TensorRT-LLM commit for quantize script
CONVERSION_SCRIPT_URL = f"https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/{GIT_HASH}/examples/quantization/quantize.py"
N_GPUS = 1
GPU_CONFIG = f"H100:{N_GPUS}" # GPU type and count for quantization, build, and runtime
DTYPE = "float16"            # Original model precision
QFORMAT = "fp8"              # Quantized weight precision
KV_CACHE_DTYPE = "fp8"       # Quantized KV cache precision
CALIB_SIZE = "512"           # Calibration dataset size for quantization
CKPT_DIR = "/root/model/model_ckpt" # Quantized checkpoint location
ENGINE_DIR = "/root/model/model_output" # Final TensorRT engine location
MAX_INPUT_LEN = 4096         # Max input tokens for the engine
MAX_OUTPUT_LEN = 4096        # Max *generated* tokens allowed by the engine runtime
MAX_NUM_TOKENS = 8192        # Max total tokens (input + output) for the engine build
MAX_BATCH_SIZE = 1           # Max batch size for the engine
QUANTIZATION_ARGS = f"--dtype={DTYPE} --qformat={QFORMAT} --kv_cache_dtype={KV_CACHE_DTYPE} --calib_size={CALIB_SIZE}"
SIZE_ARGS = f"--max_input_len={MAX_INPUT_LEN} --max_num_tokens={MAX_NUM_TOKENS} --max_batch_size={MAX_BATCH_SIZE}"
PLUGIN_ARGS = "--use_fp8_context_fmha enable" # Enable FP8 optimizations

# --- RAG Knowledge Base ---
CONTENT_DIR = "/content" # Directory within container for knowledge base files
MAX_FILE_TOKENS = 1500   # Max tokens before a file is chunked
CHUNK_SIZE = 500         # Target chunk size in tokens
CHUNK_OVERLAP = 50       # Token overlap between chunks
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2' # Sentence Transformer model for embeddings
RAG_DOC_PATH = os.path.join(CONTENT_DIR, 'documents.json')
RAG_INDEX_PATH = os.path.join(CONTENT_DIR, 'embeddings.faiss')
RAG_MODEL_NAME_PATH = os.path.join(CONTENT_DIR, 'model_name.txt')

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
COLOR = {
    "HEADER": "\033[95m",
    "BLUE": "\033[94m",
    "GREEN": "\033[92m",
    "RED": "\033[91m",
    "ENDC": "\033[0m",
}

# === Modal Image Definition ===

# --- Base Image ---
tensorrt_image = modal.Image.from_registry(
    "nvidia/cuda:12.4.1-devel-ubuntu22.04",
    add_python="3.10",  # TRT-LLM requires Python 3.10
).entrypoint([])

# --- Core Dependencies ---
tensorrt_image = tensorrt_image.apt_install(
    "openmpi-bin", "libopenmpi-dev", "git", "git-lfs", "wget"
).pip_install(
    "tensorrt_llm==0.14.0",
    "pynvml<12",  # Avoid breaking change in pynvml API

    # Web/RAG deps
    "numpy<2", # faiss-cpu dependency
    "fastapi[standard]==0.115.4",
    "pydantic==2.9.2",
    "starlette==0.41.2",
    "sentence-transformers==2.5.1",
    "faiss-cpu==1.7.4",
    "transformers>=4.38.0", # Ensure Llama3 tokenizer support
    "langchain==0.1.16",

    pre=True,
    extra_index_url="https://pypi.nvidia.com",
)

# --- Model Download Function ---
def download_model():
    import os
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    os.makedirs(MODEL_DIR, exist_ok=True)
    snapshot_download(
        MODEL_ID,
        local_dir=MODEL_DIR,
        ignore_patterns=["*.pt", "*.bin"], # Using safetensors
        revision=MODEL_REVISION,
    )
    move_cache()

# --- Add Model Download Step to Image ---
tensorrt_image = (
    tensorrt_image.pip_install(
        "hf-transfer==0.1.8",
        "huggingface_hub==0.26.2",
        "requests~=2.31.0",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"}) # Use faster downloader
    .run_function(
        download_model,
        timeout=20 * 60, # Allow ample time for download
    )
)

# --- Tokenizer Download Function (for build-time RAG setup) ---
def download_tokenizer_for_build():
    """Downloads the tokenizer during image build for use in setup_knowledge_base."""
    from transformers import AutoTokenizer # Import scoped to build context

    logger.info(f"Downloading tokenizer {MODEL_ID} during image build...")
    try:
        AutoTokenizer.from_pretrained(MODEL_ID) # Download and cache
        logger.info(f"Tokenizer {MODEL_ID} downloaded successfully.")
    except Exception as e:
        logger.error(f"Failed to download tokenizer {MODEL_ID} during build: {e}", exc_info=True)
        raise # Fail build if tokenizer download fails

# --- Add Tokenizer Download Step to Image ---
tensorrt_image = tensorrt_image.run_function(
    download_tokenizer_for_build,
    secrets=[modal.Secret.from_dict({"HF_HUB_ENABLE_HF_TRANSFER": "1"})]
)

# --- Add Model Quantization Step to Image ---
tensorrt_image = (
    tensorrt_image.run_commands(
        [
            f"wget {CONVERSION_SCRIPT_URL} -O /root/convert.py",
            f"python /root/convert.py --model_dir={MODEL_DIR} --output_dir={CKPT_DIR}"
            + f" --tp_size={N_GPUS}"
            + f" {QUANTIZATION_ARGS}",
        ],
        gpu=GPU_CONFIG, # Requires GPU
    )
)

# --- Add Engine Build Step to Image ---
tensorrt_image = (
    tensorrt_image.run_commands(
        [
            f"trtllm-build --checkpoint_dir {CKPT_DIR} --output_dir {ENGINE_DIR}"
            + f" --workers={N_GPUS}"
            + f" {SIZE_ARGS}"
            + f" {PLUGIN_ARGS}"
        ],
        gpu=GPU_CONFIG, # Requires GPU, should match runtime GPU
    ).env({"TLLM_LOG_LEVEL": "INFO"}) # Enable more detailed TRT-LLM logs
)

# --- Knowledge Base Setup Function ---
def setup_knowledge_base():
    """
    Sets up the knowledge base during image build.
    Processes files in CONTENT_DIR, chunks if necessary, embeds content,
    and saves metadata, FAISS index, and embedding model name.
    """
    import os
    import json
    from sentence_transformers import SentenceTransformer
    import numpy as np
    import faiss
    from transformers import AutoTokenizer
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    logger.info("Running setup_knowledge_base with file grouping logic...")

    container_content_dir = CONTENT_DIR
    os.makedirs(container_content_dir, exist_ok=True)

    logger.info(f"Loading tokenizer '{MODEL_ID}' for token counting...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID) # Uses pre-downloaded tokenizer
        logger.info("Tokenizer loaded.")
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}", exc_info=True)
        raise

    logger.info(f"Loading embedding model '{EMBEDDING_MODEL_NAME}' for setup...")
    cache_dir = os.environ.get("SENTENCE_TRANSFORMERS_HOME", "/cache") # Use cache dir from secret
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, cache_folder=cache_dir)
    logger.info("Embedding model loaded.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=lambda text: len(tokenizer.encode(text)) # Use LLM tokenizer for chunk length
    )

    final_documents_list = [] # Stores metadata including content snippet
    processed_files_count = 0

    if os.path.isdir(container_content_dir):
        logger.info(f"Processing files in: {container_content_dir}")
        try:
            content_files = os.listdir(container_content_dir)
        except Exception as e:
            logger.error(f"Could not list directory {container_content_dir}: {e}")
            content_files = []

        for filename in content_files:
            file_path = os.path.join(container_content_dir, filename)
            if os.path.isfile(file_path) and (filename.endswith('.txt') or filename.endswith('.md')):
                logger.info(f"Processing file: {filename}")
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        full_content = f.read()

                    token_count = len(tokenizer.encode(full_content))
                    logger.info(f"File '{filename}' token count: {token_count}")

                    if token_count <= MAX_FILE_TOKENS:
                        logger.info(f"Adding entire file '{filename}' as one item.")
                        chunk_id = f"{filename}_full"
                        final_documents_list.append({
                            'chunk_id': chunk_id,
                            'source_file': filename,
                            'was_chunked': False,
                            'chunk_index': 0,
                            'content': full_content # Store full content
                        })
                    else:
                        logger.info(f"Chunking file '{filename}'.")
                        chunks = text_splitter.split_text(full_content)
                        logger.info(f"Split into {len(chunks)} chunks.")
                        for i, chunk_text in enumerate(chunks):
                            chunk_id = f"{filename}_chunk_{i}"
                            final_documents_list.append({
                                'chunk_id': chunk_id,
                                'source_file': filename,
                                'was_chunked': True,
                                'chunk_index': i,
                                'content': chunk_text # Store chunk content
                            })
                    processed_files_count += 1
                except Exception as e:
                    logger.error(f"Error processing file {filename}: {e}", exc_info=True)
            else:
                 logger.debug(f"Skipping non-text/md or non-file item: {filename}")
    else:
        logger.warning(f"Directory {container_content_dir} not found inside container.")

    logger.info(f"Processed {len(final_documents_list)} total items (full files or chunks) from {processed_files_count} files.")

    # Define output paths using constants
    doc_output_path = RAG_DOC_PATH
    index_output_path = RAG_INDEX_PATH
    model_name_output_path = RAG_MODEL_NAME_PATH

    if not final_documents_list:
        logger.warning("No content to embed. Creating empty knowledge base files.")
        with open(doc_output_path, 'w') as f: json.dump([], f)
        dimension = embedding_model.get_sentence_embedding_dimension()
        index = faiss.IndexFlatL2(dimension)
        faiss.write_index(index, index_output_path)
        with open(model_name_output_path, 'w') as f: f.write(EMBEDDING_MODEL_NAME)
        logger.info("Empty knowledge base setup complete.")
        return

    logger.info("Extracting content for embedding...")
    all_content_to_embed = [doc['content'] for doc in final_documents_list]

    logger.info("Creating embeddings...")
    embeddings = embedding_model.encode(all_content_to_embed)
    logger.info(f"Created {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}.")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    if embeddings.dtype != np.float32: embeddings = embeddings.astype(np.float32)
    logger.info(f"Adding {embeddings.shape[0]} embeddings to FAISS index.")
    index.add(embeddings)

    logger.info(f"Saving documents metadata (including content) JSON to: {doc_output_path}")
    with open(doc_output_path, 'w') as f:
        json.dump(final_documents_list, f, indent=2)

    logger.info(f"Saving FAISS index to: {index_output_path}")
    faiss.write_index(index, index_output_path)

    logger.info(f"Saving embedding model name to: {model_name_output_path}")
    with open(model_name_output_path, 'w') as f: f.write(EMBEDDING_MODEL_NAME)

    logger.info("Knowledge base setup complete.")


# --- Add Local Content and Run KB Setup in Image ---
content_source_path = Path(__file__).parent / "content" # Assumes 'content' dir is sibling to script
tensorrt_image = tensorrt_image.add_local_dir(
    local_path=content_source_path,
    remote_path=CONTENT_DIR,
    copy=True,
).run_function(
    setup_knowledge_base,
    secrets=[modal.Secret.from_dict({"SENTENCE_TRANSFORMERS_HOME": "/cache"})] # Provide cache for embedding model download
)

# === Modal App Definition ===

app = modal.App(f"example-trtllm-rag-{MODEL_ID.split('/')[-1]}", image=tensorrt_image)

# --- API Request/Response Schemas ---

class ChatRequest(pydantic.BaseModel):
    """Schema for incoming chat requests."""
    question: str

class ChatResponse(pydantic.BaseModel):
    """Schema for outgoing chat responses."""
    response: str
    error: Optional[str] = None

class ChatRequestWithRAG(pydantic.BaseModel):
    """Alias for chat request schema (used by endpoint)."""
    question: str

# --- RAG Retriever Class ---

class RAGRetriever:
    """
    Handles retrieval of relevant context from the pre-built knowledge base.

    Loads the FAISS index, document metadata (including content snippets),
    and the embedding model necessary to perform similarity searches.
    """
    def __init__(self):
        import json
        import os
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer

        logger.info("Initializing RAGRetriever...")
        self.doc_path = RAG_DOC_PATH
        self.model_name_path = RAG_MODEL_NAME_PATH
        self.index_path = RAG_INDEX_PATH

        # Check if required RAG files exist from the image build
        if not all(os.path.exists(p) for p in [self.doc_path, self.model_name_path, self.index_path]):
             error_msg = f"Cannot initialize RAGRetriever: One or more RAG files missing in {CONTENT_DIR}."
             logger.error(error_msg)
             raise FileNotFoundError(error_msg)

        try:
            logger.info(f"Loading documents metadata from: {self.doc_path}")
            with open(self.doc_path, 'r') as f:
                self.documents_metadata = json.load(f) # Contains content snippets
            logger.info(f"Loaded metadata for {len(self.documents_metadata)} items.")

            with open(self.model_name_path, 'r') as f:
                model_name = f.read().strip()
            logger.info(f"Loading embedding model: {model_name}")
            cache_dir = os.environ.get("SENTENCE_TRANSFORMERS_HOME", "/cache")
            self.embedding_model = SentenceTransformer(model_name, cache_folder=cache_dir)

            logger.info(f"Loading FAISS index from: {self.index_path}")
            self.index = faiss.read_index(self.index_path)
            logger.info(f"FAISS index loaded. Index size: {self.index.ntotal}")

            if len(self.documents_metadata) != self.index.ntotal:
                 logger.warning(f"Metadata count ({len(self.documents_metadata)}) != FAISS index size ({self.index.ntotal}). Check KB setup.")

        except Exception as e:
            logger.error(f"Error during RAGRetriever resource loading: {e}", exc_info=True)
            raise

        logger.info(f"RAG system initialized.")
        if not self.documents_metadata or self.index.ntotal == 0:
            logger.warning("RAGRetriever initialized, but knowledge base appears empty.")

    def get_relevant_context(self, query, initial_k=10, final_k=3):
        """
        Retrieves relevant context for a given query.

        Performs FAISS search, identifies the most relevant source file based on the top hit,
        and returns either the full file content (if not chunked) or the top `final_k`
        relevant chunks belonging to that source file.

        Args:
            query (str): The user's question.
            initial_k (int): The number of initial candidates to retrieve from FAISS.
            final_k (int): The maximum number of chunks to return if the source file was chunked.

        Returns:
            str: The combined relevant context, or an empty string if none found or KB is empty.
        """
        import numpy as np

        if not self.documents_metadata or self.index.ntotal == 0:
             logger.warning(f"Retrieval skipped for query '{query}': Knowledge base is empty.")
             return ""

        try:
            query_embedding = self.embedding_model.encode([query])
            if query_embedding.dtype != np.float32:
                query_embedding = query_embedding.astype(np.float32)

            actual_initial_k = min(initial_k, self.index.ntotal)
            logger.debug(f"Performing initial FAISS search for top {actual_initial_k} items.")
            distances, indices = self.index.search(query_embedding, actual_initial_k)
            retrieved_indices = indices[0]

            if not retrieved_indices.size: return "" # No hits

            top_item_index = retrieved_indices[0]
            if top_item_index >= len(self.documents_metadata):
                 logger.error(f"FAISS index {top_item_index} out of bounds for metadata (size {len(self.documents_metadata)}).")
                 return ""
            top_item_metadata = self.documents_metadata[top_item_index]
            target_source_file = top_item_metadata['source_file']
            was_target_chunked = top_item_metadata['was_chunked']
            logger.info(f"Most relevant item from file '{target_source_file}' (chunked={was_target_chunked}).")

            final_context = ""
            if not was_target_chunked:
                # Return full content of the top-hit file
                final_context = top_item_metadata.get('content', '')
                if not final_context:
                     logger.warning(f"Metadata for non-chunked file '{target_source_file}' missing 'content'.")
                else:
                     logger.info(f"Returning full content of file '{target_source_file}'.")
            else:
                # Gather top k chunks specifically from the target file
                logger.info(f"Gathering top {final_k} relevant chunks from file '{target_source_file}'.")
                relevant_chunks_content = []
                for idx in retrieved_indices:
                    if len(relevant_chunks_content) >= final_k: break
                    if idx < len(self.documents_metadata): # Bounds check
                        item_meta = self.documents_metadata[idx]
                        if item_meta['source_file'] == target_source_file:
                            chunk_content = item_meta.get('content', '')
                            if chunk_content:
                                relevant_chunks_content.append(chunk_content)
                            else:
                                logger.warning(f"Metadata for chunk index {idx} (file {target_source_file}) missing 'content'.")

                if relevant_chunks_content:
                    final_context = "\n\n".join(relevant_chunks_content)
                    logger.info(f"Returning {len(relevant_chunks_content)} chunks from file '{target_source_file}'.")
                else:
                    logger.warning(f"Found no relevant chunks with content from file '{target_source_file}' in initial results.")

            return final_context

        except Exception as e:
            logger.error(f"Error during context retrieval for query '{query}': {e}", exc_info=True)
            return "" # Return empty context on error

# --- Helper Function ---
def extract_assistant_response(output_text):
    """
    Extracts the assistant's response from the full model output string.
    Specific to the LLaMA 3 Instruct chat template.

    See format documentation: https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/.
    """
    parts = output_text.split("<|start_header_id|>assistant<|end_header_id|>")
    if len(parts) > 1:
        response = parts[1].split("<|eot_id|>")[0].strip()
        response = response.replace("<|eot_id|>", "").strip() # Clean up any trailing tokens
        return response
    else:
        # Fallback if the expected assistant marker isn't found
        logger.warning("Could not find assistant header in model output, returning raw output.")
        return output_text.strip()


# --- Main Chat Model Class (Modal Class) ---
@app.cls(
    gpu=GPU_CONFIG,
    scaledown_window=10 * 60, # Keep alive for 10 mins after last request
    image=tensorrt_image, # Use the image with TRT engine and RAG data
    secrets=[modal.Secret.from_dict({"SENTENCE_TRANSFORMERS_HOME": "/cache"})] # Needed for RAGRetriever init
)
class ChatModel:
    """
    Modal Class responsible for loading the TRT-LLM model, RAG retriever,
    and handling chat requests via a FastAPI endpoint.
    """
    @modal.enter()
    def load_and_initialize(self):
        """Loads the TRT-LLM engine, tokenizer, and initializes the RAG retriever on container start."""
        import time
        import tensorrt_llm
        from tensorrt_llm.runtime import ModelRunner
        from transformers import AutoTokenizer

        init_start_time = time.monotonic()
        logger.info(f"{COLOR['HEADER']}🥶 Cold boot: Initializing TRT-LLM engine and RAG retriever...{COLOR['ENDC']}")

        # 1. Initialize TRT-LLM Engine
        logger.info("Loading TRT-LLM engine...")
        runner_kwargs = dict(
            engine_dir=ENGINE_DIR,
            lora_dir=None,
            rank=tensorrt_llm.mpi_rank(),
            max_output_len=MAX_OUTPUT_LEN,
        )
        self.model = ModelRunner.from_dir(**runner_kwargs)
        logger.info("TRT-LLM engine loaded.")

        # 2. Initialize Tokenizer
        logger.info(f"Loading tokenizer: {MODEL_ID}")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.tokenizer.add_special_tokens({"pad_token": self.tokenizer.eos_token})
        self.tokenizer.padding_side = "left"
        self.pad_id = self.tokenizer.pad_token_id
        self.end_id = self.tokenizer.eos_token_id
        logger.info("Tokenizer loaded.")

        # 3. Initialize RAG Retriever
        logger.info("Initializing RAGRetriever...")
        try:
            self.retriever = RAGRetriever() # Uses SENTENCE_TRANSFORMERS_HOME from secret
            logger.info("RAGRetriever initialized.")
        except Exception as e:
            logger.critical(f"FATAL: Failed to initialize RAGRetriever during startup: {e}", exc_info=True)
            raise RuntimeError("Failed to initialize RAG system") from e # Prevent startup if RAG fails

        # 4. Load System Prompt
        # Note: This system prompt is quite specific and prescriptive.
        self.system_prompt = """You are a highly professional resume chatbot representing Evan Richardson. Your sole task is to answer user questions about Evan's skills and experience based *only* on the context provided below the '---CONTEXT---' marker.

        **CRITICAL INSTRUCTIONS:**

        1.  **Direct & Concise:** Answer the question directly and succinctly. Your entire response MUST be between 2 and 4 sentences long. NO exceptions.
        2.  **Synthesize, Don't Copy:** Read the context, understand the relevant points, and explain them in your own words. Do NOT copy sentences or long phrases directly from the context. Distill the information.
        3.  **No Introductory Phrases:** Do NOT start your response with phrases like "Based on the provided context," "According to the resume," or similar phrases referring to the source. Just state the answer.
        4.  **Standard Paragraph Format:** Use standard paragraph format only. Do NOT use bullet points, numbered lists, asterisks, or any other list format. Integrate multiple points smoothly into prose sentences if necessary.
        5.  **Professional Tone:** Maintain a confident, professional, and articulate tone. Use relevant industry terms appropriately.
        6.  **Stick to the Context:** If the context doesn't contain the requested information, state clearly and professionally that the information is not available in the provided materials (e.g., "The provided context does not detail Evan's specific experience with [topic]."). Do not invent information.

        ---CONTEXT---
        {context}
        ---END CONTEXT---

        Now, answer the user's query strictly following all the instructions above.
        """
        logger.info("System prompt loaded.")

        init_duration_s = time.monotonic() - init_start_time
        logger.info(f"{COLOR['HEADER']}🚀 Cold boot finished in {init_duration_s:.2f}s{COLOR['ENDC']}")

    @modal.fastapi_endpoint(method="POST", label="chat", docs=True)
    def chat_endpoint(self, data: ChatRequestWithRAG) -> ChatResponse:
        """
        Handles chat requests: retrieves context using RAG, generates response using TRT-LLM.
        """
        import time

        question = data.question
        logger.info(f"Handling request for question: '{question}'")
        request_start_time = time.monotonic_ns()

        # 1. Retrieve Context
        context = ""
        retrieval_start_time = time.monotonic_ns()
        try:
            if not hasattr(self, 'retriever'):
                 logger.error("Retriever not initialized on ChatModel instance!")
                 return ChatResponse(response="Error: Backend RAG component not ready.", error="Retriever missing.")

            context = self.retriever.get_relevant_context(question, final_k=3) # Retrieve top 3 chunks from best file
            retrieval_duration_ms = (time.monotonic_ns() - retrieval_start_time) / 1e6
            logger.info(f"Context retrieval took {retrieval_duration_ms:.1f}ms.")

        except Exception as e:
            retrieval_duration_ms = (time.monotonic_ns() - retrieval_start_time) / 1e6
            logger.error(f"Error during context retrieval (took {retrieval_duration_ms:.1f}ms): {e}", exc_info=True)
            logger.warning("Proceeding with generation using empty context due to retrieval error.")
            # context remains ""

        # 2. Construct Prompt
        final_system_content = self.system_prompt.format(context=context if context else "No relevant context found.")
        messages = [
            {"role": "system", "content": final_system_content},
            {"role": "user", "content": question}
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True, # Important for instruct models
            tokenize=False,
        )

        # 3. Tokenize Input
        inputs_t = self.tokenizer([prompt], return_tensors="pt", padding=True, truncation=False)["input_ids"]
        input_length = inputs_t.shape[1]

        # Calculate max_new_tokens dynamically based on engine limits
        # Use a smaller generation length default (e.g., 512) if input allows, up to MAX_OUTPUT_LEN
        generation_allowance = MAX_NUM_TOKENS - input_length
        max_new_tokens = min(min(512, MAX_OUTPUT_LEN), generation_allowance)

        if max_new_tokens <= 0:
             logger.error(f"Input length {input_length} exceeds or meets MAX_NUM_TOKENS {MAX_NUM_TOKENS}. Cannot generate.")
             return ChatResponse(response="Error: Input is too long.", error="Input length exceeds model capacity.")

        logger.info(f"Input length: {input_length}, Max new tokens: {max_new_tokens}")

        # 4. Generate Response using TRT-LLM
        generation_start_time = time.monotonic_ns()
        try:
            settings = dict(
                temperature=0.2,
                top_p=0.9,
                top_k=40,
                repetition_penalty=1.1,
                max_new_tokens=max_new_tokens, # Use dynamically calculated value
                end_id=self.end_id,
                pad_id=self.pad_id,
            )

            outputs_t = self.model.generate(inputs_t, **settings)

            # Decode only the generated tokens
            # Assumes model.generate() does *not* echo the input tokens in the output tensor
            output_ids = outputs_t[0, 0] # Shape is typically [batch_size, beam_width, seq_len]

            output_text = self.tokenizer.decode(output_ids, skip_special_tokens=False)

            # Extract only the assistant's response using the helper function
            response = extract_assistant_response(output_text)

            generation_duration_s = (time.monotonic_ns() - generation_start_time) / 1e9
            num_output_tokens = len(output_ids)
            tokens_per_sec = num_output_tokens / generation_duration_s if generation_duration_s > 0 else 0

            logger.info(
                f"{COLOR['GREEN']}Generated {num_output_tokens} tokens in {generation_duration_s:.2f}s ({tokens_per_sec:.1f} tokens/sec){COLOR['ENDC']}"
            )
            # logger.debug(f"{COLOR['BLUE']}Assistant response: {response[:200]}...{COLOR['ENDC']}")

            request_duration_ms = (time.monotonic_ns() - request_start_time) / 1e6
            logger.info(f"Total request handling time: {request_duration_ms:.1f}ms")

            return ChatResponse(response=response)

        except Exception as e:
            generation_duration_s = (time.monotonic_ns() - generation_start_time) / 1e9
            logger.error(f"Error during TRT-LLM generation (took {generation_duration_s:.2f}s): {e}", exc_info=True)
            return ChatResponse(response="Error during text generation.", error=str(e))