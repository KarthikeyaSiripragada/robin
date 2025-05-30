# Robin: A multi-agent system for automating scientific discovery

See our [blog](https://www.futurehouse.org/research-announcements/demonstrating-end-to-end-scientific-discovery-with-robin-a-multi-agent-system) or [arXiv](https://arxiv.org/abs/2505.13400) preprint for more info.

## Prerequisites

- **Python:** Version 3.12 or higher.
- **API Keys:**
  - `FUTUREHOUSE_API_KEY`: For accessing FutureHouse platform agents (Crow, Falcon).
  - An API key for your chosen LLM provider (e.g., `OPENAI_API_KEY` if using OpenAI models). Robin uses LiteLLM, so it can support various providers.
  - The "Finch" (data analysis) portion of this repo needs access to the FutureHouse platform closed beta. To request access, visit https://platform.futurehouse.org/profile, and use the "Rate Limit Increase" form to request access to Finch. Without access, all the hypothesis and experiment generation code can still be run.

## Setup Instructions

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/Future-House/robin.git
    cd robin
    ```

2.  **Create and Activate a Virtual Environment (Recommended):**

    ```bash
    uv venv .venv
    source .venv/bin/activate
    ```

    OR

    ```bash
    python3 -m venv .robin_env
    source .robin_env/bin/activate
    ```

3.  **Install Dependencies:**
    The project uses `pyproject.toml` for dependency management. Install the base package and development dependencies (which include Jupyter):

    ```bash
    uv pip install -e .[dev]
    ```

    OR

    ```bash
    pip install -e .[dev]
    ```

4.  **Set API Keys:**
    It's highly recommended to set your API keys as environment variables. Create a `.env` file in the `robin` directory:
    ```
    FUTUREHOUSE_API_KEY="your_futurehouse_api_key_here"
    OPENAI_API_KEY="your_openai_api_key_here"
    # etc. for other LLM providers
    ```
    The notebook and `RobinConfiguration` will attempt to load these. Alternatively, you can pass them directly when creating the `RobinConfiguration` object in the notebook.

## Running Robin via `robin_demo.ipynb`

1.  **Launch Jupyter Notebook or JupyterLab:**
    Navigate to the `robin` directory in your terminal (ensure your virtual environment is activated) and run:

    ```bash
    jupyter notebook
    # OR
    jupyter lab
    ```

2.  **Open the Notebook:**
    In the Jupyter interface, open `robin_demo.ipynb`.

3.  **Configure Robin:**
    Locate the cell where the `RobinConfiguration` object is created:

    ```python
    config = RobinConfiguration(
        disease_name="DISEASE_NAME",  # <-- Customize the disease name here
        # You can also explicitly set API keys here if not using environment variables:
        # futurehouse_api_key="your_futurehouse_api_key_here",
        # llm_config={ # Example if you want to specify a different LLM
        #     "model_list": [{
        #         "model_name": "gpt-4-turbo",
        #         "litellm_params": {"model": "gpt-4-turbo", "api_key": os.getenv("OPENAI_API_KEY")}
        #     }]
        # },
        # llm_name="gpt-4-turbo" # if overriding default
    )
    ```

    - **Modify `disease_name`**: Change `"DISEASE_NAME"` to your target disease.
    - **API Keys**: If you didn't set environment variables, you can provide the keys directly in the `RobinConfiguration` instantiation.
    - **LLM Choice**: The default is `o4-mini`. You can change `llm_name` and `llm_config` in `RobinConfiguration` if you wish to use a different model supported by LiteLLM (ensure you have the corresponding API key set).
    - Other parameters like `num_queries`, `num_assays`, `num_candidates` can also be adjusted here if needed.

4.  **Run the Notebook Cells:**
    Execute the cells in the notebook sequentially. The notebook is structured to guide you through:
    - **Experimental Assay Generation:** Generates and ranks potential experimental assays.
    - **Therapeutic Candidate Generation:** Based on the top assay, generates and ranks therapeutic candidates.
    - **(Optional) Experimental Data Analysis:** If you have experimental data, this section can analyze it and feed insights back into candidate generation. This currently requires access to the Finch closed beta.

## Expected Output

- **Logs:** Detailed logs will be printed in the notebook output and/or your console, showing the progress of each step.
- **Files:** Results, including generated queries, literature reviews, candidate proposals, and rankings, will be saved in a new subdirectory within `robin_output/`. The subdirectory name is typically based on the `disease_name` and a timestamp (e.g., `robin_output/PCOS_2023-10-27_10-30/`).

## Advanced Usage

A full example trajectory can be found in the `robin_full.ipynb` notebook. This notebook includes the parameters and agents used in the paper. Note that the parameters used in this notebook exceeds the current free rate limits.

While this guide focuses on the `robin_demo.ipynb` notebook, the `robin` Python module (in the `robin/` directory) can be imported and its functions (`experimental_assay`, `therapeutic_candidates`, `data_analysis`) can be used programmatically in your own Python scripts for more customized workflows.
