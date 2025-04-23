# Career_guidance_chatbot

An AI-powered career guidance assistant with job search capabilities, career exploration, and personalized recommendations.

#Features

AI-powered Career Guidance: Get intelligent responses to career-related questions
LinkedIn Job Search: Find and explore current job listings on LinkedIn
Career Explorer: Browse information about various careers including responsibilities, skills, and educational requirements
Salary Information: Get estimates for different job roles and experience levels
Multi-turn Memory: The chatbot remembers your conversation context for more natural interactions
Project Structure
CareerGuidance.ipynb: Main Jupyter notebook containing the career guidance application
requirements.txt: Dependencies for running the application
/data: Directory containing career information and datasets
/eda_results & /eda_plots: Exploratory data analysis results and visualizations
/model_evaluation_results: Performance metrics and model evaluation data
Technical Implementation
Chatbot Architecture
LlamaIndex Framework: Used for document indexing, retrieval, and context management
Conversational Memory: Implemented using a deque with fixed-length history to maintain context
Query Engine: Processes user inputs and generates contextually relevant responses
Fallback Mechanisms: Gracefully handles edge cases when the primary model cannot provide a response
NLP Models & Methods
HuggingFace Transformers: Leverages pre-trained language models for natural language understanding
Sentence Transformers: Used for semantic search and semantic similarity scoring
Vector Store Index: Creates embeddings of career information for efficient retrieval
Few-shot Learning: Implements examples in prompts to guide model responses for specific query types
Data Processing
BeautifulSoup: Scrapes and parses LinkedIn job listings in real-time
Regular Expressions: Detects query types and extracts key information from user inputs
NLTK: Used for text preprocessing, tokenization, and basic NLP tasks
Career Information Retrieval
Document Storage: Career information stored as structured text files
Semantic Search: Matches user queries to relevant career information based on meaning
Context-aware Responses: Generates responses that incorporate both the query and retrieved information
Technologies Used
LlamaIndex: Document indexing and retrieval
HuggingFace Transformers: NLP models and tokenizers
BeautifulSoup: For scraping job listings
PyTorch: Deep learning framework for model development
Installation
Clone this repository:
git clone <repository-url>
cd <repository-directory>
Install dependencies:
pip install -r requirements.txt
Running the Application
To launch the Career Guidance Chatbot:

python nlp_salary_predictor.py
jupyter notebook CareerGuidance.ipynb

Then run all cells in the notebook to start the application.

Usage
Chat Interface: Ask any career-related question in the chat
Job Search: Use the job search feature to find current LinkedIn job listings
Salary Information: Ask about salary ranges for specific roles and experience levels
Salary Prediction
The application includes basic salary prediction functionality for different job roles and experience levels. The implementation:

Extracts job roles and experience levels from natural language queries
Maps experience descriptions to standardized levels (Entry, Mid, Senior)
Provides estimated salary ranges based on role and experience
Requirements
Python 3.7+
4GB+ RAM (8GB+ recommended for optimal performance)
Internet connection (for job listings and initial model downloads)
Troubleshooting
If you encounter memory issues, try closing other applications
For model loading errors, ensure you have a stable internet connection
If web scraping features aren't working, check your network connection or LinkedIn's accessibility
