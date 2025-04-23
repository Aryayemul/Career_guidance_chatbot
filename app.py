import gradio as gr
import requests
from bs4 import BeautifulSoup
import time
from collections import deque
import re
import os
import nltk

# Import salary utility functions
from salary_utils import (
    years_to_experience_level, extract_job_role_and_experience,
    is_daily_salary_query, is_monthly_salary_query, is_salary_query
)

# Try to import the salary predictor model
try:
    from nlp_salary_predictor import NLPSalaryPredictor, parse_user_input
    # Initialize the model
    salary_model = NLPSalaryPredictor()
    has_salary_model = True
    print("Salary prediction model loaded successfully!")
except Exception as e:
    print(f"Warning: Could not load salary prediction model: {e}")
    has_salary_model = False

# Multi-Turn Memory
conversation_history = deque(maxlen=5)

# Configure headers to mimic browser behavior
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

# ===== CHATBOT FUNCTIONS =====
def chatbot(input_text):
    """Handles chatbot queries with multi-turn memory."""
    conversation_history.append(f"👤 User: {input_text}")
    
    # Check if this is a salary-related query
    if is_salary_query(input_text):
        try:
            # If nlp_salary_predictor is available, use it
            if has_salary_model:
                job_role, experience_level = parse_user_input(input_text)
            else:
                # Use our internal implementation
                job_role, experience_level = extract_job_role_and_experience(input_text)
            
            # Try to get a salary prediction
            try:
                if has_salary_model:
                    prediction_result = salary_model.predict_salary(job_role, experience_level)
                else:
                    # Mock prediction if model isn't available
                    prediction_result = {
                        'job_role_matched': job_role,
                        'experience_level_matched': experience_level,
                        'min_salary': 500000 if "entry" in experience_level.lower() else (1000000 if "mid" in experience_level.lower() else 2000000),
                        'max_salary': 1000000 if "entry" in experience_level.lower() else (2000000 if "mid" in experience_level.lower() else 3500000)
                    }
                
                # Check if the user wants daily or monthly salary
                is_daily = is_daily_salary_query(input_text)
                is_monthly = is_monthly_salary_query(input_text)
                
                if is_daily:
                    # Assume 260 working days per year (52 weeks × 5 days)
                    min_salary = prediction_result['min_salary'] / 260
                    max_salary = prediction_result['max_salary'] / 260
                    period = "daily"
                elif is_monthly:
                    min_salary = prediction_result['min_salary'] / 12
                    max_salary = prediction_result['max_salary'] / 12
                    period = "monthly"
                else:
                    min_salary = prediction_result['min_salary']
                    max_salary = prediction_result['max_salary']
                    period = "yearly"
                
                # Format the response nicely
                response = f"Based on my analysis for {prediction_result['job_role_matched']} at {prediction_result['experience_level_matched']} level:\n"
                response += f"The {period} salary range is typically between ₹{min_salary:,.2f} and ₹{max_salary:,.2f}."
                
                # If we provided daily or monthly, also provide annual for reference
                if is_daily or is_monthly:
                    response += f"\nOn an annual basis, this would be ₹{prediction_result['min_salary']:,.2f} to ₹{prediction_result['max_salary']:,.2f}."
            except Exception as e:
                # If prediction fails, use query_engine
                print(f"Salary prediction failed: {e}, using query_engine")
                full_context = "\n".join(conversation_history)
                response = query_engine.query(full_context)
        except Exception as e:
            # If parsing fails, use query_engine
            print(f"Salary parsing failed: {e}, using query_engine")
            full_context = "\n".join(conversation_history)
    else:
        # For non-salary queries, use the general query engine
        full_context = "\n".join(conversation_history)
        try:
            response = query_engine.query(full_context)  # Ensure query_engine is initialized
        except Exception as e:
            # Fallback if query_engine is not available
            print(f"Query engine error: {e}")
            response = f"You said: {input_text}"

    conversation_history.append(f"\n🤖 Assistant: {response}\n")
    
    # Return formatted conversation with single newlines
    return "\n".join(conversation_history)

def clear_chat():
    """Clears the conversation history."""
    conversation_history.clear()
    return ""

# ===== LINKEDIN JOB SEARCH FUNCTIONS =====
def scrape_linkedin_jobs(query, location=""):
    """Scrape LinkedIn job listings based on search query"""
    base_url = "https://www.linkedin.com/jobs/search/"
    params = {
        "keywords": query,
        "location": location,
        "position": 1,
        "pageNum": 0
    }
    
    try:
        response = requests.get(base_url, params=params, headers=HEADERS)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        jobs = []
        job_listings = soup.find_all('div', class_='base-card')
        
        for job in job_listings[:10]:  # Limit to 10 results
            title = job.find('h3', class_='base-search-card__title').text.strip()
            company = job.find('a', class_='hidden-nested-link').text.strip()
            job_location = job.find('span', class_='job-search-card__location').text.strip()
            link = job.find('a', class_='base-card__full-link')['href']
            
            jobs.append({
                'title': title,
                'company': company,
                'location': job_location,
                'link': link.split('?')[0]  # Clean URL
            })
            
            time.sleep(0.5)  # Be polite with requests
            
        return jobs
    
    except Exception as e:
        return f"Error: {str(e)}"

def display_jobs(query, location):
    # Validate query to ensure it's job-related
    if not is_valid_job_query(query):
        return "<div style='color: #FF6B6B; background-color: rgba(255, 107, 107, 0.1); padding: 15px; border-radius: 10px; border-left: 4px solid #FF6B6B;'><strong>⚠️ Invalid Job Query:</strong> Please enter a valid job title, role, or industry. Examples: 'Software Engineer', 'Data Scientist', 'Marketing', 'Healthcare'.</div>"
    
    jobs = scrape_linkedin_jobs(query, location)
    
    if isinstance(jobs, str):  # Error case
        return f"<div style='color: red'>{jobs}</div>"
    
    if not jobs:
        return "<div style='color: orange'>No jobs found for this search.</div>"
    
    html_output = f"<h3>Found {len(jobs)} jobs:</h3>"
    for idx, job in enumerate(jobs, 1):
        html_output += f"""
        <div style='margin-bottom: 20px; border-bottom: 1px solid #ccc; padding-bottom: 10px;'>
            <h4>{idx}. {job['title']}</h4>
            <p><strong>Company:</strong> {job['company']}</p>
            <p><strong>Location:</strong> {job['location']}</p>
            <p><a href="{job['link']}" target="_blank">View Job</a></p>
        </div>
        """
    return html_output

def is_valid_job_query(query):
    """
    Validate if the query appears to be job-related.
    Returns True for valid job queries, False otherwise.
    """
    if not query or len(query.strip()) < 2:
        return False
        
    # Common job-related terms/prefixes/suffixes
    job_related_terms = [
        "developer", "engineer", "manager", "assistant", "specialist", "analyst", 
        "designer", "consultant", "coordinator", "director", "technician", "representative",
        "administrator", "supervisor", "officer", "programmer", "scientist", "associate",
        "intern", "job", "career", "position", "role", "hiring", "recruitment",
        "full-time", "part-time", "remote", "work", "employment",
        # Industries
        "tech", "it", "software", "health", "medical", "finance", "banking", "education",
        "retail", "sales", "marketing", "hr", "legal", "media", "design", "construction",
        "manufacturing", "engineering", "science", "research", "data", "ai", "ml",
        # Roles
        "ceo", "cto", "cfo", "vp", "head", "lead", "junior", "senior", "mid", "staff"
    ]
    
    query_lower = query.lower()
    
    # Check if any job-related term is in the query
    for term in job_related_terms:
        if term in query_lower or query_lower in term:
            return True
            
    # If query is very long, likely not a job search
    if len(query_lower) > 50:
        return False
        
    # If query contains question marks or specific non-job phrases
    if "?" in query or "who is" in query_lower or "what is" in query_lower or "how to" in query_lower:
        return False
        
    # Default to allowing the query if we're not sure
    return True

# Custom CSS for Styling
custom_css = """
/* Modern Gradient-based Color Scheme */
:root {
    --primary: #7C4DFF;    /* Deep purple */
    --secondary: #2A2E45;  /* Navy blue */
    --background: #1A1C28; /* Space black */
    --accent: #FF6B6B;     /* Coral pink */
    --text: #F0F0FF;       /* Soft lavender */
}

/* Base Styling with Larger Fonts */
body {
    font-size: 19px !important;
    line-height: 1.7 !important;
    background: linear-gradient(135deg, var(--background) 0%, #242736 100%);
    font-family: 'Inter', system-ui, sans-serif;
    color: var(--text) !important;
}

/* Text Elements Scaling */
h1 { font-size: 2.8rem !important; }
h2 { font-size: 2.2rem !important; }
h3 { font-size: 1.9rem !important; }
p { font-size: 1.1em !important; }

/* Chat Interface */
.gr-box {
    background: rgba(42, 46, 69, 0.85) !important;
    backdrop-filter: blur(12px);
    border-radius: 18px;
    border: 1px solid rgba(124, 77, 255, 0.25);
    font-size: 1.15rem !important;
}

/* Reduced gap between buttons */
.button-row {
    gap: 5px !important;
}

/* Conversation text spacing */
.gr-textbox span {
    line-height: 1.4 !important;
    margin-bottom: 0.3rem !important;
}

/* Chat display customization */
#chat-display {
    padding: 8px !important;
    margin-bottom: 10px !important;
}

#chat-display textarea {
    line-height: 1.3 !important;
}

/* Input Fields */
input[type="text"], textarea {
    font-size: 1.2rem !important;
    padding: 18px !important;
    background: rgba(255, 255, 255, 0.08) !important;
    border: 2px solid rgba(124, 77, 255, 0.3) !important;
}

/* Buttons */
button {
    font-size: 1.25rem !important;
    padding: 16px 28px !important;
    background-image: linear-gradient(135deg, var(--primary) 0%, #906BFF 100%) !important;
    border-radius: 14px !important;
}

button:hover {
    transform: translateY(-2px) scale(1.02);
    box-shadow: 0 10px 20px rgba(124, 77, 255, 0.25) !important;
}

/* Job Listings */
.job-card {
    background: linear-gradient(145deg, #2A2E45 0%, #1F2235 100%);
    border-radius: 16px;
    padding: 24px;
    font-size: 1.1rem !important;
}

.job-card h4 {
    font-size: 1.5rem !important;
    background: linear-gradient(45deg, var(--primary), var(--accent));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Chat History */
#chat-history {
    font-size: 1.2rem !important;
    line-height: 1.8 !important;
    padding: 24px !important;
}

/* Interactive Elements */
.gr-textbox:focus, button:focus {
    box-shadow: 0 0 0 4px rgba(124, 77, 255, 0.3) !important;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 10px;
    background: var(--secondary);
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(var(--primary), var(--accent));
    border-radius: 6px;
}

/* Section Dividers */
.section-divider {
    height: 4px;
    background: linear-gradient(90deg, transparent 0%, var(--primary) 50%, transparent 100%);
    margin: 2.5rem 0;
}

/* Status Messages */
.gr-label {
    font-size: 1.3rem !important;
    letter-spacing: 0.5px;
}

/* Responsive Scaling */
@media (max-width: 768px) {
    body { font-size: 17px !important; }
    h1 { font-size: 2.2rem !important; }
    button { font-size: 1.1rem !important; }
}
"""

# Create the combined Gradio interface
with gr.Blocks(css=custom_css, title="AI Assistant with Job Search") as demo:
    gr.Markdown("# 🤖 **Career Guidance AI Chatbot **", elem_id="title")
    
    # Chatbot Section (Top)
    with gr.Group():
        gr.Markdown("### 💬 Chat with an intelligent assistant!", elem_id="subtitle")
        
        chatbox = gr.Textbox(label="Conversation History", interactive=False, lines=12, elem_id="chat-display")
        user_input = gr.Textbox(lines=1, placeholder="Type your message...", label="Your Message")
        
        with gr.Row(elem_classes=["button-row"]):
            submit_btn = gr.Button("🚀 Send")
            clear_btn = gr.Button("🗑️ Clear Chat")

        submit_btn.click(fn=chatbot, inputs=user_input, outputs=chatbox)
        user_input.submit(fn=chatbot, inputs=user_input, outputs=chatbox)
        clear_btn.click(fn=clear_chat, outputs=chatbox)
    
    # Divider
    gr.Markdown("---", elem_id="divider", elem_classes=["section-divider"])
    
    # LinkedIn Job Search Section (Bottom)
    with gr.Group():
        gr.Markdown("## 🔍 LinkedIn Job Search", elem_id="subtitle")
        gr.Markdown("Enter your job search query below:")
        
        with gr.Row():
            job_query = gr.Textbox(label="Job title or keywords", placeholder="Software Engineer")
            job_location = gr.Textbox(label="Location (optional)", placeholder="New York")
        
        search_btn = gr.Button("🔍 Search Jobs")
        
        job_output = gr.HTML()
        
        search_btn.click(
            fn=display_jobs,
            inputs=[job_query, job_location],
            outputs=job_output,
        )

# Define prediction function for Salary Predictor UI
def predict_salary_from_query(query):
    if not query or query.strip() == "":
        return "Please enter a job role or query."
    
    try:
        # Parse natural language input to extract job role and experience
        if has_salary_model:
            job_role, experience_level = parse_user_input(query)
        else:
            job_role, experience_level = extract_job_role_and_experience(query)
        
        parsed_info = f"## Parsed Input\n"
        parsed_info += f"- Job Role: **{job_role}**\n"
        parsed_info += f"- Experience Level: **{experience_level if experience_level else 'Not specified (using Entry-Level)'}**\n\n"
        
        # Get prediction
        if has_salary_model:
            result = salary_model.predict_salary(job_role, experience_level)
        else:
            # Mock prediction if model isn't available
            result = {
                'job_role_matched': job_role,
                'experience_level_matched': experience_level,
                'min_salary': 500000 if "entry" in str(experience_level).lower() else (1000000 if "mid" in str(experience_level).lower() else 2000000),
                'max_salary': 1000000 if "entry" in str(experience_level).lower() else (2000000 if "mid" in str(experience_level).lower() else 3500000)
            }
        
        if isinstance(result, dict):
            # Check if the user wants specific salary period
            is_monthly = is_monthly_salary_query(query)
            is_daily = is_daily_salary_query(query)
            
            # Create detailed output
            output = parsed_info
            output += f"## Matched To\n"
            output += f"- Job Role: **{result['job_role_matched']}**\n"
            output += f"- Experience Level: **{result['experience_level_matched']}**\n\n"
            
            # Display salary in all formats
            output += f"## Annual Salary Range\n"
            output += f"- Annual Minimum: **₹{result['min_salary']:,.2f}**\n"
            output += f"- Annual Maximum: **₹{result['max_salary']:,.2f}**\n\n"
            
            output += f"## Monthly Salary Range\n"
            output += f"- Monthly Minimum: **₹{result['min_salary']/12:,.2f}**\n"
            output += f"- Monthly Maximum: **₹{result['max_salary']/12:,.2f}**\n\n"
            
            # Assume 260 working days per year (52 weeks × 5 days)
            output += f"## Daily Salary Range\n"
            output += f"- Daily Minimum: **₹{result['min_salary']/260:,.2f}**\n"
            output += f"- Daily Maximum: **₹{result['max_salary']/260:,.2f}**\n"
            
            return output
        else:
            return parsed_info + str(result)
    except Exception as e:
        return f"Error processing your query: {str(e)}"

if __name__ == "__main__":
    demo.launch() 