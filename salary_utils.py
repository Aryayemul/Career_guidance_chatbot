import re

# Pattern to identify salary queries
SALARY_PATTERNS = [
    r'salary',
    r'earn',
    r'pay',
    r'income',
    r'compensation',
    r'wage',
    r'stipend',
    r'remuneration',
    r'how much.*make',
    r'how much.*paid',
    r'how much.*get',
]

# Pattern to identify monthly salary requests
MONTHLY_SALARY_PATTERNS = [
    r'monthly',
    r'per month',
    r'month',
    r'monthly salary',
    r'monthly income',
    r'monthly pay',
    r'monthly wage',
]

# Pattern to identify daily salary requests
DAILY_SALARY_PATTERNS = [
    r'daily',
    r'per day',
    r'day',
    r'daily salary',
    r'daily income',
    r'daily pay',
    r'daily wage',
    r'day rate',
    r'per diem',
]

# Function to convert years of experience to experience level
def years_to_experience_level(years_text):
    """Convert years of experience to standardized experience level."""
    # Try to extract the number of years from the text
    year_pattern = re.compile(r'(\d+)\s*(?:year|yr)')
    match = year_pattern.search(str(years_text).lower())
    
    if match:
        years = int(match.group(1))
        if years < 2:
            return "Entry"
        elif years < 5:
            return "Mid"
        else:
            return "Senior"
    
    # If we couldn't extract years, look for level keywords
    text_lower = str(years_text).lower()
    if any(term in text_lower for term in ['senior', 'lead', 'head', 'principal']):
        return "Senior"
    elif any(term in text_lower for term in ['mid', 'intermediate', 'experienced']):
        return "Mid"
    elif any(term in text_lower for term in ['entry', 'junior', 'fresher', 'graduate', 'trainee']):
        return "Entry"
    
    # Default experience level
    return "Entry"

# Function to extract job role and experience from query
def extract_job_role_and_experience(text):
    """Extract job role and experience from natural language input."""
    # Check for years of experience pattern in the input
    year_pattern = re.compile(r'(\d+)\s*(?:year|yr)')
    year_match = year_pattern.search(text.lower())
    
    # Initialize experience level
    experience_level = None
    job_role = text
    
    if year_match:
        years = int(year_match.group(1))
        # Convert years to experience level
        experience_level = years_to_experience_level(years)
        
        # Remove the years part from the job role
        # Look for common phrases that separate job role from experience
        separators = [
            "with", "having", "of", "for", "experience", 
            "experiance", "years", "year", "yr", "yrs"
        ]
        
        for separator in separators:
            parts = text.lower().split(separator, 1)
            if len(parts) > 1:
                job_role = parts[0].strip()
                break
    else:
        # If no years are mentioned, check for experience level keywords
        experience_keywords = {
            "senior": "Senior",
            "lead": "Senior",
            "principal": "Senior",
            "mid": "Mid",
            "intermediate": "Mid",
            "junior": "Entry",
            "entry": "Entry",
            "fresher": "Entry",
            "trainee": "Entry",
            "graduate": "Entry"
        }
        
        for keyword, level in experience_keywords.items():
            if keyword in text.lower():
                experience_level = level
                # Try to extract job role
                parts = text.lower().split(keyword, 1)
                if len(parts) > 1:
                    if parts[0].strip():
                        job_role = parts[0].strip()
                    else:
                        job_role = parts[1].strip()
                break
    
    # If still no experience level, default to Entry
    if not experience_level:
        experience_level = "Entry"
    
    return job_role, experience_level

def is_daily_salary_query(text):
    """Check if the query is specifically about daily salary."""
    text_lower = text.lower()
    
    # Check for daily salary-related patterns
    for pattern in DAILY_SALARY_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    
    return False

def is_monthly_salary_query(text):
    """Check if the query is specifically about monthly salary."""
    text_lower = text.lower()
    
    # Check for monthly salary-related patterns
    for pattern in MONTHLY_SALARY_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    
    return False

def is_salary_query(text):
    """Check if a query is about salary information."""
    text_lower = text.lower()
    
    # Check for salary-related patterns
    for pattern in SALARY_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    
    return False 