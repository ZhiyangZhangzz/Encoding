RESUME_EXTRACTION_PROMPT = """You are a professional resume analysis assistant. 
From the following resume, extract the key information and return **ONLY valid JSON**. 

Do not include explanations, Markdown formatting, or code fences (no ```json). 
The output must be strictly valid JSON with double quotes.

The JSON should have this structure:

{{
  "title": "", 
  "skills": "comma-separated list of skills",
  "experience": [
    {{"position": "...", "company": "...", "years": "..."}}
  ],
  "education": {{
    "school": "...",
    "degree": "..."
  }},
  "responsibilities": [],
  "requirements": [],
  "keywords": "comma-separated keywords relevant to the job description"
}}

Here is the resume text:
{resume_text}
"""

JD_EXTRACTION_PROMPT = """You are a professional job description analysis assistant. 
From the following job description, extract the key information and return **ONLY valid JSON**. 

Do not include explanations, Markdown formatting, or code fences (no ```json). 
The output must be strictly valid JSON with double quotes.

The JSON should have this structure:

JSON schema:
{{
  "title": "job title",
  "skills": "comma-separated list of required skills",
  "experience": [],
  "education": {{
    "school": "",
    "degree": ""
  }},
  "responsibilities": [
    "responsibility 1",
    "responsibility 2"
  ],
  "requirements": [
    "requirement 1",
    "requirement 2"
  ],
  "keywords": "comma-separated keywords extracted from the JD"
}}

Here is the job description text:
{jd_text}
"""
