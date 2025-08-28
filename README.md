# Resume-Ranker

Resume-Ranker is a resume screening and ranking system based on natural language processing.  
It supports three main strategies:

## Bi-Encoder
Encodes job descriptions and resumes independently into embeddings and calculates cosine similarity.  
This method is efficient and suitable for initial candidate filtering.

## Cross-Encoder
Encodes job description and resume pairs jointly, providing more accurate semantic similarity.  
This method is slower but used for reranking the top candidates.

## JSON Encoding
Parses both job descriptions and resumes into structured JSON, then compares specific fields:
- Skills
- Experience
- Education  

Each field is scored separately using cosine similarity, and the final score is a weighted combination of these field-level similarities.
