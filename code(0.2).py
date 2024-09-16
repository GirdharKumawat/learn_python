# Version: 0.2

# What it does:
# Calculates the similarity between the candidate and the filter based on the education, skills, certifications, and experience.

# How it works:
# 1. Load the pre-trained universal sentence encoder model.
# 2. Define the sample data for candidates and candidate filters.
# 3. Define functions to get the embedding for the given text and calculate the cosine similarity between two embeddings.
# 4. Define a function to calculate the similarity score between the candidate and the candidate filter based on education, skills, certifications, and experience.
# 5. Iterate over the candidate filters and find the matched candidates based on the match score.
# 6. Sort the matched candidates based on the match score and display the results.

# Project setup:
# pip install tensorflow-hub
# pip install scikit-learn

# Kaggle notebook link:
# https://www.kaggle.com/code/surajchopra16/matching

# Import necessary libraries
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained universal sentence encoder model
embed = hub.load("https://www.kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/large/2")

# Sample data
candidates = [
    {"Name": "Anjali Verma", "Education": "B.Com", "Skills": ["Financial Analysis", "Accounting", "Tally"], "Certifications": ["CA Foundation", "Excel Certification"], "Experience (Years)": 3},
    {"Name": "Rajat Singh", "Education": "B.Tech", "Skills": ["Java", "Python", "Machine Learning"], "Certifications": ["AI & ML by Coursera", "AWS Certified"], "Experience (Years)": 2},
    {"Name": "Sneha Chatterjee", "Education": "MBA", "Skills": ["Digital Marketing", "SEO", "Social Media"], "Certifications": ["Google Analytics", "Digital Marketing Pro"], "Experience (Years)": 5},
    {"Name": "Arjun Nair", "Education": "BBA", "Skills": ["Management", "MS Excel", "Business Analysis"], "Certifications": ["Six Sigma Green Belt"], "Experience (Years)": 4},
    {"Name": "Kavita Sharma", "Education": "B.Ed", "Skills": ["Classroom Management", "Educational Tech"], "Certifications": ["CTET Qualified", "ICT in Education"], "Experience (Years)": 7},
    {"Name": "Vinay Mehra", "Education": "B.Tech", "Skills": ["CAD", "SolidWorks", "Product Design"], "Certifications": ["AutoCAD Certified", "Six Sigma Yellow Belt"], "Experience (Years)": 3},
    {"Name": "Priya Patel", "Education": "B.Com", "Skills": ["Auditing", "Taxation", "Tally"], "Certifications": ["TDS Certification", "CA Intermediate"], "Experience (Years)": 2},
    {"Name": "Neha Reddy", "Education": "M.Tech", "Skills": ["Python", "R", "Data Visualization"], "Certifications": ["Data Science Specialization"], "Experience (Years)": 4},
    {"Name": "Ashok Yadav", "Education": "B.A", "Skills": ["Research", "Data Analysis", "MS Excel"], "Certifications": ["Excel Advanced", "CFA Level 1"], "Experience (Years)": 3},
    {"Name": "Pooja Bhardwaj", "Education": "M.Com", "Skills": ["Corporate Finance", "Financial Modeling"], "Certifications": ["CFA Level 2", "Financial Modeling"], "Experience (Years)": 5},
    {"Name": "Akshay Kulkarni", "Education": "MBA", "Skills": ["HR Management", "Recruitment", "Employee Training"], "Certifications": ["SHRM Certified", "HR Analytics"], "Experience (Years)": 6},
    {"Name": "Isha Dhingra", "Education": "B.Sc", "Skills": ["HTML", "CSS", "JavaScript", "UI/UX"], "Certifications": ["Web Development Certified", "UI/UX Design"], "Experience (Years)": 2},
    {"Name": "Sameer Rao", "Education": "BBA", "Skills": ["Financial Markets", "Investment Banking"], "Certifications": ["NISM Certified", "Financial Markets (BSE)"], "Experience (Years)": 4},
    {"Name": "Meenal Gupta", "Education": "B.Tech", "Skills": ["Structural Analysis", "AutoCAD", "Revit"], "Certifications": ["PMP Certified", "Revit Professional"], "Experience (Years)": 5},
    {"Name": "Karan Kapoor", "Education": "MBA", "Skills": ["Brand Management", "Sales", "Market Research"], "Certifications": ["Google Ads Certified", "CRM Systems"], "Experience (Years)": 4},
    {"Name": "Sunil Prasad", "Education": "B.Ed", "Skills": ["Curriculum Development", "Teaching Pedagogy"], "Certifications": ["CTET Qualified", "Montessori Training"], "Experience (Years)": 8},
    {"Name": "Shruti Desai", "Education": "BBA", "Skills": ["Content Writing", "Social Media Strategy"], "Certifications": ["Content Marketing Certified"], "Experience (Years)": 3},
    {"Name": "Aditya Chauhan", "Education": "M.Tech", "Skills": ["AI", "Machine Learning", "Big Data"], "Certifications": ["Google Cloud Certified", "TensorFlow Dev"], "Experience (Years)": 6},
    {"Name": "Nisha Menon", "Education": "B.A", "Skills": ["Content Writing", "Editing", "Proofreading"], "Certifications": ["Cambridge English Level 5"], "Experience (Years)": 2},
    {"Name": "Rahul Sharma", "Education": "B.Tech", "Skills": ["Embedded Systems", "IoT", "Circuit Design"], "Certifications": ["IoT Specialist", "Robotics"], "Experience (Years)": 4}
]

candidate_filters = [
    {
        "Name": "Suraj",
        "Education": "B.Com",
        "Skills": ["Financial Analysis", "Accounting", "Taxation"],
        "Certifications": ["CA Foundation", "Excel Certification"],
        "Experience (Years)": 6
    },
    {
        "Name": "Girdhar",
        "Education": "B.Tech",
        "Skills": ["Java", "Python", "R"],
        "Certifications": ["AI & ML by Coursera", "AWS Certified"],
        "Experience (Years)": 4
    },
    {
        "Name": "Veer",
        "Education": "MBA",
        "Skills": ["Digital Marketing", "SEO", "Recruitment"],
        "Certifications": ["Google Analytics", "Digital Marketing Pro"],
        "Experience (Years)": 3
    }
]

# Get the embedding for given text
def get_embedding(text):
    return embed([text])

# Calculate the cosine similarity between two embeddings
def calculate_similarity(embedding1, embedding2):
    # Generate embedding
    vector1 = get_embedding(embedding1)
    vector2 = get_embedding(embedding2)
    return cosine_similarity(vector1, vector2)[0][0]

# Return the similarity score between candidate and filter
def match(candidate, candidate_filter):
    # Education similarity
    education_similarity = calculate_similarity(candidate['Education'], candidate_filter['Education'])

    # Skills similarity
    skills_similarity = calculate_similarity(' '.join(candidate['Skills']), ' '.join(candidate_filter['Skills']))

    # Certifications similarity
    certifications_similarity = calculate_similarity(' '.join(candidate['Certifications']), ' '.join(candidate_filter['Certifications']))

    # Experience similarity
    experience_similarity = calculate_similarity(str(candidate['Experience (Years)']),
                                                 str(candidate_filter['Experience (Years)']))
    # Set thresholds for matching criteria
    return (education_similarity  + skills_similarity + certifications_similarity + experience_similarity) / 4

for candidate_filter in candidate_filters:
    matched_candidates = []

    for candidate in candidates:
        match_score = match(candidate, candidate_filter)
        if match_score >= 0.5:
            matched_candidates.append({'Name': candidate['Name'], 'Match Score': match_score})

    # Sort the matched candidates based on match score
    matched_candidates = sorted(matched_candidates, key=lambda x: x['Match Score'], reverse=True)

    # Display the matched candidates
    print(f"\nFor interviewer {candidate_filter['Name']}: \n")
    for matched_candidate in matched_candidates:
        print(f"Name: {matched_candidate['Name']} and Match Score: {matched_candidate['Match Score']}")