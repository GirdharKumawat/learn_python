# Version: 0.1

# What it does:
# This code demonstrates how to match candidates with interviewers based on their profiles.

# How it works:
# 1. Profiles are created by combining multiple attributes (skills, experience, education, certifications) into a single string.
# 2. The profiles are vectorized using TF-IDF.
# 3. Cosine similarity is calculated to find the best matches between candidates and interviewers.

# Project setup:
# pip install scikit-learn

# Import necessary libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample data
interviewers = [
    {"name": "John", "skills": "python machine_learning data_analysis", "experience": 5,
     "education": "BSc Computer Science", "certifications": "AWS"},
    {"name": "Alice", "skills": "java web_development databases", "experience": 7,
     "education": "MSc Software Engineering", "certifications": "Oracle Java SE"},
    {"name": "Bob", "skills": "c++ algorithms system_design", "experience": 10,
     "education": "BSc Electrical Engineering", "certifications": "Microsoft Azure"}
]

candidates = [
    {"name": "Eve", "skills": "python data_analysis neural_networks", "experience": 3, "education": "MSc Data Science",
     "certifications": "AWS"},
    {"name": "Charlie", "skills": "java web_development mobile_apps", "experience": 5,
     "education": "BSc Computer Engineering", "certifications": "Oracle Java SE"},
    {"name": "David", "skills": "c++ operating_systems databases", "experience": 8, "education": "BSc Computer Science",
     "certifications": "Google Cloud"}
]


def create_weighted_profile(person):
    # Repeat skills for stronger emphasis (skills get more weight)
    skills_weight = person['skills'] * 2  # Multiply to give more weight to skills

    # Normalize years of experience into a phrase (e.g., 5 years -> "medium experience")
    if person['experience'] < 3:
        experience = "junior level experience"
    elif 3 <= person['experience'] <= 7:
        experience = "medium level experience"
    else:
        experience = "senior level experience"

    # Give education and certifications equal weight
    education_weight = person['education']
    certification_weight = person['certifications']

    # Combine all attributes into a profile
    return f"{skills_weight} {experience} {education_weight} {certification_weight}"

# Generate profiles
interviewer_profiles = [create_weighted_profile(interviewer) for interviewer in interviewers]
candidate_profiles = [create_weighted_profile(candidate) for candidate in candidates]

# Vectorize profiles using TF-IDF
vectorizer = TfidfVectorizer()
interviewer_vectors = vectorizer.fit_transform(interviewer_profiles)
candidate_vectors = vectorizer.transform(candidate_profiles)

# Calculate cosine similarity between interviewers and candidates
similarity_matrix = cosine_similarity(candidate_vectors, interviewer_vectors)

# Display similarity results with interviewers as the main reference
for i, interviewer in enumerate(interviewers):
    print(f"Matching results for {interviewer['name']}:")
    for j, candidate in enumerate(candidates):
        print(f"  {candidate['name']}: {similarity_matrix[i, j]:.2f}")
    print()
