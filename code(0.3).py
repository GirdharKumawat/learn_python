# Version: 0.3

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
embed = hub.load("https://www.kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/universal-sentence-encoder/2")

# Sample data
candidates = [
    {
        "name": "Priya Iyer",
        "education": {
            "bachelor": {
                "degree": "B.Tech",
                "field": "Computer Science",
                "institute": "IIT Bombay"
            },
            "master": {
                "degree": "M.Tech",
                "field": "Artificial Intelligence",
                "institute": "IIT Delhi"
            }
        },
        "area_of_expertise": {
            "fields": ["Artificial Intelligence", "Deep Learning", "Natural Language Processing"],
            "specialization_level": "Intermediate"
        },
        "experience_years": 3,
        "research_projects": ["AI for Social Good", "NLP for Indian Languages"]
    },
    {
        "name": "Rahul Sharma",
        "education": {
            "bachelor": {
                "degree": "B.E.",
                "field": "Electronics and Communication",
                "institute": "BITS Pilani"
            },
            "master": {
                "degree": "M.S.",
                "field": "Robotics",
                "institute": "Carnegie Mellon University"
            },
            "phd": {
                "degree": "Ph.D.",
                "field": "Autonomous Systems",
                "institute": "IISc Bangalore"
            }
        },
        "area_of_expertise": {
            "fields": ["Robotics", "Computer Vision", "Sensor Fusion"],
            "specialization_level": "Advanced"
        },
        "experience_years": 7,
        "research_projects": ["Autonomous Drones for Defense", "Multi-agent Robotic Systems"]
    },
    {
        "name": "Anjali Desai",
        "education": {
            "bachelor": {
                "degree": "B.Tech",
                "field": "Aerospace Engineering",
                "institute": "IIT Madras"
            },
            "master": {
                "degree": "M.S.",
                "field": "Aeronautics and Astronautics",
                "institute": "MIT"
            }
        },
        "area_of_expertise": {
            "fields": ["Propulsion Systems", "Computational Fluid Dynamics", "Aerospace Structures"],
            "specialization_level": "Intermediate"
        },
        "experience_years": 5,
        "research_projects": ["Hypersonic Vehicle Design", "Advanced Propulsion Technologies"]
    },
    {
        "name": "Vikram Malhotra",
        "education": {
            "bachelor": {
                "degree": "B.Sc",
                "field": "Physics",
                "institute": "Delhi University"
            },
            "master": {
                "degree": "M.Sc",
                "field": "Nuclear Physics",
                "institute": "Jawaharlal Nehru University"
            },
            "phd": {
                "degree": "Ph.D.",
                "field": "Particle Physics",
                "institute": "TIFR Mumbai"
            }
        },
        "area_of_expertise": {
            "fields": ["Nuclear Physics", "Particle Detectors", "Radiation Shielding"],
            "specialization_level": "Expert"
        },
        "experience_years": 10,
        "research_projects": ["Next-gen Particle Detectors", "Nuclear Reactor Safety Systems"]
    },
    {
        "name": "Sneha Gupta",
        "education": {
            "bachelor": {
                "degree": "B.Tech",
                "field": "Mechanical Engineering",
                "institute": "NIT Trichy"
            },
            "master": {
                "degree": "M.Tech",
                "field": "Thermal Engineering",
                "institute": "IIT Kanpur"
            }
        },
        "area_of_expertise": {
            "fields": ["Heat Transfer", "Thermodynamics", "Energy Systems"],
            "specialization_level": "Intermediate"
        },
        "experience_years": 4,
        "research_projects": ["Thermal Management in Military Vehicles", "Energy-efficient Cooling Systems"]
    },
    {
        "name": "Arjun Nair",
        "education": {
            "bachelor": {
                "degree": "B.Tech",
                "field": "Computer Science",
                "institute": "IIIT Hyderabad"
            },
            "master": {
                "degree": "M.S.",
                "field": "Cybersecurity",
                "institute": "Georgia Tech"
            }
        },
        "area_of_expertise": {
            "fields": ["Network Security", "Cryptography", "Ethical Hacking"],
            "specialization_level": "Advanced"
        },
        "experience_years": 6,
        "research_projects": ["Quantum-resistant Encryption", "Secure Communication Protocols for Defense"]
    },
    {
        "name": "Meera Patel",
        "education": {
            "bachelor": {
                "degree": "B.E.",
                "field": "Chemical Engineering",
                "institute": "VJTI Mumbai"
            },
            "phd": {
                "degree": "Ph.D.",
                "field": "Nanotechnology",
                "institute": "IIT Kharagpur"
            }
        },
        "area_of_expertise": {
            "fields": ["Nanomaterials", "Smart Materials", "Sensors"],
            "specialization_level": "Expert"
        },
        "experience_years": 8,
        "research_projects": ["Nanotech-based Armor Materials", "Self-healing Composites for Military Applications"]
    },
    {
        "name": "Karthik Sundar",
        "education": {
            "bachelor": {
                "degree": "B.Tech",
                "field": "Electrical Engineering",
                "institute": "Anna University"
            },
            "master": {
                "degree": "M.Tech",
                "field": "Signal Processing",
                "institute": "IIT Guwahati"
            }
        },
        "area_of_expertise": {
            "fields": ["Digital Signal Processing", "Radar Systems", "Electronic Warfare"],
            "specialization_level": "Intermediate"
        },
        "experience_years": 5,
        "research_projects": ["Advanced Radar Signal Processing", "Cognitive Electronic Warfare Systems"]
    },
    {
        "name": "Zoya Khan",
        "education": {
            "bachelor": {
                "degree": "B.Sc",
                "field": "Biotechnology",
                "institute": "University of Mumbai"
            },
            "master": {
                "degree": "M.Sc",
                "field": "Genetics",
                "institute": "University of Pune"
            },
            "phd": {
                "degree": "Ph.D.",
                "field": "Molecular Biology",
                "institute": "NCBS Bangalore"
            }
        },
        "area_of_expertise": {
            "fields": ["Genetic Engineering", "Biosensors", "Bioinformatics"],
            "specialization_level": "Expert"
        },
        "experience_years": 9,
        "research_projects": ["Biosensors for Chemical/Biological Warfare Agents", "Genetic Engineering for Enhanced Stress Resistance"]
    },
    {
        "name": "Rajesh Venkatesh",
        "education": {
            "bachelor": {
                "degree": "B.Tech",
                "field": "Metallurgical Engineering",
                "institute": "NIT Rourkela"
            },
            "master": {
                "degree": "M.Tech",
                "field": "Materials Science",
                "institute": "IIT Roorkee"
            }
        },
        "area_of_expertise": {
            "fields": ["Advanced Materials", "Composite Materials", "Failure Analysis"],
            "specialization_level": "Advanced"
        },
        "experience_years": 7,
        "research_projects": ["High-strength Lightweight Alloys for Defense", "Advanced Composite Materials for Aerospace"]
    }
]

experts = [
    {
        "name": "Dr. Radhika Menon",
        "domain_expertise": {
            "fields": ["Data Science", "Artificial Intelligence", "Machine Learning"],
            "specialization_level": "Intermediate"
        },
        "experience": {
            "years": 15,
            "roles": ["Research Scientist", "Data Science Lead at TCS"],
            "industries": ["IT", "Healthcare"]
        },
        "education": {
            "bachelor": {
                "degree": "B.Tech",
                "field": "Computer Science",
                "institute": "IIT Madras"
            },
            "master": {
                "degree": "M.Tech",
                "field": "Artificial Intelligence",
                "institute": "IISc Bangalore"
            },
            "phd": {
                "degree": "PhD",
                "field": "Computer Science",
                "institute": "IISc Bangalore"
            }
        },
        "publications": 25,
        "previous_interview_experience": 5,
        "industry_projects": ["AI for Healthcare at Apollo Hospitals", "TCS Data Analytics"]
    },
    {
        "name": "Prof. Arun Kumar",
        "domain_expertise": {
            "fields": ["Aerospace Engineering", "Propulsion Systems", "Fluid Dynamics"],
            "specialization_level": "Expert"
        },
        "experience": {
            "years": 20,
            "roles": ["Professor at IIT Bombay", "Consultant for ISRO"],
            "industries": ["Academia", "Space Technology"]
        },
        "education": {
            "bachelor": {
                "degree": "B.Tech",
                "field": "Aerospace Engineering",
                "institute": "IIT Kanpur"
            },
            "master": {
                "degree": "M.S.",
                "field": "Aeronautics and Astronautics",
                "institute": "Stanford University"
            },
            "phd": {
                "degree": "PhD",
                "field": "Aerospace Engineering",
                "institute": "Caltech"
            }
        },
        "publications": 40,
        "previous_interview_experience": 10,
        "industry_projects": ["Next-gen Rocket Propulsion for ISRO", "Hypersonic Vehicle Design"]
    },
    {
        "name": "Dr. Sunita Sharma",
        "domain_expertise": {
            "fields": ["Cybersecurity", "Network Security", "Cryptography"],
            "specialization_level": "Advanced"
        },
        "experience": {
            "years": 18,
            "roles": ["Senior Scientist at DRDO", "Visiting Faculty at IIT Delhi"],
            "industries": ["Defense", "Academia"]
        },
        "education": {
            "bachelor": {
                "degree": "B.E.",
                "field": "Electronics and Communication",
                "institute": "Delhi College of Engineering"
            },
            "master": {
                "degree": "M.Tech",
                "field": "Computer Science",
                "institute": "IIT Delhi"
            },
            "phd": {
                "degree": "PhD",
                "field": "Information Security",
                "institute": "IIT Delhi"
            }
        },
        "publications": 30,
        "previous_interview_experience": 15,
        "industry_projects": ["Secure Communication Systems for Armed Forces", "Quantum Cryptography Research"]
    }
]


# Get the embedding for given text
def get_embedding(text):
    return embed([text])


# Calculate the cosine similarity between two embeddings
def calculate_similarity(text1, text2):
    vector1 = get_embedding(text1)
    vector2 = get_embedding(text2)

    return cosine_similarity(vector1, vector2)[0][0]


# Calculate field similarity
def calculate_field_similarity(candidate, expert):
    candidate_sentence = ' '.join(candidate['area_of_expertise']['fields'])
    expert_sentence = ' '.join(expert['domain_expertise']['fields'])

    return calculate_similarity(candidate_sentence, expert_sentence)


# Calculate education score
def calculate_education_score(candidate, expert):
    candidate_education_degrees = []
    expert_education_degrees = []

    candidate_education_fields = []
    expert_education_fields = []

    for candidate_education in candidate['education'].values():
        candidate_education_degrees.append(candidate_education['degree'])
        candidate_education_fields.append(candidate_education['field'])

    for expert_education in expert['education'].values():
        expert_education_degrees.append(expert_education['degree'])
        expert_education_fields.append(expert_education['field'])

    degree_similarity = calculate_similarity(' '.join(candidate_education_degrees), ' '.join(expert_education_degrees))
    field_similarity = calculate_similarity(' '.join(candidate_education_fields), ' '.join(expert_education_fields))

    return (degree_similarity + field_similarity) / 2


# Calculate project similarity
def calculate_project_similarity(candidate, expert):
    candidate_projects = ' '.join(candidate.get('research_projects', []))
    expert_projects = ' '.join(expert.get('industry_projects', []))

    return calculate_similarity(candidate_projects, expert_projects)


# Return the matching score between candidate and expert
def calculate_matching_score(candidate, expert):
    # Field similarity
    field_similarity = calculate_field_similarity(candidate, expert)

    # Education similarity
    education_similarity = calculate_education_score(candidate, expert)

    # Project similarity
    project_similarity = calculate_project_similarity(candidate, expert)

    # Total score
    total_score = field_similarity * 0.4 + education_similarity * 0.3 + project_similarity * 0.3

    return {
        'name': candidate['name'],
        'field_similarity': field_similarity,
        'education_similarity': education_similarity,
        'project_similarity': project_similarity,
        'total_score': total_score
    }

for expert in experts:
    matched_candidates = []

    for candidate in candidates:
        matched_candidate = calculate_matching_score(candidate, expert)
        if matched_candidate['total_score'] >= 0.5:
            matched_candidates.append(matched_candidate)

    # Sort the matched candidates based on match score
    matched_candidates = sorted(matched_candidates, key=lambda x: x['total_score'], reverse=True)

    # Display the matched candidates
    print(f"For {expert['name']}: \n")
    for matched_candidate in matched_candidates:
        print(f"Name: {matched_candidate['name']}")
        print(f"Total Score: {int(matched_candidate['total_score']*100)}")
        print(f"Field Similarity: {int(matched_candidate['field_similarity']*100)}")
        print(f"Education Similarity: {int(matched_candidate['education_similarity']*100)}")
        print(f"Project Similarity: {int(matched_candidate['project_similarity']*100)} \n")