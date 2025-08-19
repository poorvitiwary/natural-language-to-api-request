# Natural Language to API Request

This project demonstrates how to generate structured API request bodies from unstructured natural language input.

The system extracts key details such as product models, configuration options, and delivery dates from user prompts, and uses them to build a valid API request body.

It combines **custom Named Entity Recognition (NER)** with SpaCy and regex/date parsing to showcase natural language processing, rule-based extraction, and request body generation.

---

## Features
- Extracts models and configurations from natural language text.  
- Identifies and normalizes dates from multiple formats.  
- Builds structured request bodies for downstream APIs.  
- Includes error handling and input validation.  
- Allows optional modification of generated request bodies.  
- Comes with unit tests for key functions.  

---

## Prerequisites
- Python 3.7+  
- [`spacy`](https://spacy.io/)  
- [`nltk`](https://www.nltk.org/)  
- `random`, `re`, `datetime`, `dateutil` (standard libraries + external for parsing).  

---

##  Getting Started

1. **Clone this repository**  
- git clone https://github.com/yourusername/natural-language-to-api-request.git
  cd natural-language-to-api-request

2. **Install dependencies**
- pip install spacy nltk python-dateutil
- import nltk
  nltk.download("stopwords")

3. **Run the main file**
- python main.py

4. **Testing**
- python -m unittest test.py

##  Customization
- Can be done by providing new statement samples and providing the location of new target entity.


