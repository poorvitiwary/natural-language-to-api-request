import nltk
from nltk import sent_tokenize
import re
from datetime import datetime
from dateutil import parser
import spacy
import random
from spacy.util import minibatch, compounding
from pathlib import Path
from spacy.training.example import Example
nltk.download("stopwords")

nlp = spacy.load('en_core_web_sm')
car_model_regex = re.compile(r'iX xDrive50|iX xDrive40|X7 xDrive40i|X7 xDrive40d|M8|318i', re.IGNORECASE)
date_regex = re.compile(r"\b(?:(?:start|end|mid) of (?:January|February|March|April|May|June|July|August|September|October|November|December) \d{4}|\d{1,2}(?:st|nd|rd|th)? (?:of )?(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{4}|\d{2}/\d{2}/\d{4}|\d{2}-\d{2}-\d{4})\b",re.IGNORECASE)
train= [("I want to order a BMW iX xDrive50 with right-hand drive configuration. I will be ordering it at the start of October 2022.",{"entities" :[(22,33,"Sales Description"),(34,55,"Steering Wheel Configuration")]}),
       ("Hello, is the X7 xDrive40i available without a panorama glass roof and with the EU Comfort Package. I need the vehicle on the 8th of November 2024.",{"entities": [(14,26,"Sales Description"),(37,66,"Roof Configuration"),(71,98,"Available Packages")]}),
       ("I am planning to order the BMW M8 with a sunroof or panorama glass roof sky lounge, and the M Sport Package on 12th April 2018. Is this configuration possible?",{"entities" :[(31,33,"Sales Description"),(34,48,"Roof Configuration"),(49,82,"Roof Configuration"),(84,107,"Available Packages")]}),
       ("Can you provide a quote for a X7 xDrive40d with the Comfort Package EU and sunroof? I am looking to purchase it by the start of October 2023.",{"entities":[(30,42,"Sales Description"),(43,70,"Available Packages"),(71,82,"Roof Configuration")]}),
       ("I wish to buy BMW 318i without sunroof and left-hand drive, any package would work except m sport package and eu comfort package.",{"entities":[(18,22,"Sales Description"),(23,38,"Roof Configuration"),(39,58,"Steering Wheel Configuration"),(83,105,"Available Packages"),(106,128,"Available Packages")]}),
       ("I wish to buy BMW X7 xDrive40i the car should have left-hand drive with sunroof.",{"entities":[(18,30,"Sales Description"),(46,66,"Steering Wheel Configuration"),(67,79,"Roof Configuration")]

def validate_input(text):
    # Check if the prompt contains a valid model type code
    match = re.search(car_model_regex, text)

    if match is None:
        raise ValueError("Please enter a valid BMW car model.")

    # Check if the prompt contains a valid date
    match = re.search(date_regex, text)
    if match is None:
        raise ValueError("Please enter a valid date for product delivery.")


    # If all checks pass, return True
    return True


def getmodel(text):
    #Dictionary that maps car models to model type code
    model_code_mapping = {
        'ix xdrive50': '21CF',
        'ix xdrive40': '11CF',
        'x7 xdrive40i': '21EM',
        'x7 xdrive40d': '21EN',
        'm8': 'DZ01',
        '318i': '28FF'
    }

    doc = nlp(text)

    target_label = "Sales Description"  # Specify the label of the entities you want to extract from the trained model
    target_entities = []

    for ent in doc.ents:
        if ent.label_ == target_label:                 #Extract the matching entities to a list
            target_entities.append(ent.text)

    for key in target_entities:
        if key.lower() in model_code_mapping:           #match the extracted entity(car model) to the model code
            model_code = model_code_mapping[key.lower()]

        else:
            model_code = None

    return model_code


def getdate(text):

    #Extracting date from the prompt if it is in numerics
    dates = re.findall(date_regex, text)
    dates = ', '.join(dates)
    dates = dates.replace('th', '')
    dates = dates.replace('of', '')

    if dates:

        dt = datetime.strptime(dates, '%d %B %Y')
        dates = dt.strftime('%Y-%m-%d')

    else:
        # Extracting date from the prompt if it is in "Start/end/mid" of the month format
        pattern = re.compile(r"(start|end|mid) of (\w+) (\d{4})")

        match = pattern.search(text)
        if match:
            date_string = match.group(0)
            date_string = date_string.replace('th', '')
            date_string = date_string.replace('of', '')
            date_obj = date_string.split()
            month_mapping = {
                "january": "01",
                "february": "02",
                "march": "03",
                "april": "04",
                "may": "05",
                "june": "06",
                "july": "07",
                "august": "08",
                "september": "09",
                "october": "10",
                "november": "11",
                "december": "12"
            }
            year = date_obj[2]
            month = date_obj[1]
            if month in month_mapping:
                mont = month_mapping[month]

            date = date_obj[0]
            pos = ["start", "mid", "end"]

            for x in pos:
                if date == pos[0]:
                    final_date_string = f"{year}-{mont}-01"
                    dates = final_date_string
                elif date == pos[1]:
                    final_date_string = f"{year}-{mont}-15"
                    dates = final_date_string
                else:
                    final_date_string = f"{year}-{mont}-30"
                    dates = final_date_string


    return dates


def train_model(train_data, iterations):
    # create the Named Entity Recognition pipeline
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")

    # add labels for the custom entities to the NER pipeline
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # disable all other pipelines except NER during training
    disable_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*disable_pipes):
        # initialize the optimizer
        optimizer = nlp.begin_training()

        # train the model
        for i in range(iterations):
            random.shuffle(train_data)
            losses = {}
            batches = minibatch(train_data, size=compounding(1.0, 5.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                examples = []
                for i in range(len(texts)):
                    doc = nlp.make_doc(texts[i])
                    examples.append(Example.from_dict(doc, annotations[i]))
                nlp.update(examples, drop=0.5, sgd=optimizer, losses=losses)

    return nlp

def get_configurations(text):
    doc = nlp(text)
    list = []
    entities_to_drop = ["Sales Description"]  #here we only deal with information used to generate boolean formula therefore we remove car models
    for ent in doc.ents:
        if ent.label_ not in entities_to_drop:
            list.append(ent.text)

    # download the stopwords corpus from NLTK
    nltk.download("stopwords")

    # create a custom list of stop words to not remove
    custom_stopwords = ["or", "not", "and", "with", "without", "m","have"]

    # get the default stop words set
    default_stopwords = set(nltk.corpus.stopwords.words("english"))

    # remove the custom stop words from the default set
    final_stopwords = default_stopwords.difference(custom_stopwords)

    # use the final stop words set for your text processing tasks
    filtered_list = []
    for sentence in list:
        filtered_sentence = [
            word for word in sentence.split() if word.lower() not in final_stopwords
        ]
        filtered_list.append(" ".join(filtered_sentence))

    final = [i.split(" ", 1) for i in filtered_list] #seprate  the connecting word (and/or etc) from the entity and map to generate final formula
    new_list = [word for sublist in final for word in sublist]
    configurations = {
        "left-hand drive": "LL",
        "right-hand drive": "RL",
        "m sport package": "P337A",
        "m sport package pro": "P33BA",
        "eu comfort package": "P7LGA",
        "panorama glass roof": "S402A",
        "panorama glass roof sky lounge": "S407A",
        "sunroof": "S403A",
        "with": "+",
        "without": "-",
        "or": "/",
        "and": "+",
        "valid": "+",
        "except": "+-",
        "all" : "+"
    }

    result = []
    for key in new_list:
        if key in configurations:
            result.append(configurations[key])

    bracket_added = False         #to add precedence of and/or to the formula

    for item in result:
        if item == '/':
            index = result.index('/')
            # Add an item before the specified index
            result.insert(index - 1, '(')
            # Add an item after the specified index
            result.insert(index + 3, ')')
            bracket_added = True
            break

    return result



def modify_request_body(request_body):
    """
    Allows the user to modify the request body. If the user wants to edit the request directly before sending it they can add the new model code, date or boolen formula in the format valid in the request.
    """
    while True:
        print("Current request body:")
        print(request_body)
        modify = input("Would you like to modify the request body? (y/n)")
        if modify.lower() == 'n':
            break
        elif modify.lower() == 'y':
            field = input("Which field would you like to modify? (model_type, boolean_formula, date)")
            if field == 'model_type':
                model_update = input("Enter the new model type code:")
                request_body['model_type'] = model_update

            elif field == 'boolean_formula':
                boolean_formula = input("Enter the new boolean formula:")
                request_body['boolean_formula'] = boolean_formula
            elif field == 'date':
                date = input("Enter the new delivery date (YYYY-MM-DD):")
                request_body['dates'] = date
            else:
                print("Invalid field name.")
        else:
            print("Invalid input.")

    return request_body

def generate_request_body(text):
    #Helps generate final request body by calling different functions responsible.
    r = get_configurations(text)
    boolean = ''.join(r)
    model = getmodel(text)
    date = getdate(text)
    request_body = {
        "modelTypeCodes": f"[{model}]",
        "booleanFormulas": f"[{boolean}]",
        "dates": f"[{date}]"
    }

    return request_body

def main():
        text = input("Enter your text:")
        text = text.lower()
        nlp = train_model(train, 100)

        # Generate a query using the sentence as the prompt
        try:              #provides error handling for invalid input
            validate_input(text)
            output = generate_request_body(text)
            print(output)
        except ValueError as e:
            print(e)
            main()
        modify_request_body(output) #provides option to edit the query directly


        generate_another_request = input("Would you like to generate another request body? (Y/N)").upper()  #Helps generate another query

        if generate_another_request == "N":
            print("end") #enables program exit
        else:
            text1 = input("Enter your text:")
            text1 = text1.lower()
            output1 = generate_request_body(text1)
            print(output1)

main()
