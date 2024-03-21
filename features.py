import email
import mailbox
import re
from email import message_from_string
from email.message import EmailMessage
from email.utils import getaddresses

import spacy
from bs4 import BeautifulSoup
from spacy.matcher import Matcher
from spellchecker import SpellChecker
import readability



def extract(email, nlp, matcher, url_pattern, script_pattern):
    features = {}

    email_message, email_body, email_subject, email_recipients, soup = clean_and_parse_email(email)
    html_content, plain_text_content = extract_html_and_plain_text(email_message)
    body_doc = nlp(email_body)
    body_matches = matcher(body_doc)
    features['has_sensitive_info'], features['sensitive_info_phrases'] = contains_sensitive_info_request(nlp, body_doc,
                                                                                                         body_matches)
    features['has_imperatives'], features['imperative_phrases'] = contains_imperatives(body_doc)
    features['has_urgency'], features['urgency_phrases'] = contains_urgency(nlp, body_matches, body_doc)
    features['spelling_errors'] = contains_spelling_errors(email_body)
    features['has_generic_greeting'] = contains_generic_greeting(nlp, body_matches)
    features['subject_length'] = len(str(email_subject))
    features['body_length'] = len(email_body)
    features['has_html'] = bool(html_content)
    features['has_javascript'] = contains_javascript(email_message, soup, script_pattern, html_content)
    features['has_links'], features['links_count'] = extract_link_features(email_message, soup, url_pattern,
                                                                           html_content, plain_text_content)
    features['num_of_recipients'] = len(email_recipients)
    features['has_attachments'], features['attachments_count'] = extract_attachment_features(email_message)
    if plain_text_content.strip():
        features['readability_score'] = readability.getmeasures(plain_text_content, lang='en')['readability grades'][
            'FleschReadingEase']
    else:
        features['readability_score'] = None
    return features


def clean_and_parse_email(input_data):
    # Check if input_data is an mbox message or a standard email message
    if isinstance(input_data, (mailbox.mboxMessage, EmailMessage, email.message.Message)):
        # Directly use the email object as it is already a Message object
        email_message = input_data
    elif isinstance(input_data, str):
        # Directly parse the string to EmailMessage
        email_message = message_from_string(input_data)
    else:
        raise ValueError("Unsupported email input type.")

    # Extracting recipients
    to_recipients = email_message.get_all('To', [])
    cc_recipients = email_message.get_all('Cc', [])
    bcc_recipients = email_message.get_all('Bcc', [])
    all_recipients = set(getaddresses(to_recipients + cc_recipients + bcc_recipients))
    # Extract the subject and body
    subject = email_message['Subject']
    body = ""
    if email_message.is_multipart():
        for part in email_message.walk():
            if part.get_content_type() in ["text/plain", "text/html"]:
                try:
                    charset = part.get_content_charset() or 'utf-8'
                    body = part.get_payload(decode=True).decode(charset, errors='ignore')
                    break
                except LookupError:
                    print(f"Unknown encoding {charset}, using 'utf-8' as fallback.")
                    body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                    break
    else:
        try:
            charset = email_message.get_content_charset() or 'utf-8'
            body = email_message.get_payload(decode=True).decode(charset, errors='ignore')
        except LookupError:
            print(f"Unknown encoding {charset}, using 'utf-8' as fallback.")
            body = email_message.get_payload(decode=True).decode('utf-8', errors='ignore')

    # Clean HTML from the body if present
    soup = BeautifulSoup(body, 'html.parser')
    clean_body = soup.get_text(separator=' ')

    # Return subject and clean body
    return email_message, clean_body, subject, all_recipients, soup


def decode_payload(part):
    try:
        charset = part.get_content_charset() or 'utf-8'
        return part.get_payload(decode=True).decode(charset, 'ignore')
    except LookupError:
        return part.get_payload(decode=True).decode('utf-8', 'ignore')


def extract_html_and_plain_text(email_message):
    html_content = ""
    plain_text_content = ""

    if email_message.is_multipart():
        for part in email_message.walk():
            if part.get_content_type() == 'text/html':
                html_content += decode_payload(part)
            elif part.get_content_type() == 'text/plain':
                plain_text_content += decode_payload(part)
    else:
        charset = email_message.get_content_charset() or 'utf-8'
        try:
            if email_message.get_content_type() == 'text/html':
                html_content = email_message.get_payload(decode=True).decode(charset, 'ignore')
            elif email_message.get_content_type() == 'text/plain':
                plain_text_content = email_message.get_payload(decode=True).decode(charset, 'ignore')
        except LookupError:
            html_content = email_message.get_payload(decode=True).decode('utf-8', 'ignore')

    return html_content, plain_text_content


# 1. Sensitive Information Request Detection
sensitive_info_patterns = [
    # Personal Information
    [{"LOWER": "full"}, {"LOWER": "name"}],
    [{"LOWER": "date"}, {"LOWER": "of"}, {"LOWER": "birth"}],
    [{"LOWER": "address"}],
    [{"LOWER": "phone"}, {"LOWER": "number"}],
    [{"LOWER": "email"}, {"LOWER": "address"}],
    [{"LOWER": "account"}, {"LOWER": "details"}],
    [{"LOWER": "account"}, {"LOWER": "information"}],
    [{"LOWER": "your"}, {"LOWER": "account"}],
    [{"LOWER": "social"}, {"LOWER": "security"}, {"LOWER": "number"}, {"IS_PUNCT": True, "OP": "?"}],
    [{"LOWER": "password"}],

    # Financial Information
    [{"LOWER": "bank"}, {"LOWER": "account"}],
    [{"LOWER": "routing"}, {"LOWER": "number"}],
    [{"LOWER": "credit"}, {"LOWER": "card"}],
    [{"LOWER": "debit"}, {"LOWER": "card"}],
    [{"LOWER": "card"}, {"LOWER": "expiry"}, {"LOWER": "date"}],
    [{"LOWER": "cvv"}],
    [{"LOWER": "pin"}],

    # Security Information
    [{"LOWER": "security"}, {"LOWER": "question"}, {"LOWER": "answer"}],
    [{"LOWER": "verification"}, {"LOWER": "code"}],
    [{"LOWER": "login"}, {"LOWER": "details"}],
    [{"LOWER": "username"}, {"LOWER": "and"}, {"LOWER": "password"}],
    [{"LOWER": "authenticate"}, {"LOWER": "your"}, {"LOWER": "account"}],

    # Urgent Requests
    [{"LOWER": "confirm"}, {"LOWER": "identity"}],
    [{"LOWER": "verify"}, {"LOWER": "account"}],
    [{"LOWER": "verify"}, {"LOWER": "information"}],
    [{"LOWER": "update"}, {"LOWER": "account"}],
    [{"LOWER": "update"}, {"LOWER": "payment"}, {"LOWER": "info"}],
    [{"LOWER": "immediate"}, {"LOWER": "action"}, {"LOWER": "required"}],

    # Phrases that might be used to trick users into providing information
    [{"LOWER": "unlock"}, {"LOWER": "your"}, {"LOWER": "account"}],
    [{"LOWER": "secure"}, {"LOWER": "your"}, {"LOWER": "account"}],
    [{"LOWER": "account"}, {"LOWER": "suspension"}, {"LOWER": "notice"}],
    [{"LOWER": "compliance"}, {"LOWER": "verification"}],
]


def contains_sensitive_info_request(nlp, doc, matches):
    # Retrieve the match ID for 'SENSITIVE_INFO_REQUEST' to filter matches
    sensitive_info_request_id = nlp.vocab.strings['SENSITIVE_INFO_REQUEST']

    # Initialize an empty list to hold the text of the matched spans
    sensitive_info_matches = []

    # Check if any matches belong to 'SENSITIVE_INFO_REQUEST'
    for match_id, start, end in matches:
        if match_id == sensitive_info_request_id:
            # Add the matched span text to the list
            sensitive_info_matches.append(doc[start:end].text)

    # Return True if there are sensitive info request matches found, and the list of matched spans
    return len(sensitive_info_matches) > 0, sensitive_info_matches


# 2. Imperative Detection
def contains_imperatives(doc):
    imperatives = []
    for sent in doc.sents:
        verb = sent.root
        # Checking verb form and excluding sentences with explicit non-"you" subjects
        if verb.pos_ == "VERB" and verb.tag_ == "VB" and not any(
                token.dep_ == "nsubj" and token.text.lower() != "you" for token in sent):
            # Further filter out sentences with modal verbs as roots, though this may exclude some imperatives
            if not verb.text.lower() in ["can", "could", "may", "might", "must", "shall", "should", "will", "would"]:
                # Check if there are any children that could indicate an imperative structure
                if any(child.dep_ in ("dobj", "xcomp", "advmod", "npadvmod") for child in verb.children):
                    imperatives.append(sent.text)
    return len(imperatives) > 0, imperatives


# 3. Urgency Detection
urgency_keywords = [
    "immediate", "now", "urgent", "important", "action required",
    "as soon as possible", "asap", "right away", "at your earliest convenience",
    "prompt", "don't delay", "hurry", "quick", "immediately",
    "deadline", "limited time", "expire", "final notice", "last chance",
    "today only", "time sensitive", "time-sensitive", "critical",
    "rush", "warning", "alert", "instant", "immediate action",
    "requires your attention", "requires immediate attention",
    "don’t miss out", "before it’s too late"
]


def contains_urgency(nlp, matches, doc):
    urgency_matches = []

    # Iterate through matches to extract the text manually
    for match_id, start, end in matches:
        span = doc[start:end]  # Get the span from the doc
        if nlp.vocab.strings[match_id] == "URGENCY":
            urgency_matches.append(span.text)
    return len(urgency_matches) > 0, urgency_matches


def contains_spelling_errors(text):
    # Initialize the spell checker
    spell = SpellChecker()

    # Tokenize the text into words
    words = re.findall(r'\b[a-z]+\b', text.lower())  # This regex will match words and ignore punctuation and numbers

    # Find words that are misspelled
    misspelled = spell.unknown(words)

    return misspelled


generic_greeting_patterns = [
    # Simple greetings
    [{"LOWER": "hi"}], [{"LOWER": "hello"}], [{"LOWER": "hey"}],
    [{"LOWER": "greetings"}], [{"LOWER": "dear"}, {"IS_ALPHA": True, "OP": "+"}],
    [{"LOWER": "good"}, {"LOWER": {"IN": ["morning", "afternoon", "evening", "day"]}}],

    # Formal greetings
    [{"LOWER": "dear"}, {"LOWER": "sir"}], [{"LOWER": "dear"}, {"LOWER": "madam"}],
    [{"LOWER": "to"}, {"LOWER": "whom"}, {"LOWER": "it"}, {"LOWER": "may"}, {"LOWER": "concern"}],

    # Greetings with titles and names
    [{"LOWER": "mr."}, {"IS_ALPHA": True}], [{"LOWER": "ms."}, {"IS_ALPHA": True}],
    [{"LOWER": "mrs."}, {"IS_ALPHA": True}], [{"LOWER": "dr."}, {"IS_ALPHA": True}],

    # Multilingual greetings
    [{"LOWER": {"IN": ["hola", "bonjour", "hallo", "ciao", "こんにちは", "안녕하세요"]}}],

    # Email-specific greetings
    [{"LOWER": {"IN": ["dear", "attention", "hello", "hi"]}}, {"IS_PUNCT": True, "OP": "?"},
     {"IS_SPACE": True, "OP": "*"}, {"LOWER": "all"}],
    [{"LOWER": {"IN": ["dear", "attention", "hello", "hi"]}}, {"LOWER": "team"}],
    [{"LOWER": {"IN": ["dear", "attention", "hello", "hi"]}}, {"LOWER": "valued"}, {"LOWER": "customer"}],
    [{"LOWER": {"IN": ["dear", "attention", "hello", "hi"]}}, {"LOWER": "valued"}, {"LOWER": "client"}],

    # Greetings that may appear in phishing
    [{"LOWER": {"IN": ["urgent", "important", "immediate"]}}, {"IS_SPACE": True, "OP": "*"}, {"LOWER": "attention"}],
    [{"LOWER": "attention"}, {"LOWER": {"IN": ["required", "needed"]}}],
]


# Add the patterns to the matcher


def contains_generic_greeting(nlp, matches):
    # Retrieve the match ID for "GREETINGS" patterns
    greetings_id = nlp.vocab.strings["GREETINGS"]

    # Check if any matches belong to 'GREETINGS'
    for match_id, start, end in matches:
        if match_id == greetings_id:
            return True  # A greeting match was found

    return False  # No greeting matches found


def contains_javascript(email_message, soup, script_pattern, html_content):
    # Check for standard script tags
    if soup.find('script') is not None:
        return True

    # Check for obfuscated or malformed script tags
    if script_pattern.search(html_content):
        return True

    # Check for JavaScript event handlers
    for tag in soup.find_all(True):
        if any(attr in tag.attrs for attr in ['onload', 'onclick', 'onerror', 'onmouseover']):
            return True

    return False


def extract_link_features(email_message, soup, url_pattern, html_content, plain_text_content):
    links_count = 0

    # Use BeautifulSoup to parse HTML content and find <a> tags
    if html_content:
        links_count += len(soup.find_all('a', href=True))

    plain_text_links = url_pattern.findall(plain_text_content)
    links_count += len(plain_text_links)

    # Determine if links are present and count them
    has_links = links_count > 0

    return has_links, links_count


def extract_attachment_features(email_message):
    attachments_count = 0
    # Check if the email is multipart (attachments are in separate parts)
    if email_message.is_multipart():
        for part in email_message.walk():
            # The Content-Disposition header can be 'attachment' or 'inline' (for attachments shown directly in the
            # email body)
            if part.get_content_maintype() != 'multipart' and part.get("Content-Disposition") is not None:
                attachments_count += 1

    return attachments_count > 0, attachments_count
