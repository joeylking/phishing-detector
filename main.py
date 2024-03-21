import mailbox
import os
import random
import re
import time
import pickle

import spacy
# from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.svm import SVC
from spacy.matcher import Matcher
from xgboost import XGBClassifier

import features


def load_emails_from_directory(directory_path, emails):
    """
    Recursively load emails from a directory and its subdirectories.

    Args:
    - directory_path: The path to the directory to load emails from.
    - emails: A list to append the loaded email contents to.
    """
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        if os.path.isdir(item_path):
            # If the item is a directory, recurse into it
            load_emails_from_directory(item_path, emails)
        else:
            # If the item is a file, assume it's an email and try to read it
            try:
                with open(item_path, 'r', encoding='latin1') as email_file:
                    email_content = email_file.read()
                    emails.append(email_content)
                    # print(f"Email added from {directory_path} : {item_path}")
            except Exception as e:
                print(f"Error reading {item_path}: {e}")


def load_enron_dataset(maildir_root, folders_to_include=None, sample_fraction=0.5):
    """
    Load emails from the Enron dataset, navigating through subdirectories recursively.

    Args:
    - maildir_root: The root directory of the Enron maildir dataset.
    - folders_to_include: A list of folder names to include (e.g., ['inbox', 'sent_mail']). If None, all folders are
    included.
    - sample_fraction: The fraction of emails to sample from each folder (0.5 for half).
    Returns:
    - A list of email contents.
    """
    emails = []

    # Iterate through all user directories
    for person_dir in os.listdir(maildir_root):
        person_path = os.path.join(maildir_root, person_dir)
        if os.path.isdir(person_path):
            # If folders_to_include is specified, only look into those folders
            if folders_to_include:
                for folder_name in folders_to_include:
                    specific_path = os.path.join(person_path, folder_name)
                    if os.path.exists(specific_path):
                        load_emails_from_directory(specific_path, emails)
            else:
                # If no specific folders are specified, recursively load emails from all directories
                load_emails_from_directory(person_path, emails)
    print(f"Loaded {len(emails)} emails from specified folders.")
    # Sample a subset of emails randomly
    sampled_emails = random.sample(emails, int(len(emails) * sample_fraction))

    print(f"Loaded {len(sampled_emails)} emails from specified folders.")
    return sampled_emails


def load_nazario_phishing_dataset(nazario_dir,sample_fraction=0.5):
    print("Loading phishing dataset from:", nazario_dir)
    phishing_emails = []
    for root, dirs, files in os.walk(nazario_dir):
        for file in files:
            file_path = os.path.join(root, file)
            print("Processing file:", file_path)
            if file.endswith('.mbox'):
                try:
                    mbox = mailbox.mbox(file_path)
                    for message in mbox:
                        phishing_emails.append(message)
                except UnicodeDecodeError as e:
                    print(f"Unicode decode error in file {file_path}: {e}")
            else:
                print("not mbox file")
    print(f"Loaded {len(phishing_emails)} emails from specified folders.")

    # Sample a subset of emails randomly
    sampled_emails = random.sample(phishing_emails, int(len(phishing_emails) * sample_fraction))
    return sampled_emails


def safe_print_email(email_message):
    try:
        # Attempt to print the email as a string, assuming it's correctly encoded
        print(email_message.as_string())
    except UnicodeEncodeError:
        # If an error occurs, handle it by encoding the payload directly
        if email_message.is_multipart():
            for part in email_message.walk():
                charset = part.get_content_charset() or 'utf-8'
                try:
                    print(part.get_payload(decode=True).decode(charset, errors='replace'))
                except UnicodeError:
                    print(part.get_payload(decode=True).decode('utf-8', errors='replace'))
        else:
            charset = email_message.get_content_charset() or 'utf-8'
            try:
                print(email_message.get_payload(decode=True).decode(charset, errors='replace'))
            except UnicodeError:
                print(email_message.get_payload(decode=True).decode('utf-8', errors='replace'))


def extract_features_with_logging(dataset, index, email, nlp, matcher, url_pattern, script_pattern):
    print(f"Extracting features from email #{index} of {dataset} dataset")
    return features.extract(email, nlp, matcher, url_pattern, script_pattern)


# Function to train and evaluate a model
def train_and_evaluate(model, name, train_vectors, train_labels, test_vectors, test_labels):
    training_start_time = time.time()
    print(f"Training {name}...")
    dense_required = ['G Naive Bayes', 'CatBoost', 'Hist Gradient Boosting']

    if name in dense_required:
        # Convert to dense format for models that prefer dense data
        train_vectors_dense = train_vectors.toarray()
        test_vectors_dense = test_vectors.toarray()
        model.fit(train_vectors_dense, train_labels)
        predicted_labels = model.predict(test_vectors_dense)
    else:
        model.fit(train_vectors, train_labels)
        predicted_labels = model.predict(test_vectors)

    training_end_time = time.time()
    print(f"Training time: {training_end_time - training_start_time}")

    print(f"Evaluating {name}...")
    accuracy = accuracy_score(test_labels, predicted_labels)
    precision = precision_score(test_labels, predicted_labels)
    recall = recall_score(test_labels, predicted_labels)
    f1 = f1_score(test_labels, predicted_labels)

    print(f"{name} accuracy: {accuracy:.3f}")
    print(f"{name} precision: {precision:.3f}")
    print(f"{name} recall: {recall:.3f}")
    print(f"{name} F1 score: {f1:.3f}")

    # Save the trained model
    model_filename = f"{name.replace(' ', '_')}_model.pkl"
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Saved {name} model to {model_filename}")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Begin timer
    start_time = time.time()

    # Load emails
    enron_emails = load_enron_dataset('data/enron-data/maildir', ['inbox'], .5)
    nazario_emails = load_nazario_phishing_dataset('data/phishing-data/')
    print("All emails loaded")

    # Apply labels
    enron_labels = [0] * len(enron_emails)  # 0 for regular
    nazario_labels = [1] * len(nazario_emails)  # 1 for phishing

    # Split the datasets into training and testing sets
    enron_train, enron_test, enron_train_labels, enron_test_labels = train_test_split(enron_emails, enron_labels,
                                                                                      test_size=0.20,
                                                                                      random_state=42)
    nazario_train, nazario_test, nazario_train_labels, nazario_test_labels = train_test_split(nazario_emails,
                                                                                              nazario_labels,
                                                                                              test_size=0.20,
                                                                                              random_state=42)

    # Combine the training and testing sets
    train_emails = enron_train + nazario_train
    test_emails = enron_test + nazario_test
    train_labels = enron_train_labels + nazario_train_labels
    test_labels = enron_test_labels + nazario_test_labels

    # Define the file paths
    train_features_file = 'train_features.pkl'
    test_features_file = 'test_features.pkl'

    # Check if the pickle files exist and load them
    if os.path.exists(train_features_file) and os.path.exists(test_features_file):
        print("Loading features from pickle files...")
        with open(train_features_file, 'rb') as f:
            train_features = pickle.load(f)
        with open(test_features_file, 'rb') as f:
            test_features = pickle.load(f)
    else:
        # Extraction set up
        nlp = spacy.load('en_core_web_md')
        matcher = Matcher(nlp.vocab)
        matcher.add("SENSITIVE_INFO_REQUEST", features.sensitive_info_patterns)
        matcher.add("URGENCY", [[{"LOWER": keyword}] for keyword in features.urgency_keywords])
        matcher.add("GREETINGS", features.generic_greeting_patterns)
        url_pattern = re.compile(r'https?://[^\s]+')
        script_pattern = re.compile(r'<\w+\s+src\s*=\s*["\']https?://[^\s"\']+\.js["\']', re.IGNORECASE)
        # Extract features for training and testing sets
        feature_start_time = time.time()
        print("Extracting features and saving to pickle files...")
        # Your existing code to extract features
        # Assuming functions and setup are already defined and loaded:
        train_features = [extract_features_with_logging("training", train_emails.index(email), email, nlp, matcher,
                                                        url_pattern, script_pattern) for email in train_emails]
        test_features = [extract_features_with_logging("testing", test_emails.index(email), email, nlp, matcher,
                                                       url_pattern, script_pattern) for email in test_emails]
        feature_end_time = time.time()
        print(f"Feature extraction time: {feature_end_time - feature_start_time}")

        # Save the extracted features to pickle files
        with open(train_features_file, 'wb') as f:
            pickle.dump(train_features, f)
        with open(test_features_file, 'wb') as f:
            pickle.dump(test_features, f)

    # Convert the list of feature dicts to a feature matrix
    print("begin vectorization")
    vectorizer = DictVectorizer(sparse=True)
    train_vectors = vectorizer.fit_transform(train_features)
    test_vectors = vectorizer.transform(test_features)
    print("vector conversions completed")

    # Create a dictionary of models for easier access
    models = {
        'Logistic Regression': make_pipeline(SimpleImputer(strategy="mean"), MaxAbsScaler(),
                                             LogisticRegression(max_iter=1000)),
        'Random Forest': make_pipeline(SimpleImputer(strategy="mean"), RandomForestClassifier(n_estimators=100)),
        'Gradient Boosting': make_pipeline(SimpleImputer(strategy="mean"), GradientBoostingClassifier()),
        'SVC': make_pipeline(SimpleImputer(strategy="mean"), StandardScaler(with_mean=False), SVC()),
        'XGBoost': make_pipeline(SimpleImputer(strategy="mean"),
                                 XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
        'Hist Gradient Boosting': HistGradientBoostingClassifier(),  # HistGradientBoosting handles NaNs natively
        # 'LightGBM': LGBMClassifier(),  # LightGBM handles NaNs natively, uncomment if using
        'CatBoost': make_pipeline(SimpleImputer(strategy="mean"), CatBoostClassifier(silent=True)),
        # CatBoost handles NaNs, but pipeline for consistency
        'KNN': make_pipeline(SimpleImputer(strategy="mean"), MaxAbsScaler(), KNeighborsClassifier()),
        'G Naive Bayes': make_pipeline(SimpleImputer(strategy="mean"), GaussianNB())
    }

    # Train and evaluate each model
    for name, model in models.items():
        train_and_evaluate(model, name, train_vectors, train_labels, test_vectors, test_labels)

    end_time = time.time()
    print(f"Total time: {end_time - start_time}")
