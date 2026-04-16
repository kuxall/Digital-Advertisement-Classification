"""
retrain_model.py
────────────────
Generates synthetic training data for all 5 ad categories,
merges with the original dataset, and retrains the classifier.

Usage:
    python retrain_model.py

Output:
    - data/synthetic_data.csv          (generated samples)
    - notebook/model/adv_model.sav     (new model, old one backed up)
"""

import os
import re
import pickle
import random
import shutil

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

import warnings
warnings.filterwarnings("ignore")

random.seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────
# TEXT PREPROCESSING
# ─────────────────────────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ─────────────────────────────────────────────
# SYNTHETIC DATA TEMPLATES & VOCABULARY
# ─────────────────────────────────────────────

# ── Jobs – IT ──
IT_TEMPLATES = [
    "We are hiring a {role} to join our {team} team. The ideal candidate has {exp} years of experience with {tech1} and {tech2}. Responsibilities include {duty1} and {duty2}. {cert} certification is a plus.",
    "Seeking a {role} to design, build and maintain {tech1} systems. You will work closely with {team} teams on {duty1}. Must be proficient in {tech2}. Remote-friendly position.",
    "{role} needed for a fast-growing tech company. Must have strong knowledge of {tech1}, {tech2}, and {tech3}. You will be responsible for {duty1}, {duty2}, and {duty3}.",
    "Join our {team} team as a {role}. We need someone experienced in {tech1} to handle {duty1} and {duty2}. Experience with {tech2} and {tech3} is required.",
    "Looking for a skilled {role}. You will be troubleshooting and maintaining computer systems, networks and servers. Experience with {tech1} and {tech2} required.",
    "{company} is looking for a {role}. Key responsibilities: {duty1}, {duty2}. Required skills: {tech1}, {tech2}, {tech3}. Competitive salary and benefits.",
    "We need a {role} to support our IT infrastructure. Tasks include {duty1}, {duty2}, and ensuring uptime of {tech1} systems. {cert} preferred.",
    "Remote {role} position available. Must have hands-on experience with {tech1} and {tech2}. You will {duty1} and work with the {team} team.",
    "{role} role: {exp} years of experience with {tech1} required. Responsibilities include {duty1}, {duty2}. Knowledge of {tech2} and {tech3} is a strong plus.",
    "We are expanding our {team} team and need a {role}. Daily tasks: {duty1}, {duty2}, {duty3}. Stack: {tech1}, {tech2}.",
]
IT_ROLES = [
    "Software Engineer", "Full Stack Developer", "Backend Developer", "Frontend Developer",
    "DevOps Engineer", "Cloud Engineer", "Network Administrator", "IT Support Specialist",
    "Systems Administrator", "Data Engineer", "Cybersecurity Analyst", "Database Administrator",
    "Site Reliability Engineer", "Machine Learning Engineer", "QA Engineer",
    "IT Help Desk Technician", "Infrastructure Engineer", "Web Developer",
]
IT_TEAMS = ["engineering", "infrastructure", "cloud", "security", "platform", "data", "DevOps"]
IT_TECH1 = ["Python", "Java", "AWS", "Azure", "Linux", "Docker", "Kubernetes", "SQL", "React",
            "Node.js", "C#", ".NET", "Terraform", "Ansible", "networking protocols", "firewalls"]
IT_TECH2 = ["CI/CD pipelines", "REST APIs", "Git", "PostgreSQL", "MongoDB", "Windows Server",
            "Active Directory", "VMware", "TCP/IP", "bash scripting", "cloud services"]
IT_TECH3 = ["Agile methodology", "microservices", "containerization", "load balancers",
            "monitoring tools", "cybersecurity best practices"]
IT_DUTIES = [
    "troubleshoot and resolve network issues", "maintain server infrastructure",
    "develop and deploy web applications", "write clean and maintainable code",
    "manage cloud resources and costs", "perform security audits",
    "collaborate with cross-functional teams", "design database schemas",
    "automate deployment pipelines", "provide IT support to end users",
    "monitor system performance and uptime", "implement security best practices",
    "configure and manage firewalls", "conduct code reviews",
    "develop internal tools and dashboards",
]
IT_CERTS = ["AWS Certified Solutions Architect", "CompTIA Security+", "CCNA", "CompTIA A+",
            "Google Cloud Professional", "Certified Ethical Hacker (CEH)", "PMP"]
IT_COMPANIES = ["TechCorp", "CloudSystems Inc.", "Nexus Technologies", "DataBridge",
                "Apex Software", "InfraNet", "ByteWorks", "ServerPro Solutions"]

# ── Jobs – Retail ──
RETAIL_TEMPLATES = [
    "We are hiring a {role} for our {location} store. The ideal candidate will have {exp} in customer service. Responsibilities include {duty1} and {duty2}. {perk}.",
    "Looking for a part-time {role} to join our team at {company}. Duties: {duty1}, {duty2}. Must be available on weekends.",
    "{company} is seeking a {role}. You will {duty1} and {duty2}. Previous retail experience preferred but not required. We offer {perk}.",
    "Join the {company} family as a {role}. Must be friendly, reliable and able to {duty1}. {perk} and employee discount offered.",
    "Hiring immediately! {role} needed for our {location} location. Experience with {duty1} required. Full-time and part-time shifts available.",
    "{role} wanted for busy retail store. Duties include {duty1}, {duty2}, {duty3}. Great team environment, apply today.",
    "We need a reliable {role} to {duty1} and ensure our customers have an excellent shopping experience.",
    "{company} has openings for {role}. Responsibilities: {duty1}, {duty2}. Flexible hours available.",
    "Experienced {role} needed for our {location} location. Must have experience with point of sale systems and {duty1}.",
    "Full-time {role} position at {company}. You will {duty1}, {duty2}. Competitive pay and employee benefits included.",
]
RETAIL_ROLES = [
    "Retail Sales Associate", "Cashier", "Store Manager", "Assistant Manager",
    "Sales Floor Associate", "Inventory Specialist", "Visual Merchandiser",
    "Customer Service Representative", "Shift Supervisor", "Team Lead",
    "Stock Associate", "Keyholder", "Loss Prevention Officer",
]
RETAIL_DUTIES = [
    "assist customers on the sales floor", "operate the cash register",
    "restock shelves and organize merchandise", "process returns and exchanges",
    "maintain store cleanliness", "greet and assist customers",
    "handle cash and credit card transactions", "help with inventory counts",
    "set up promotional displays", "assist with opening and closing procedures",
    "upsell products to customers", "manage fitting room area",
    "price merchandise and apply markdowns",
]
RETAIL_PERKS = [
    "competitive pay and benefits", "employee discount program",
    "flexible scheduling", "paid training provided", "opportunity for advancement",
    "health and dental benefits",
]
RETAIL_COMPANIES = ["Fashion Outlet", "StyleMart", "ValueMart", "The Clothing Co.",
                    "HomeGoods Plus", "TrendStore", "SportZone", "QuickMart"]
RETAIL_LOCATIONS = ["downtown", "mall", "suburban", "flagship", "outlet"]

# ── Banking ──
BANKING_TEMPLATES = [
    "We are seeking a {role} to join our {dept} department. Ideal candidate has {exp} years of experience in {area1} and {area2}. {cert} certification preferred.",
    "{company} Bank is hiring a {role}. Responsibilities include {duty1} and {duty2}. CPA or CFP certification is a plus.",
    "Looking for an experienced {role} with knowledge of {area1}. You will {duty1}, {duty2}. Strong analytical skills required.",
    "{role} needed at a leading financial institution. Duties: {duty1}, {duty2}. Must understand {area1} regulations.",
    "Join our {dept} team as a {role}. You will be responsible for {duty1}, {duty2}, and portfolio management.",
    "Financial {role} position available. Experience in {area1} and {area2} required. Must have {exp} years in banking sector.",
    "{company} is expanding. We need a {role} to assist clients with {area1} needs and {duty1}.",
    "Experienced {role} sought for our growing bank. Key skills: {area1}, {area2}. Responsibilities include {duty1}.",
]
BANKING_ROLES = [
    "Loan Officer", "Bank Teller", "Financial Advisor", "Credit Analyst",
    "Mortgage Specialist", "Investment Banker", "Accountant", "Financial Analyst",
    "Risk Manager", "Compliance Officer", "Underwriter", "Bookkeeper",
    "Branch Manager", "Insurance Agent", "Tax Consultant",
]
BANKING_DEPTS = ["lending", "retail banking", "investment", "compliance", "risk management", "accounting"]
BANKING_AREAS = [
    "mortgage underwriting", "credit risk analysis", "financial planning",
    "investment portfolios", "tax preparation", "regulatory compliance",
    "loan origination", "financial reporting", "wealth management", "insurance products",
]
BANKING_DUTIES = [
    "assess creditworthiness of loan applicants", "prepare financial reports",
    "advise clients on investment strategies", "review mortgage applications",
    "ensure regulatory compliance", "manage client accounts",
    "process banking transactions", "analyze financial statements",
    "develop budgets and forecasts", "conduct risk assessments",
]
BANKING_CERTS = ["CPA", "CFP", "CFA", "Series 7", "Series 63", "NMLS"]
BANKING_COMPANIES = ["First National Bank", "Heritage Financial", "Pacific Trust",
                     "Meridian Bank", "Capitol Federal", "Liberty Savings"]

# ── Sell – House ──
HOUSE_TEMPLATES = [
    "{beds}-bedroom, {baths}-bathroom home for sale in {area}. {sqft} sq ft. Features include {feat1} and {feat2}. Asking price: ${price}.",
    "Beautiful {beds} bed/{baths} bath house for sale. Newly {feat1} {area} neighborhood. {sqft} sqft, {garage}. Listed at ${price}.",
    "{beds}BR/{baths}BA home in desirable {area}. Features {feat1}, {feat2}, {feat3}. Open house this weekend. MLS #{mls}.",
    "For Sale: Spacious {beds}-bedroom home with {baths} bathrooms in {area}. {sqft} sq ft. {feat1} and {feat2} included.",
    "Charming {beds} bed home for sale. {feat1}, {feat2}. Located in quiet {area} neighborhood. ${price} OBO.",
    "Investor special! {beds}-bedroom house in {area}. Needs TLC but priced to sell at ${price}. Great rental potential.",
    "Move-in ready {beds}BR home in {area}. Recently {feat1} and {feat2}. {sqft} sq ft. Asking ${price}.",
    "Luxury {beds}-bed, {baths}-bath property in {area}. High-end finishes, {feat1}, {feat2}. Listing price ${price}.",
]
HOUSE_BEDS = ["2", "3", "4", "5"]
HOUSE_BATHS = ["1", "2", "3"]
HOUSE_AREAS = ["Oak Park", "Lakeside", "Maple Grove", "Sunset Hills", "Riverside",
               "downtown Dallas", "North Dallas", "Plano", "McKinney", "Frisco"]
HOUSE_SQFT = ["1,200", "1,500", "1,800", "2,100", "2,400", "2,800", "3,200"]
HOUSE_FEATS = [
    "renovated kitchen", "updated bathrooms", "hardwood floors", "open floor plan",
    "large backyard", "in-ground pool", "attached garage", "modern appliances",
    "granite countertops", "vaulted ceilings", "master suite", "fireplace",
    "covered patio", "new roof", "fresh paint throughout",
]
HOUSE_PRICES = ["185,000", "220,000", "265,000", "310,000", "375,000", "425,000", "510,000"]
HOUSE_GARAGES = ["2-car garage", "1-car garage", "carport", "no garage"]

# ── Rent – Apartment ──
APARTMENT_TEMPLATES = [
    "{beds} bedroom apartment for rent in {area}. ${rent}/month. {feat1} and {feat2} included. Available {avail}.",
    "Spacious {beds}BR/{baths}BA apartment in {area}. Rent: ${rent}/mo. {feat1}. Utilities {util}. Call for a tour.",
    "For rent: {beds} bed apartment, ${rent}/month. {area} location. {feat1} and {feat2}. Lease term: {lease}.",
    "{beds}-bedroom apartment available in {area}. ${rent}/month includes {util}. {feat1}. No pets policy.",
    "Studio/1BR for rent in {area}. Great location, ${rent}/mo. {feat1}, {feat2}. Available immediately.",
    "Nice {beds}BR apartment in {area}. ${rent}. Features: {feat1}, {feat2}. {lease} lease, deposit required.",
    "Furnished {beds}-bedroom apartment for rent. {area} area. ${rent}/month, utilities included. {feat1}.",
    "{beds}BD/{baths}BA unit in {area} complex. ${rent}/mo. Amenities: {feat1}, {feat2}. Renters insurance required.",
]
APT_BEDS = ["Studio", "1", "2", "3"]
APT_BATHS = ["1", "2"]
APT_AREAS = ["Uptown Dallas", "Deep Ellum", "Oak Cliff", "Addison", "Richardson",
             "Garland", "Irving", "Arlington", "Denton", "Lewisville"]
APT_RENTS = ["800", "950", "1,100", "1,250", "1,400", "1,650", "1,900"]
APT_FEATS = [
    "in-unit washer/dryer", "on-site laundry", "swimming pool", "gym access",
    "covered parking", "pet friendly", "balcony", "updated kitchen",
    "hardwood floors", "central A/C", "gated community", "24-hr security",
    "walk-in closet", "dishwasher", "stainless steel appliances",
]
APT_UTILS = ["water and trash", "all utilities", "water only", "not included"]
APT_LEASES = ["12-month", "6-month", "month-to-month"]
APT_AVAIL = ["immediately", "1st of next month", "in 2 weeks", "30 days notice"]

# ─────────────────────────────────────────────
# GENERATORS
# ─────────────────────────────────────────────
def gen_it(n):
    rows = []
    for _ in range(n):
        t = random.choice(IT_TEMPLATES)
        text = t.format(
            role=random.choice(IT_ROLES), team=random.choice(IT_TEAMS),
            exp=random.randint(2, 8), tech1=random.choice(IT_TECH1),
            tech2=random.choice(IT_TECH2), tech3=random.choice(IT_TECH3),
            duty1=random.choice(IT_DUTIES), duty2=random.choice(IT_DUTIES),
            duty3=random.choice(IT_DUTIES), cert=random.choice(IT_CERTS),
            company=random.choice(IT_COMPANIES),
        )
        rows.append({"Job_Description": text, "JobType": "Jobs \u2013 IT"})
    return rows

def gen_retail(n):
    rows = []
    for _ in range(n):
        t = random.choice(RETAIL_TEMPLATES)
        text = t.format(
            role=random.choice(RETAIL_ROLES), company=random.choice(RETAIL_COMPANIES),
            location=random.choice(RETAIL_LOCATIONS), exp=f"{random.randint(0, 3)} year(s)",
            duty1=random.choice(RETAIL_DUTIES), duty2=random.choice(RETAIL_DUTIES),
            duty3=random.choice(RETAIL_DUTIES), perk=random.choice(RETAIL_PERKS),
        )
        rows.append({"Job_Description": text, "JobType": "Jobs \u2013 Retail"})
    return rows

def gen_banking(n):
    rows = []
    for _ in range(n):
        t = random.choice(BANKING_TEMPLATES)
        text = t.format(
            role=random.choice(BANKING_ROLES), company=random.choice(BANKING_COMPANIES),
            dept=random.choice(BANKING_DEPTS), exp=random.randint(2, 10),
            area1=random.choice(BANKING_AREAS), area2=random.choice(BANKING_AREAS),
            duty1=random.choice(BANKING_DUTIES), duty2=random.choice(BANKING_DUTIES),
            cert=random.choice(BANKING_CERTS),
        )
        rows.append({"Job_Description": text, "JobType": "Banking"})
    return rows

def gen_house(n):
    rows = []
    for _ in range(n):
        t = random.choice(HOUSE_TEMPLATES)
        feats = random.sample(HOUSE_FEATS, 3)
        text = t.format(
            beds=random.choice(HOUSE_BEDS), baths=random.choice(HOUSE_BATHS),
            area=random.choice(HOUSE_AREAS), sqft=random.choice(HOUSE_SQFT),
            feat1=feats[0], feat2=feats[1], feat3=feats[2],
            price=random.choice(HOUSE_PRICES), garage=random.choice(HOUSE_GARAGES),
            mls=random.randint(100000, 999999),
        )
        rows.append({"Job_Description": text, "JobType": "Sell \u2013 House"})
    return rows

def gen_apartment(n):
    rows = []
    for _ in range(n):
        t = random.choice(APARTMENT_TEMPLATES)
        feats = random.sample(APT_FEATS, 2)
        text = t.format(
            beds=random.choice(APT_BEDS), baths=random.choice(APT_BATHS),
            area=random.choice(APT_AREAS), rent=random.choice(APT_RENTS),
            feat1=feats[0], feat2=feats[1], util=random.choice(APT_UTILS),
            lease=random.choice(APT_LEASES), avail=random.choice(APT_AVAIL),
        )
        rows.append({"Job_Description": text, "JobType": "Rent \u2013 Apartment"})
    return rows

# ─────────────────────────────────────────────
# MAIN: GENERATE → MERGE → RETRAIN → SAVE
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Generate synthetic data
    # Target ~550 per class after merging with real data
    # Real counts: Sell-House=607, Retail=517, Banking=191, Apt=135, IT=91
    print("Generating synthetic data...")
    synthetic = (
        gen_it(460)        # 91  + 460 = 551
        + gen_banking(360) # 191 + 360 = 551
        + gen_apartment(415) # 135 + 415 = 550
        + gen_retail(35)   # 517 +  35 = 552
        # Sell-House already at 607, no synthetic needed
    )
    syn_df = pd.DataFrame(synthetic)
    syn_df.to_csv("data/synthetic_data.csv", index=False)
    print(f"  Generated {len(syn_df)} samples")
    print(syn_df["JobType"].value_counts().to_string())

    # 2. Load & merge with real data
    print("\nLoading original data...")
    real_df = pd.read_excel("data/ConcatenatedDigitalAdData.xlsx")
    real_df = real_df[["JobType", "Job_Description"]].dropna()
    real_df["JobType"] = real_df["JobType"].str.strip()
    syn_df["JobType"] = syn_df["JobType"].str.strip()

    combined = pd.concat(
        [real_df[["Job_Description", "JobType"]], syn_df[["Job_Description", "JobType"]]],
        ignore_index=True
    )
    combined["Cleaned_Text"] = combined["Job_Description"].apply(clean_text)
    combined = combined[combined["Cleaned_Text"].str.len() > 20].reset_index(drop=True)

    print(f"\nCombined dataset: {len(combined)} samples")
    print(combined["JobType"].value_counts().to_string())

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        combined["Cleaned_Text"], combined["JobType"],
        test_size=0.2, random_state=42, stratify=combined["JobType"]
    )
    print(f"\nTrain: {len(X_train)}  |  Test: {len(X_test)}")

    # 4. Build & train pipeline
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            sublinear_tf=True, min_df=2, max_df=0.90,
            norm="l2", encoding="latin-1",
            ngram_range=(1, 2), stop_words="english",
            max_features=20000,
        )),
        ("clf", CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", C=1.0, max_iter=10000),
            cv=5,
        )),
    ])

    print("\nTraining...")
    pipeline.fit(X_train, y_train)

    # 5. Evaluate
    y_pred = pipeline.predict(X_test)
    print(f"\nTest Accuracy : {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, combined["Cleaned_Text"], combined["JobType"], cv=cv, scoring="accuracy")
    print(f"5-Fold CV     : {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")

    # 6. Quick sanity checks
    print("\nSanity checks:")
    checks = [
        ("IT",      "We are seeking a Software Engineer with experience troubleshooting computer systems and networks."),
        ("IT",      "Senior Python developer needed for cloud infrastructure. AWS, Docker, Kubernetes required."),
        ("Retail",  "Hiring cashier for grocery store. Part time, no experience needed."),
        ("Sell",    "3-bedroom house for sale in Plano. Open house this Sunday. Asking 320000."),
        ("Rent",    "2 bedroom apartment for rent in Uptown Dallas. 1200 per month utilities included."),
        ("Bank",    "Loan officer needed at regional bank. Mortgage underwriting experience preferred."),
    ]
    passed = 0
    for expected, text in checks:
        pred = pipeline.predict([clean_text(text)])[0]
        conf = np.max(pipeline.predict_proba([clean_text(text)])[0]) * 100
        ok = expected.lower() in pred.lower()
        if ok:
            passed += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {expected:8s} => {pred} ({conf:.1f}%)")
    print(f"  {passed}/{len(checks)} passed")

    # 7. Save model (backup old one first)
    save_path = "notebook/model/adv_model.sav"
    backup_path = "notebook/model/adv_model_backup.sav"
    if os.path.exists(save_path):
        shutil.copy2(save_path, backup_path)
        print(f"\nBacked up old model -> {backup_path}")

    with open(save_path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Saved new model  -> {save_path}")
    print("\nDone. Restart the Streamlit app to load the new model.")
