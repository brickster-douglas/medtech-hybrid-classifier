# Databricks notebook source
# MAGIC %md
# MAGIC # Step 0: Setup — Tables & Synthetic Data
# MAGIC
# MAGIC Creates the schema and generates realistic MedTech synthetic data:
# MAGIC - **iso_codes**: Reference table of ISO 9999 assistive product codes
# MAGIC - **labeled_items**: Historical items with known ISO codes (training data)
# MAGIC - **new_items**: Unlabeled vendor price-list items to classify
# MAGIC - **human_corrections**: Expert corrections for the feedback loop

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

try:
    CATALOG = spark.conf.get("bundle.var.catalog")
except Exception:
    CATALOG = "serverless_stable_m3qkky_catalog"
try:
    SCHEMA = spark.conf.get("bundle.var.schema")
except Exception:
    SCHEMA = "embla_hybrid_classifier"

try:
    spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
except Exception:
    print(f"Note: Cannot create catalog '{CATALOG}' — it likely already exists or you lack CREATE CATALOG privilege. Continuing.")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")

print(f"Using: {CATALOG}.{SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ISO 9999 Reference Codes
# MAGIC
# MAGIC ISO 9999:2022 uses a 3-level hierarchy: **Class** (2 digits) → **Subclass** (4 digits) → **Division** (6 digits).
# MAGIC We focus on classes relevant to MedTech prosthetics/orthotics.

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, BooleanType

iso_codes_data = [
    # Class 06: Orthoses and prostheses
    # -- Spinal orthoses
    ("06 03 03", "Lumbar orthoses", "06 03", True, "Supports and stabilizes the lumbar spine"),
    ("06 03 06", "Thoraco-lumbar orthoses", "06 03", True, "Supports thoracic and lumbar spine"),
    ("06 03 09", "Cervical orthoses", "06 03", True, "Supports and immobilizes the cervical spine"),
    ("06 03 12", "Cervico-thoracic orthoses", "06 03", True, "Supports cervical and thoracic spine"),
    ("06 03 15", "Thoraco-lumbo-sacral orthoses", "06 03", True, "Full trunk support"),
    # -- Abdominal orthoses
    ("06 04 03", "Hernia orthoses", "06 04", True, "Supports abdominal wall hernias"),
    ("06 04 06", "Abdominal supports", "06 04", True, "General abdominal support garments"),
    # -- Upper limb orthoses
    ("06 06 03", "Finger orthoses", "06 06", True, "Supports or immobilizes finger joints"),
    ("06 06 06", "Hand orthoses", "06 06", True, "Supports or positions the hand"),
    ("06 06 09", "Wrist orthoses", "06 06", True, "Supports or immobilizes the wrist"),
    ("06 06 12", "Wrist-hand orthoses", "06 06", True, "Combined wrist and hand support"),
    ("06 06 15", "Elbow orthoses", "06 06", True, "Supports or restricts elbow motion"),
    ("06 06 18", "Elbow-wrist-hand orthoses", "06 06", True, "Full upper extremity support"),
    ("06 06 21", "Shoulder orthoses", "06 06", True, "Supports or immobilizes the shoulder"),
    # -- Lower limb orthoses
    ("06 12 03", "Foot orthoses", "06 12", True, "Custom insoles and arch supports"),
    ("06 12 06", "Ankle-foot orthoses", "06 12", True, "Supports ankle and foot, controls motion"),
    ("06 12 09", "Knee orthoses", "06 12", True, "Supports or restricts knee motion"),
    ("06 12 12", "Knee-ankle-foot orthoses", "06 12", True, "Full lower extremity bracing"),
    ("06 12 15", "Hip orthoses", "06 12", True, "Supports or restricts hip motion"),
    ("06 12 18", "Hip-knee-ankle-foot orthoses", "06 12", True, "Full lower limb support"),
    # -- Upper limb prostheses
    ("06 18 03", "Partial hand prostheses", "06 18", True, "Replaces partial hand function"),
    ("06 18 06", "Wrist disarticulation prostheses", "06 18", True, "Prosthesis at wrist level"),
    ("06 18 09", "Trans-radial prostheses", "06 18", True, "Below-elbow prostheses"),
    ("06 18 12", "Elbow disarticulation prostheses", "06 18", True, "Prosthesis at elbow level"),
    ("06 18 15", "Trans-humeral prostheses", "06 18", True, "Above-elbow prostheses"),
    ("06 18 18", "Shoulder disarticulation prostheses", "06 18", True, "Full arm prostheses"),
    # -- Lower limb prostheses
    ("06 24 03", "Partial foot prostheses", "06 24", True, "Replaces toes or partial foot"),
    ("06 24 06", "Ankle disarticulation prostheses", "06 24", True, "Syme prostheses"),
    ("06 24 09", "Trans-tibial prostheses", "06 24", True, "Below-knee prostheses"),
    ("06 24 12", "Knee disarticulation prostheses", "06 24", True, "Prosthesis at knee level"),
    ("06 24 15", "Trans-femoral prostheses", "06 24", True, "Above-knee prostheses"),
    ("06 24 18", "Hip disarticulation prostheses", "06 24", True, "Full leg prostheses"),
    ("06 24 21", "Prosthetic socks and sheaths", "06 24", True, "Interface between limb and socket"),
    # -- Non-limb prostheses
    ("06 30 03", "Breast prostheses", "06 30", True, "External breast forms"),
    ("06 30 06", "Ocular prostheses", "06 30", True, "Artificial eyes"),
    ("06 30 09", "Nasal prostheses", "06 30", True, "Prosthetic noses"),
    ("06 30 12", "Auricular prostheses", "06 30", True, "Prosthetic ears"),
    # Class 04: Assistive products for body functions
    ("04 24 03", "Compression stockings", "04 24", True, "Graduated compression garments for circulation"),
    ("04 24 06", "Compression arm sleeves", "04 24", True, "Upper limb compression garments"),
    ("04 24 09", "Anti-embolism stockings", "04 24", True, "Thrombosis prevention garments"),
    ("04 25 03", "TENS units", "04 25", True, "Transcutaneous electrical nerve stimulation"),
    ("04 25 06", "Neuromuscular stimulators", "04 25", True, "Electrical muscle stimulation devices"),
    # Class 09: Self-care
    ("09 03 03", "Dressing aids", "09 03", True, "Devices to help put on/remove clothing"),
    ("09 06 03", "Wound care dressings", "09 06", True, "Adhesive and non-adhesive wound dressings"),
    ("09 06 06", "Wound closure strips", "09 06", True, "Adhesive wound closure devices"),
    ("09 06 09", "Wound irrigation systems", "09 06", True, "Devices for cleaning wounds"),
    ("09 06 12", "Negative pressure wound therapy", "09 06", True, "Vacuum-assisted wound healing"),
    # Class 12: Mobility
    ("12 03 03", "Walking sticks", "12 03", True, "Single-point walking aids"),
    ("12 03 06", "Crutches", "12 03", True, "Axillary or forearm crutches"),
    ("12 03 09", "Walking frames", "12 03", True, "Rollators and walkers"),
    ("12 06 03", "Wheelchairs, manual", "12 06", True, "Self-propelled or attendant-pushed"),
    ("12 06 06", "Wheelchairs, powered", "12 06", True, "Electric wheelchairs"),
    ("12 18 03", "Therapeutic shoes", "12 18", True, "Custom orthopedic footwear"),
    ("12 18 06", "Shoe inserts", "12 18", True, "Removable orthotic insoles"),
]

schema = StructType([
    StructField("iso_code", StringType(), False),
    StructField("name", StringType(), False),
    StructField("iso_code_level_2", StringType(), False),
    StructField("part_of_iso_standard", BooleanType(), False),
    StructField("description", StringType(), True),
])

df_iso = spark.createDataFrame(iso_codes_data, schema)
df_iso.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.iso_codes")
print(f"Created iso_codes table: {df_iso.count()} codes")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Synthetic Vendor Items

# COMMAND ----------

import random
from datetime import datetime, timedelta
from pyspark.sql.types import FloatType, DateType

random.seed(42)

# Vendor profiles — each has specialties that influence item descriptions
vendors = {
    "Ossur Nordic": {
        "country": "IS", "currency": "EUR",
        "specialties": ["06 12", "06 24", "06 18"],
        "style": "prosthetic",
    },
    "Ottobock GmbH": {
        "country": "DE", "currency": "EUR",
        "specialties": ["06 24", "06 18", "06 12"],
        "style": "prosthetic",
    },
    "Bauerfeind AG": {
        "country": "DE", "currency": "EUR",
        "specialties": ["06 12", "06 06", "06 03", "04 24"],
        "style": "orthotic",
    },
    "DJO Global": {
        "country": "US", "currency": "USD",
        "specialties": ["06 12", "06 06", "06 03"],
        "style": "orthotic",
    },
    "Medi GmbH": {
        "country": "DE", "currency": "EUR",
        "specialties": ["04 24", "06 12", "06 03"],
        "style": "compression",
    },
    "Thuasne Group": {
        "country": "FR", "currency": "EUR",
        "specialties": ["06 03", "06 04", "06 12", "04 24"],
        "style": "orthotic",
    },
    "Breg Inc": {
        "country": "US", "currency": "USD",
        "specialties": ["06 12", "06 06"],
        "style": "orthotic",
    },
    "Hanger Clinic": {
        "country": "US", "currency": "USD",
        "specialties": ["06 24", "06 18", "06 12"],
        "style": "prosthetic",
    },
    "Smith & Nephew": {
        "country": "GB", "currency": "EUR",
        "specialties": ["09 06", "04 24"],
        "style": "wound_care",
    },
    "Molnlycke Health": {
        "country": "SE", "currency": "SEK",
        "specialties": ["09 06"],
        "style": "wound_care",
    },
    "Invacare Nordic": {
        "country": "SE", "currency": "SEK",
        "specialties": ["12 03", "12 06"],
        "style": "mobility",
    },
    "Sunrise Medical": {
        "country": "DE", "currency": "EUR",
        "specialties": ["12 03", "12 06"],
        "style": "mobility",
    },
    "Sigvaris Group": {
        "country": "CH", "currency": "EUR",
        "specialties": ["04 24"],
        "style": "compression",
    },
    "Profisee MedTech": {
        "country": "IS", "currency": "ISK",
        "specialties": ["06 24", "06 12", "06 18", "12 18"],
        "style": "prosthetic",
    },
    "BSN Medical": {
        "country": "DE", "currency": "EUR",
        "specialties": ["04 24", "09 06", "06 03"],
        "style": "compression",
    },
]

# Product description templates per ISO subclass
description_templates = {
    "06 03 03": [
        "Lumbar support brace {adj} for lower back stabilization",
        "LSO {adj} lumbar orthosis with rigid stays",
        "{adj} lumbosacral support belt with posterior panel",
    ],
    "06 03 06": [
        "TLSO {adj} thoraco-lumbar orthosis",
        "Thoraco-lumbar brace {adj} with anterior/posterior panels",
        "{adj} thoracolumbar spinal orthosis with strapping",
    ],
    "06 03 09": [
        "Cervical collar {adj} foam padded",
        "Philadelphia collar {adj} rigid cervical orthosis",
        "{adj} neck brace cervical support",
    ],
    "06 03 12": [
        "CTO {adj} cervico-thoracic orthosis with chin support",
        "{adj} cervicothoracic brace sternal plate",
    ],
    "06 03 15": [
        "TLSO {adj} full trunk orthosis with pelvic band",
        "{adj} thoraco-lumbo-sacral body jacket",
    ],
    "06 04 03": [
        "Inguinal hernia truss {adj}",
        "{adj} hernia support belt with pad",
    ],
    "06 04 06": [
        "{adj} abdominal binder post-surgical",
        "Abdominal support {adj} elastic wrap",
    ],
    "06 06 03": [
        "Finger splint {adj} for MCP joint",
        "Mallet finger orthosis {adj} stack type",
        "{adj} buddy splint for finger immobilization",
    ],
    "06 06 06": [
        "Hand resting splint {adj} functional position",
        "{adj} hand orthosis with thumb post",
    ],
    "06 06 09": [
        "Wrist brace {adj} with aluminium stay",
        "{adj} wrist immobilizer carpal tunnel",
        "Wrist support {adj} neoprene wrap",
    ],
    "06 06 12": [
        "WHO {adj} wrist-hand orthosis with finger extension",
        "{adj} resting hand splint wrist neutral",
    ],
    "06 06 15": [
        "Elbow brace {adj} hinged ROM",
        "{adj} tennis elbow strap with gel pad",
        "Elbow orthosis {adj} post-operative",
    ],
    "06 06 18": [
        "EWHO {adj} elbow-wrist-hand orthosis dynamic",
        "{adj} long arm splint with wrist control",
    ],
    "06 06 21": [
        "Shoulder immobilizer {adj} sling type",
        "{adj} shoulder abduction orthosis with pillow",
        "Shoulder brace {adj} post-dislocation",
    ],
    "06 12 03": [
        "Custom foot orthotic {adj} arch support",
        "{adj} heel cup silicone for plantar fasciitis",
        "Metatarsal pad {adj} forefoot offloading",
        "Foot orthosis {adj} rigid carbon fiber",
    ],
    "06 12 06": [
        "AFO {adj} ankle-foot orthosis posterior leaf spring",
        "{adj} carbon fiber AFO dynamic response",
        "Articulated AFO {adj} with dorsiflexion assist",
        "Drop foot orthosis {adj} spring loaded",
    ],
    "06 12 09": [
        "Knee brace {adj} hinged ligament support",
        "{adj} patella stabilizer knee orthosis",
        "ACL knee brace {adj} functional sport",
        "Knee orthosis {adj} offloading for OA",
    ],
    "06 12 12": [
        "KAFO {adj} knee-ankle-foot orthosis locked",
        "{adj} stance control KAFO with sensor",
    ],
    "06 12 15": [
        "Hip orthosis {adj} post-operative abduction",
        "{adj} hip brace flexion restriction",
    ],
    "06 12 18": [
        "HKAFO {adj} reciprocating gait orthosis",
        "{adj} hip-knee-ankle-foot orthosis bilateral",
    ],
    "06 18 03": [
        "Silicone partial hand prosthesis {adj}",
        "{adj} cosmetic finger prosthesis custom-molded",
    ],
    "06 18 06": [
        "Wrist disarticulation prosthesis {adj} myoelectric",
        "{adj} WD prosthesis body-powered",
    ],
    "06 18 09": [
        "Below-elbow prosthesis {adj} myoelectric hand",
        "{adj} transradial prosthesis body-powered hook",
        "TR prosthesis {adj} with quick-disconnect wrist",
    ],
    "06 18 12": [
        "ED prosthesis {adj} with polycentric elbow",
        "{adj} elbow disarticulation prosthesis hybrid",
    ],
    "06 18 15": [
        "Above-elbow prosthesis {adj} myoelectric",
        "{adj} transhumeral prosthesis with Utah arm",
    ],
    "06 18 18": [
        "Shoulder disarticulation prosthesis {adj}",
        "{adj} full arm prosthesis cosmetic",
    ],
    "06 24 03": [
        "Partial foot prosthesis {adj} silicone toe filler",
        "{adj} toe prosthesis custom cosmetic",
    ],
    "06 24 06": [
        "Syme prosthesis {adj} with expandable wall",
        "{adj} ankle disarticulation prosthesis lightweight",
    ],
    "06 24 09": [
        "Below-knee prosthesis {adj} with energy-storing foot",
        "{adj} transtibial prosthesis PTB socket",
        "BK prosthesis {adj} carbon fiber dynamic foot",
        "Trans-tibial {adj} prosthesis total surface bearing",
    ],
    "06 24 12": [
        "KD prosthesis {adj} with polycentric knee",
        "{adj} knee disarticulation prosthesis four-bar",
    ],
    "06 24 15": [
        "Above-knee prosthesis {adj} microprocessor knee",
        "{adj} transfemoral prosthesis C-Leg",
        "AK prosthesis {adj} with hydraulic knee unit",
        "Trans-femoral {adj} prosthesis ischial containment socket",
    ],
    "06 24 18": [
        "Hip disarticulation prosthesis {adj} Canadian type",
        "{adj} hemipelvectomy prosthesis modular",
    ],
    "06 24 21": [
        "Prosthetic sock {adj} 3-ply wool blend",
        "{adj} gel liner with pin lock",
        "Prosthetic sheath {adj} nylon knit",
        "Silicone liner {adj} cushion with seal-in",
    ],
    "06 30 03": [
        "Breast prosthesis {adj} silicone full form",
        "{adj} partial breast form lightweight",
    ],
    "06 30 06": [
        "Ocular prosthesis {adj} custom painted",
        "{adj} artificial eye scleral shell",
    ],
    "06 30 09": [
        "Nasal prosthesis {adj} silicone custom",
    ],
    "06 30 12": [
        "Auricular prosthesis {adj} implant-retained",
    ],
    "04 24 03": [
        "Compression stocking {adj} Class II 23-32 mmHg",
        "{adj} graduated compression knee-high",
        "Medical compression hose {adj} thigh-length",
        "Compression sock {adj} Class I 18-21 mmHg",
    ],
    "04 24 06": [
        "Compression arm sleeve {adj} for lymphedema",
        "{adj} upper extremity compression gauntlet",
    ],
    "04 24 09": [
        "Anti-embolism stocking {adj} knee-high 18 mmHg",
        "{adj} TED hose post-surgical",
    ],
    "04 25 03": [
        "TENS unit {adj} dual-channel portable",
        "{adj} transcutaneous nerve stimulator rechargeable",
    ],
    "04 25 06": [
        "EMS device {adj} neuromuscular stimulator",
        "{adj} functional electrical stimulation unit",
    ],
    "09 03 03": [
        "Button hook dressing aid {adj}",
        "{adj} sock aid with long handles",
    ],
    "09 06 03": [
        "Foam wound dressing {adj} non-adhesive",
        "{adj} hydrocolloid wound dressing sterile",
        "Alginate wound dressing {adj} for exudate",
        "Silver antimicrobial dressing {adj}",
    ],
    "09 06 06": [
        "Wound closure strip {adj} adhesive",
        "{adj} butterfly closure strips sterile",
    ],
    "09 06 09": [
        "Wound irrigation syringe {adj} 60ml",
        "{adj} saline wound wash spray",
    ],
    "09 06 12": [
        "NPWT device {adj} portable negative pressure",
        "{adj} vacuum wound therapy system with canister",
    ],
    "12 03 03": [
        "Walking stick {adj} ergonomic handle",
        "{adj} folding walking cane adjustable",
    ],
    "12 03 06": [
        "Forearm crutch {adj} ergonomic grip",
        "{adj} axillary crutch pair aluminium",
    ],
    "12 03 09": [
        "Rollator {adj} four-wheel with seat",
        "{adj} walking frame folding lightweight",
    ],
    "12 06 03": [
        "Manual wheelchair {adj} lightweight folding",
        "{adj} self-propelled wheelchair sport",
    ],
    "12 06 06": [
        "Power wheelchair {adj} mid-wheel drive",
        "{adj} electric wheelchair with tilt recline",
    ],
    "12 18 03": [
        "Therapeutic shoe {adj} extra-depth diabetic",
        "{adj} orthopedic shoe custom molded",
    ],
    "12 18 06": [
        "Orthotic insole {adj} heat-moldable EVA",
        "{adj} shoe insert rigid polypropylene",
    ],
}

adjectives = [
    "premium", "standard", "advanced", "clinical-grade", "professional",
    "pediatric", "geriatric", "sport", "heavy-duty", "ultra-light",
    "bilateral", "unilateral", "adjustable", "modular", "custom-fit",
    "breathable", "waterproof", "antimicrobial", "hypoallergenic", "latex-free",
]

# Price ranges by ISO class
price_ranges = {
    "06 03": (80, 450),    "06 04": (40, 200),    "06 06": (30, 350),
    "06 12": (50, 800),    "06 18": (2000, 25000), "06 24": (1500, 35000),
    "06 30": (200, 3000),  "04 24": (20, 120),     "04 25": (50, 400),
    "09 03": (10, 50),     "09 06": (5, 80),       "12 03": (20, 200),
    "12 06": (500, 8000),  "12 18": (80, 600),
}


def generate_item(item_id, iso_code, vendor_name, vendor_info, include_label=True):
    """Generate a single synthetic item."""
    templates = description_templates.get(iso_code, [f"Medical device for {iso_code}"])
    desc = random.choice(templates).format(adj=random.choice(adjectives))

    subclass = iso_code[:5]
    price_lo, price_hi = price_ranges.get(subclass, (50, 500))
    price = round(random.uniform(price_lo, price_hi), 2)

    days_ago = random.randint(1, 730)
    created = datetime.now() - timedelta(days=days_ago)

    row = {
        "item_id": f"ITM-{item_id:05d}",
        "vendor_name": vendor_name,
        "product_description": desc,
        "unit_price": float(price),
        "currency": vendor_info["currency"],
        "vendor_country": vendor_info["country"],
        "created_date": created.date(),
    }
    if include_label:
        row["iso_code"] = iso_code
    return row


# Generate labeled items (training data)
labeled_rows = []
item_counter = 1
iso_code_list = [row[0] for row in iso_codes_data]

for vendor_name, vendor_info in vendors.items():
    # Each vendor generates items primarily in their specialties
    specialty_codes = [c for c in iso_code_list if c[:5] in vendor_info["specialties"]]
    other_codes = [c for c in iso_code_list if c[:5] not in vendor_info["specialties"]]

    # 80% specialty items, 20% random (simulates real distribution)
    n_specialty = random.randint(80, 130)
    n_other = random.randint(5, 20)

    for _ in range(n_specialty):
        code = random.choice(specialty_codes) if specialty_codes else random.choice(iso_code_list)
        labeled_rows.append(generate_item(item_counter, code, vendor_name, vendor_info))
        item_counter += 1

    for _ in range(n_other):
        code = random.choice(other_codes) if other_codes else random.choice(iso_code_list)
        labeled_rows.append(generate_item(item_counter, code, vendor_name, vendor_info))
        item_counter += 1

random.shuffle(labeled_rows)
print(f"Generated {len(labeled_rows)} labeled items")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write Tables

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, FloatType, DateType

labeled_schema = StructType([
    StructField("item_id", StringType(), False),
    StructField("vendor_name", StringType(), False),
    StructField("product_description", StringType(), False),
    StructField("unit_price", FloatType(), False),
    StructField("currency", StringType(), False),
    StructField("vendor_country", StringType(), False),
    StructField("created_date", DateType(), False),
    StructField("iso_code", StringType(), False),
])

df_labeled = spark.createDataFrame(labeled_rows, labeled_schema)
df_labeled.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.labeled_items")
print(f"Wrote labeled_items: {df_labeled.count()} rows, {df_labeled.select('iso_code').distinct().count()} unique ISO codes")

# COMMAND ----------

# Generate new (unlabeled) items for scoring
new_rows = []
for _ in range(500):
    vendor_name = random.choice(list(vendors.keys()))
    vendor_info = vendors[vendor_name]
    specialty_codes = [c for c in iso_code_list if c[:5] in vendor_info["specialties"]]
    code = random.choice(specialty_codes) if specialty_codes else random.choice(iso_code_list)
    row = generate_item(item_counter, code, vendor_name, vendor_info, include_label=False)
    # Store ground truth separately for evaluation (not visible to model)
    row["_ground_truth_iso"] = code
    new_rows.append(row)
    item_counter += 1

new_schema = StructType([
    StructField("item_id", StringType(), False),
    StructField("vendor_name", StringType(), False),
    StructField("product_description", StringType(), False),
    StructField("unit_price", FloatType(), False),
    StructField("currency", StringType(), False),
    StructField("vendor_country", StringType(), False),
    StructField("created_date", DateType(), False),
    StructField("_ground_truth_iso", StringType(), True),
])

df_new = spark.createDataFrame(new_rows, new_schema)
df_new.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.new_items")
print(f"Wrote new_items: {df_new.count()} rows")

# COMMAND ----------

# Create empty corrections table for the feedback loop
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA}.human_corrections (
    item_id STRING,
    predicted_iso_code STRING,
    corrected_iso_code STRING,
    correction_date DATE,
    corrector_name STRING
)
""")
print("Created human_corrections table")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Setup

# COMMAND ----------

for table in ["iso_codes", "labeled_items", "new_items", "human_corrections"]:
    count = spark.table(f"{CATALOG}.{SCHEMA}.{table}").count()
    print(f"  {table}: {count} rows")

print("\nLabel distribution (top 10):")
display(
    df_labeled.groupBy("iso_code")
    .count()
    .orderBy("count", ascending=False)
    .limit(10)
)
