import os
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from tqdm.auto import tqdm



# This program prepares a product catalog dataset, generates text embeddings 
# for each product using a HuggingFace model, and indexes those embeddings 
# in a Pinecone vector database for semantic search or similarity queries.



load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


product_catalog = [
    {
        "id": "SNK001",
        "name": "Kärcher K 5 Power Control High-Pressure Washer",
        "category": "Garden & Outdoor Equipment",
        "price": "349.99 Eur",
        "features": [
            "Pressure: 20-145 bar", "Flow rate: max. 500 l/h", "Motor: Water-cooled induction motor",
            "Hose length: 10 m", "Includes: Vario Power spray lance, Dirt Blaster spray lance",
            "Smart connectivity: Home & Garden app via Bluetooth"
        ],
        "description": "The Kärcher K 5 Power Control is the perfect high-pressure washer for tackling stubborn dirt on patios, cars, and stone walls. The water-cooled motor ensures long life and high performance. Easily adjust the pressure using the Vario Power lance or consult the mobile app for recommended settings for any cleaning task. A reliable choice for all medium-duty cleaning jobs around your home and garden."
    },
    {
        "id": "SNK002",
        "name": "Bosch Professional GSB 18V-55 Cordless Combi Drill",
        "category": "Power Tools",
        "price": "199.99 Eur",
        "features": [
            "Voltage: 18V", "Max. torque (hard/soft): 55 / 28 Nm", "Chuck capacity: 1.5 - 13 mm",
            "No-load speed (1st/2nd gear): 0 – 480 / 0 – 1,800 rpm",
            "Brushless motor for longer tool life and battery runtime",
            "Includes: 2x 2.0Ah batteries, charger, L-Case carrying case"
        ],
        "description": "A powerful and durable all-rounder from the Bosch Professional series. The GSB 18V-55 cordless combi drill with its brushless motor is perfect for screw-driving and drilling in wood, metal, and masonry. Its compact design allows for comfortable work in tight spaces, making it an essential tool for electricians, carpenters, and installers."
    },
    {
        "id": "SNK003",
        "name": "Sadolin Bindo 7 Deep Matt Interior Paint (10L)",
        "category": "Paint & Finishing",
        "price": "45.99 Eur",
        "features": [
            "Volume: 10 Liters", "Finish: Deep matt (non-reflective)",
            "Color: Base A (can be tinted to thousands of shades)", "Coverage: up to 10 m²/l",
            "High wash resistance (Class 1)",
            "Application: Walls and ceilings in living rooms, bedrooms, and offices"
        ],
        "description": "Sadolin Bindo 7 is a high-quality, water-based acrylic paint for interior surfaces. It creates a beautiful, deep matt finish that hides minor surface imperfections. Its excellent wash resistance makes it easy to clean, ensuring your walls look fresh for years. Ideal for use on plaster, concrete, gypsum board, and previously painted surfaces."
    },
    {
        "id": "SNK004",
        "name": "Classen Oak Laminate Flooring (8mm, Class 32)",
        "category": "Flooring & Tiles",
        "price": "14.99 Eur / m²",
        "features": [
            "Thickness: 8 mm",
            "Abrasion Class: AC4 / Class 32 (suitable for heavy domestic and moderate commercial use)",
            "Plank size: 1286 x 194 mm", "Locking system: Megaloc for easy, glueless installation",
            "V-groove on all 4 sides for a realistic plank look", "Pack size: 2.245 m²"
        ],
        "description": "Achieve a timeless and durable wood-look with Classen laminate flooring in a classic oak finish. The 8mm thickness and Class 32 rating make it suitable for high-traffic areas like hallways and living rooms. The Megaloc click system allows for fast and easy installation, making it a perfect weekend DIY project."
    },
    {
        "id": "SNK005",
        "name": "Weber Master-Touch GBS E-5750 Charcoal Grill (57cm)",
        "category": "Grills & Outdoor Cooking",
        "price": "379.00 Eur",
        "features": [
            "Diameter: 57 cm", "Gourmet BBQ System (GBS) hinged cooking grate",
            "One-Touch cleaning system with enclosed ash catcher", "Lid-integrated thermometer",
            "Tuck-Away lid holder", "All-weather wheels for easy transport"
        ],
        "description": "The iconic Weber Master-Touch kettle grill is the gold standard in charcoal grilling. The Gourmet BBQ System grate allows you to use a variety of inserts like a pizza stone or wok. Features like the One-Touch cleaning system and integrated thermometer make grilling simple and convenient, delivering perfect results every time."
    },
    {
        "id": "SNK006",
        "name": "Kärcher K 6 Power Control High-Pressure Washer",
        "category": "Garden & Outdoor Equipment",
        "price": "319.99 Eur",
        "features": [
            "Pressure: 20-145 bar", "Flow rate: max. 600 l/h", "Motor: Water-cooled induction motor",
            "Hose length: 10 m", "Includes: Vario Power spray lance, Dirt Blaster spray lance",
            "Smart connectivity: Home & Garden app via Bluetooth"
        ],
        "description": "The Kärcher K 5 Power Control is the perfect high-pressure washer for tackling stubborn dirt on patios, cars, and stone walls. The water-cooled motor ensures long life and high performance. Easily adjust the pressure using the Vario Power lance or consult the mobile app for recommended settings for any cleaning task. A reliable choice for all medium-duty cleaning jobs around your home and garden."
    },
    {
        "id": "SNK007",
        "name": "Kärcher T 7 Plus T-Racer Surface Cleaner",
        "category": "Garden & Outdoor Equipment",
        "price": "49.99 Eur",
        "features": [
            "Compatible with Kärcher K 5 and K 6 Power Washers",
            "Dual-rotating high-pressure jets for fast, streak-free cleaning",
            "Integrated rinsing function",
            "Ideal for patios and driveways"
        ],
        "description": "The T 7 Plus T-Racer drastically reduces cleaning time for large flat surfaces and delivers impeccable results."
    },
    {
        "id": "SNK008",
        "name": "Kärcher FJ 6 Foam Nozzle",
        "category": "Garden & Outdoor Equipment",
        "price": "18.99 Eur",
        "features": [
            "Wide coverage foam application",
            "Quick connect for Kärcher pressure washers",
            "Adjustable foam concentration",
            "Suitable for car washing"
        ],
        "description": "Apply detergents easily for deep cleaning of cars, bikes, and windows with this foam sprayer attachment."
    },
    {
        "id": "SNK009",
        "name": "Kärcher 3-in-1 Stone & Paving Cleaner",
        "category": "Cleaning Chemicals",
        "price": "11.99 Eur",
        "features": [
            "Special formula for stone, patio, brick, and concrete",
            "Compatible with all Kärcher pressure washers",
            "Powerful dirt removal and long-lasting protection"
        ],
        "description": "This detergent provides thorough cleaning and protection for outdoor hard surfaces."
    },
    {
        "id": "SNK010",
        "name": "Bosch Professional Drill Bit Set (7 pcs)",
        "category": "Power Tools",
        "price": "24.99 Eur",
        "features": [
            "Includes bits for wood, masonry, and metal",
            "Optimal for cordless drills",
            "Robust storage case",
            "High durability"
        ],
        "description": "Universal drill bit set for all major tasks, perfect companion for your Bosch GSB 18V-55."
    },
    {
        "id": "SNK011",
        "name": "Safety Glasses (Anti-Fog, Impact Resistant)",
        "category": "Safety Equipment",
        "price": "7.99 Eur",
        "features": [
            "Anti-fog coating",
            "Wrap-around protection",
            "Suitable for power tool work"
        ],
        "description": "Protect your eyes during drilling or cutting jobs with comfortable, clear-lens safety glasses."
    },
    {
        "id": "SNK012",
        "name": "Work Gloves (Grip & Cut Resistant)",
        "category": "Safety Equipment",
        "price": "9.99 Eur",
        "features": [
            "Size: Universal fit",
            "Textured palm for enhanced grip",
            "Protects against splinters and scrapes"
        ],
        "description": "Durable gloves providing grip, dexterity, and protection for all general tool use."
    },
    {
        "id": "SNK013",
        "name": "Premium Synthetic Paint Brush (50 mm)",
        "category": "Paint & Finishing",
        "price": "5.99 Eur",
        "features": [
            "Suitable for acrylic paints and varnishes",
            "Comfort-grip handle",
            "Medium width for walls and trim"
        ],
        "description": "High-quality bristles for a smooth, professional finish on all wall surfaces."
    },
    {
        "id": "SNK014",
        "name": "Reusable Paint Tray (Large)",
        "category": "Paint & Finishing",
        "price": "6.49 Eur",
        "features": [
            "Fits rollers up to 25 cm",
            "Heavy duty plastic, easy to clean"
        ],
        "description": "Paint tray for even distribution and mess-free reloads with rollers or brushes."
    },
    {
        "id": "SNK015",
        "name": "Wall Primer (Universal, 2.5L)",
        "category": "Paint & Finishing",
        "price": "18.99 Eur",
        "features": [
            "Ensures better paint adhesion",
            "Seals porous surfaces",
            "Quick-drying formula"
        ],
        "description": "Primer for prepping interior walls, improving paint coverage and durability."
    },
    {
        "id": "SNK016",
        "name": "Underlayment for Laminate Flooring (5mm, Roll)",
        "category": "Flooring Accessories",
        "price": "19.99 Eur",
        "features": [
            "Enhances acoustic comfort",
            "Provides moisture barrier",
            "Easy-to-install roll format"
        ],
        "description": "Perfect base layer for laminate flooring to reduce sound and prevent moisture."
    },
    {
        "id": "SNK017",
        "name": "Laminate Installation Kit",
        "category": "Flooring Accessories",
        "price": "15.99 Eur",
        "features": [
            "Includes spacers, pull bar, and tapping block",
            "Speeds up installation and prevents damage"
        ],
        "description": "Essential tools for a professional, easy laminate installation."
    }
]

df = pd.DataFrame(product_catalog)




def create_embedding_text(row):
    features_str = ", ".join(row['features'])
    return (
        f"Product Name: {row['name']}. {row['name']}. "
        f"Category: {row['category']}. "
        f"Price: {row['price']}. "
        f"Key Features: {features_str}. "
        f"Description: {row['description']}"
    )

df['combined_features'] = df.apply(create_embedding_text, axis=1)
print("Prepared data for embedding.")



INDEX_NAME = "senukai-chatbot-mvp"
EMBEDDING_DIM = 384

pc = Pinecone(api_key=PINECONE_API_KEY)

if INDEX_NAME not in pc.list_indexes().names():
    print(f"Creating new index: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIM,
        metric="cosine", 
        spec=ServerlessSpec(
            cloud='aws', 
            region='us-east-1'
        )
    )
else:
    print(f"Index '{INDEX_NAME}' already exists.")

index = pc.Index(INDEX_NAME)
print(f"Connected to index: {INDEX_NAME}")
index.describe_index_stats()

print("Initializing embedding model...")
embeddings_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

print("Generating embeddings and upserting data to Pinecone...")
batch_size = 32
for i in tqdm(range(0, len(df), batch_size)):
    i_end = min(i + batch_size, len(df))
    batch = df.iloc[i:i_end]

    texts_to_embed = batch['combined_features'].tolist()
    embeddings = embeddings_model.embed_documents(texts_to_embed)

    ids = batch['id'].tolist()
    metadata = batch.to_dict('records')

    index.upsert(vectors=zip(ids, embeddings, metadata))

print("\nData indexing complete!")
index.describe_index_stats()