# README.md (Face Verification API + Amazon Laptop Scraper  )

A collection of two Python-based projects demonstrating practical applications of **FastAPI**, **computer vision**, **web scraping**, and **data processing**.

This repository contains:

- **Project 1:** Face Verification API (FastAPI + FaceNet)  
- **Project 2:** Amazon Laptop Scraper (Requests + BeautifulSoup + Selenium optional)  
- `requirements.txt` for all dependencies  
- Simple front-end HTML page for testing the API  
- Easy-to-follow instructions for running both projects  

---

# **Project 1: Face Verification API (FastAPI + keras-facenet)**

This project provides an API to compare two face images and determine whether they belong to the same person. It uses:

- **MTCNN** for face detection  
- **keras-facenet** (FaceNet) for embedding extraction  
- **Cosine similarity** to compare faces  

### Features
- Detects faces from uploaded images  
- Extracts normalized FaceNet embeddings  
- Returns:
  - Similarity score
  - Whether images belong to the same person
  - Bounding boxes + detection probabilities

### How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the FastAPI server:

   ```bash
   uvicorn main:app --reload
   ```

3. Open Swagger Docs:

   ```
   http://localhost:8000/docs
   ```

4. Test using the included `index.html`:

   * Located in `frontend/` folder
   * Open in browser and upload two images

### Sample API Response

```json
{
  "verification_result": "Same person",
  "similarity_score": 0.82,
  "threshold_used": 0.7
}
```

---

# **Project 2: Amazon Laptop Scraper**

A Python script that scrapes Amazon.in search results for **laptop listings**, extracting:

* ASIN
* Product name
* Price
* Rating
* Image URL
* Product URL
* Whether result is Sponsored or Organic

### Tech Used

* `requests`
* `beautifulsoup4`
* `lxml`
* `pandas`
* `tqdm`
* Optional: `selenium` + `webdriver-manager`

### How to Run

Basic use:

```bash
python scrape.py
```

Scrape a specific number of pages:

```bash
python scrape.py --pages 5
```

Force Selenium mode:

```bash
python scrape.py --selenium
```

Choose custom output directory:

```bash
python scrape.py --out scraped_data
```

Output CSV stored automatically with a timestamp.

---

# **Requirements**

All required dependencies for both projects are listed in `requirements.txt`.

---
